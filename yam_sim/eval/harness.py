"""Eval harness: run a (model x task x N-episode) job matrix and aggregate results.

The harness connects to one or more already-running openpi websocket servers
(one per checkpoint), rolls out each task with the batched ``mjwarp`` env so that
``num_worlds`` episodes run in parallel, records one video per episode, and writes
crash-safe per-episode results plus an aggregated summary.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from yam_sim.eval.config import VISUAL_SALT, EvalConfig, ModelEndpoint, TaskSpecEntry
from yam_sim.eval.failure_injection import build_injector_factory
from yam_sim.eval.results import (
    EpisodeResult,
    aggregate,
    append_jsonl,
    completed_keys,
    format_table,
    load_jsonl,
    write_summary,
)
from yam_sim.eval.rollout import (
    per_world_frames,
    prepare_policy_obs,
    slice_execute_actions,
    ttrtc_prefix,
)
from yam_sim.eval.video import EpisodeVideoWriters

# Early-stop tuning (config.early_stop_on_success): success must hold this long
# before a world counts as done, and saved qpos keeps this much tail past the
# success moment so offline renders end just after the placement.
_SUCCESS_HOLD_S = 0.5
_POST_SUCCESS_MARGIN_S = 1.0


def _slice_metrics(info: dict[str, Any], world: int, num_worlds: int) -> dict[str, Any]:
    """Extract per-world metric values from a ``TaskEvalResult.to_info`` dict."""
    metrics: dict[str, Any] = {}
    for key, value in info.items():
        if key in ("reward", "success"):
            continue
        if isinstance(value, list) and len(value) == num_worlds:
            metrics[key] = value[world]
        else:
            metrics[key] = value
    return metrics


def _resolve_max_chunks(config: EvalConfig, entry: TaskSpecEntry, env: Any, execute_dim: int) -> int:
    """Per-episode chunk horizon from the env's physics (max_seconds -> chunks)."""
    base = getattr(env, "base_env", env)
    physics_dt = float(getattr(base, "_physics_dt", 0.002))
    control_decimation = int(getattr(base, "_control_decimation", 17))
    seconds_per_chunk = execute_dim * physics_dt * control_decimation
    return config.horizon_chunks(entry, seconds_per_chunk)


def _run_one_batch(
    config: EvalConfig,
    model: ModelEndpoint,
    policy: Any,
    env: Any,
    results_path: Path,
    done: set[tuple[str, str, int | None]],
    new_results: list[EpisodeResult],
    *,
    task_name: str,
    prompt: str,
    task_dir: Path,
    execute_dim: int,
    prefix_len: int,
    max_chunks: int,
    num_worlds: int,
    max_seed_exclusive: int,
    batch_seed_base: int,
    eval_mode: str,
    label: str,
    injector_factory: Any = None,
    injection_attempts: dict[int, int] | None = None,
    max_injection_attempts: int = 0,
) -> None:
    """Reset one batch (all worlds), roll it out, score, and record each episode.

    Shared by the standard multi-batch loop and the grouped visual-variety loop.
    Worlds whose seed is out of range or already recorded are skipped.
    ``injector_factory``, when set, builds a per-batch failure injector that may
    mutate each step's actions in place (see eval/failure_injection.py).
    """
    obs, info = env.reset(seed=batch_seed_base)
    injector = injector_factory(env) if injector_factory is not None else None

    base = getattr(env, "base_env", env)
    control_dt = float(getattr(base, "_physics_dt", 0.002)) * int(
        getattr(base, "_control_decimation", 17)
    )
    # Early stop: a world counts as done once task success has held for
    # SUCCESS_HOLD_S (filters transient in-bin bounces); the batch stops when
    # every recorded world is done. Saved qpos is truncated per world at
    # success + POST_SUCCESS_MARGIN_S so offline renders end at the placement.
    success_hold = max(1, round(_SUCCESS_HOLD_S / control_dt))
    success_step = np.full(num_worlds, -1, dtype=np.int64)
    success_run = np.zeros(num_worlds, dtype=np.int64)
    global_step = 0
    world_seeds = info["world_seeds"]
    # The deterministic visual config for this batch (None outside visual-group mode);
    # identical for every episode in the group and accurately reflects what is rendered
    # (the construction model), unlike per-world re-sampled colors.
    visual_seed = info.get("visual_seed")
    visual_config = info.get("visual_config")
    gs = config.visual_group_size

    record_worlds = []
    for w in range(num_worlds):
        s = world_seeds[w]
        if s is None or int(s) >= max_seed_exclusive:
            continue
        if (model.label, task_name, int(s)) in done:
            continue
        record_worlds.append(w)
    if not record_worlds:
        return

    writers = None
    if config.video:
        video_paths = [task_dir / f"seed_{world_seeds[w]}.mp4" for w in record_worlds]
        writers = EpisodeVideoWriters(video_paths, fps=config.fps)
        writers.append(per_world_frames(obs, env.camera_names)[record_worlds])

    init_q = np.asarray(obs["state"], dtype=np.float32)
    previous_chunk_actions = None
    executed_actions: list[np.ndarray] = []
    # Hold-pose command for worlds excluded from inference (finished early or
    # never recorded): policy state and actions share the 14D position space,
    # so the last executed action (init pose at first) holds the arm in place.
    last_actions = init_q.copy()

    qpos_frames: list[np.ndarray] = []
    # Mocap pose (e.g. the mug_flip tray) is held in per-world data, not qpos, and is
    # static within an episode -- capture it once at frame 0 so the offline rerender can
    # place the fixture at this episode's pose (qpos replay alone would lose it).
    mocap0 = env.mocap_batch() if config.save_states else None
    if config.save_states:
        qpos_frames.append(env.qpos_batch())

    for _chunk_idx in range(max_chunks):
        # Inference runs only for worlds that still need policy control: the
        # openpi server processes one observation per call, so skipping
        # finished/unrecorded worlds is the dominant wall-clock saving (a 90 s
        # straggler no longer drags 15 finished worlds through inference).
        if config.early_stop_on_success:
            active = [w for w in record_worlds if success_step[w] < 0]
        else:
            active = list(record_worlds)
        if not active:
            break
        sliced = len(active) != num_worlds
        if sliced:
            infer_obs = {
                "state": np.asarray(obs["state"])[active],
                "images": {
                    k: np.asarray(v)[active] for k, v in obs.get("images", {}).items()
                },
            }
        else:
            infer_obs = obs
        policy_obs = prepare_policy_obs(
            infer_obs, prompt=prompt, jpeg_quality=config.jpeg_quality
        )
        action_prefix = None
        prefix_length = None
        if config.use_ttrtc:
            action_prefix, prefix_length = ttrtc_prefix(
                previous_chunk_actions, init_q, prefix_len, is_batched=True
            )
            if sliced and action_prefix is not None:
                action_prefix = action_prefix[active]

        result = policy.infer(
            policy_obs, action_prefix=action_prefix, prefix_length=prefix_length
        )
        predicted = np.asarray(result["actions"], dtype=np.float32)
        if predicted.ndim == 2:  # (T, 14) -> (1, T, 14)
            predicted = predicted[None, ...]
        if sliced:
            full = np.repeat(last_actions[:, None, :], predicted.shape[1], axis=1)
            full[active] = predicted
            predicted = full

        execute_actions = slice_execute_actions(
            predicted,
            prefix_length=prefix_length,
            execute_dim=execute_dim,
            use_ttrtc=config.use_ttrtc,
            is_batched=True,
        )
        executed_actions.append(execute_actions)

        n_steps = execute_actions.shape[1]
        for step_idx in range(n_steps):
            if injector is not None:
                injector.before_step(
                    env,
                    execute_actions[:, step_idx, :],
                    chunk_idx=_chunk_idx,
                    step_idx=step_idx,
                )
            render_obs = writers is not None or step_idx == n_steps - 1
            obs, _, _, _, step_info = env.step(
                execute_actions[:, step_idx, :], render_obs=render_obs
            )
            if config.save_states:
                qpos_frames.append(env.qpos_batch())
            if writers is not None:
                writers.append(per_world_frames(obs, env.camera_names)[record_worlds])
            if config.early_stop_on_success:
                succ = step_info.get("task_success")
                if succ is not None:
                    succ = np.asarray(succ, dtype=bool)
                    success_run = np.where(succ, success_run + 1, 0)
                    newly = (success_step < 0) & (success_run >= success_hold)
                    success_step[newly] = global_step
            global_step += 1

        previous_chunk_actions = execute_actions
        last_actions = execute_actions[:, -1, :].copy()
        if config.early_stop_on_success and all(
            success_step[w] >= 0 for w in record_worlds
        ):
            break

    if writers is not None:
        writers.close()

    task_eval = env.evaluate_task()
    info_dict = task_eval.to_info(squeeze=False) if task_eval is not None else {}
    all_actions = np.concatenate(executed_actions, axis=1)  # (W, T, 14)
    injector_records = injector.finalize() if injector is not None else None

    for w in record_worlds:
        seed = world_seeds[w]
        success = bool(task_eval.success[w]) if task_eval is not None else None
        reward = float(task_eval.reward[w]) if task_eval is not None else None
        metrics = _slice_metrics(info_dict, w, num_worlds) if task_eval is not None else {}
        if injector_records is not None:
            metrics.update(injector_records[w])
        w_success_step = int(success_step[w]) if success_step[w] >= 0 else None
        if w_success_step is not None:
            # The episode ended (for scoring) at sustained success; the world keeps
            # simulating only to serve the rest of the batch.
            success = True
        end_step = global_step
        if w_success_step is not None:
            end_step = min(
                global_step,
                w_success_step + max(1, round(_POST_SUCCESS_MARGIN_S / control_dt)),
            )
        metrics["success_step"] = w_success_step
        metrics["end_step"] = int(end_step)
        metrics["end_time_s"] = float(end_step * control_dt)

        # Injection-retry: a failed injection (release fell in the bin / never
        # left the gripper) is not recorded while attempts remain -- the seed
        # stays un-done, and a later packed retry round re-rolls it with fresh
        # policy sampling on the identical scene.
        if (
            injector_records is not None
            and injection_attempts is not None
            and seed is not None
            and injector_records[w].get("slip_outcome")
            in ("accidental_bin", "not_released")
            and injection_attempts.get(int(seed), 0) < max_injection_attempts
        ):
            injection_attempts[int(seed)] = injection_attempts.get(int(seed), 0) + 1
            print(
                f"    retry seed {seed}: {injector_records[w]['slip_outcome']} "
                f"(attempt {injection_attempts[int(seed)]}/{max_injection_attempts + 1})"
            )
            continue
        if injector_records is not None:
            metrics["injection_attempt"] = (
                injection_attempts.get(int(seed), 0) + 1
                if (injection_attempts is not None and seed is not None)
                else 1
            )

        actions_path = task_dir / f"seed_{seed}_actions.npy"
        actions_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(actions_path), all_actions[w])

        states_path = None
        if config.save_states and qpos_frames:
            states = np.stack([frame[w] for frame in qpos_frames], axis=0)
            # +1: frame 0 is the reset state, steps land at index step+1.
            states = states[: end_step + 1]
            states_path = task_dir / f"seed_{seed}_qpos.npy"
            np.save(str(states_path), states)
            if mocap0 is not None:
                # frame-0 mocap pose for this world (tray is static within the episode)
                np.savez(
                    str(task_dir / f"seed_{seed}_mocap.npz"),
                    mocap_pos=mocap0[0][w],
                    mocap_quat=mocap0[1][w],
                )

        video_path = str(task_dir / f"seed_{seed}.mp4") if config.video else None
        visual_group_index = (int(seed) // gs) if (gs and seed is not None) else None
        episode_result = EpisodeResult(
            model_label=model.label,
            task=task_name,
            seed=int(seed) if seed is not None else None,
            success=success,
            reward=reward,
            num_chunks=max_chunks,
            metrics=metrics,
            video_path=video_path,
            actions_path=str(actions_path),
            states_path=str(states_path) if states_path is not None else None,
            eval_mode=eval_mode,
            scene_config=visual_config,
            visual_seed=visual_seed,
            visual_group_index=visual_group_index,
        )
        append_jsonl(results_path, episode_result)
        done.add((model.label, task_name, episode_result.seed))
        new_results.append(episode_result)

    scored = [r for r in new_results if r.success is not None]
    succ = sum(int(r.success) for r in scored)  # type: ignore[arg-type]
    sr = f"{succ}/{len(scored)}" if scored else "n/a"
    print(f"    {label}: recorded {len(record_worlds)} (cum succ {sr})")


def _run_task_for_model(
    config: EvalConfig,
    model: ModelEndpoint,
    entry: TaskSpecEntry,
    policy: Any,
    results_path: Path,
    done: set[tuple[str, str, int | None]],
) -> list[EpisodeResult]:
    """Roll out one task for one model across all requested episodes."""
    import yam_sim

    spec = entry.resolve()
    task_name = spec.name
    env_task = spec.env_task
    prompt = spec.prompt
    episodes = config.episodes_for(entry)
    num_worlds = config.num_worlds_for(entry)
    num_batches = math.ceil(episodes / num_worlds)

    task_dir = Path(config.output_dir) / model.label / task_name

    # Pre-skip: if every requested episode is already recorded, do nothing.
    wanted_seeds = [config.seed_base + i for i in range(episodes)]
    if all((model.label, task_name, s) in done for s in wanted_seeds):
        print(f"  [{model.label} / {task_name}] all {episodes} episodes present; skipping")
        return []

    # `execute_chunk_dim` is the number of predicted steps actually executed per
    # inference cycle, equivalent to the realtime ActionChunkBroker's
    # `action_horizon` (robots_realtime `sync` mode). With TTRTC off this is a
    # hard chunk swap with no prefix (matches deployment); with TTRTC on the same
    # count is executed but a `prefix_length`-step time-travel prefix is sent.
    execute_dim = config.execute_chunk_dim
    prefix_len = config.prefix_length if config.use_ttrtc else 0

    max_seed_exclusive = config.seed_base + episodes
    gs = config.visual_group_size
    new_results: list[EpisodeResult] = []

    injector_factory = build_injector_factory(config.failure_injection)

    if gs:
        # Grouped visual-variety eval: rebuild the env once per group with a
        # deterministic visual config (visual_seed = VISUAL_SALT + global_group);
        # poses still vary per episode within the group. Sharding must be group
        # aligned so each shard covers whole groups -> scenes are a pure function of
        # the global seeds, identical across policies/shardings.
        if num_worlds != gs:
            raise ValueError(
                f"visual_group_size={gs} requires num_worlds == visual_group_size "
                f"(got num_worlds={num_worlds})"
            )
        if episodes % gs != 0 or config.seed_base % gs != 0:
            raise ValueError(
                f"visual_group_size={gs} requires episodes ({episodes}) and seed_base "
                f"({config.seed_base}) to be multiples of {gs}"
            )
        eval_mode = f"batched_visual_group{gs}"
        n_groups = episodes // gs
        print(
            f"  [{model.label} / {task_name}] {episodes} episodes, visual-group eval "
            f"({n_groups} groups x {gs} eps; one visual config per group, env rebuilt per group)"
        )
        for group_local in range(n_groups):
            batch_seed_base = config.seed_base + group_local * gs
            global_group = batch_seed_base // gs
            group_seeds = [batch_seed_base + j for j in range(gs)]
            if all((model.label, task_name, s) in done for s in group_seeds):
                continue
            env = yam_sim.make_batched_env(
                scene=config.scene,
                task=env_task,
                prompt=prompt,
                num_worlds=gs,
                camera_backend=config.camera_backend,
                camera_gpu_id=config.camera_gpu_id,
                force_object_count=config.force_object_count,
                visual_seed=VISUAL_SALT + global_group,
                extra_reset_options=config.reset_options,
                bin_stabilize=config.bin_stabilize,
            )
            try:
                max_chunks = _resolve_max_chunks(config, entry, env, execute_dim)
                _run_one_batch(
                    config, model, policy, env, results_path, done, new_results,
                    task_name=task_name, prompt=prompt, task_dir=task_dir,
                    execute_dim=execute_dim, prefix_len=prefix_len, max_chunks=max_chunks,
                    num_worlds=gs, max_seed_exclusive=max_seed_exclusive,
                    batch_seed_base=batch_seed_base, eval_mode=eval_mode,
                    label=f"group {global_group} (visual_seed {VISUAL_SALT + global_group})",
                    injector_factory=injector_factory,
                )
            finally:
                env.close()
        return new_results

    # Standard path: one env for the whole task, looped over batches of num_worlds.
    # fixed_visual_seed pins ONE visual config for the whole run (any seed_base /
    # num_worlds), so the run can be sharded by position across GPUs while staying
    # scene-identical to a visual_group_size single-batch run.
    env = yam_sim.make_batched_env(
        scene=config.scene,
        task=env_task,
        prompt=prompt,
        num_worlds=num_worlds,
        camera_backend=config.camera_backend,
        camera_gpu_id=config.camera_gpu_id,
        mask_variable_count=config.mask_variable_count,
        # mug_inline pins the inline colored mugs (hang_mug) and supersedes force count.
        force_object_count=None if config.mug_inline else config.force_object_count,
        mug_inline=config.mug_inline,
        visual_seed=config.fixed_visual_seed,
        extra_reset_options=config.reset_options,
        bin_stabilize=config.bin_stabilize,
    )
    if config.mug_inline:
        eval_mode = "batched_inline_mugs"
    elif config.mask_variable_count:
        eval_mode = "batched_masked_count"
    elif config.fixed_visual_seed is not None:
        eval_mode = f"batched_fixed_visual{int(config.fixed_visual_seed)}"
    elif config.force_object_count is not None:
        eval_mode = f"batched_fixed_count_{int(config.force_object_count)}"
    else:
        eval_mode = "batched"

    max_chunks = _resolve_max_chunks(config, entry, env, execute_dim)
    print(
        f"  [{model.label} / {task_name}] {episodes} episodes, "
        f"{num_worlds} worlds/batch, {num_batches} batches, "
        f"{max_chunks} chunks/episode (execute {execute_dim})"
    )

    # Batches are packed from the not-yet-recorded seeds (identical to the old
    # contiguous windows on a fresh run, and efficient on resume). With
    # retry_failed_injection > 0, later rounds re-pack seeds whose injection
    # attempt was skipped by _run_one_batch; short groups are padded with
    # out-of-range seeds that roll out but never record.
    max_injection_attempts = (
        int(config.retry_failed_injection) if injector_factory is not None else 0
    )
    injection_attempts: dict[int, int] = {}
    try:
        for round_idx in range(1 + max_injection_attempts):
            missing = [
                s for s in wanted_seeds if (model.label, task_name, s) not in done
            ]
            if not missing:
                break
            if round_idx > 0:
                print(
                    f"  [{model.label} / {task_name}] injection-retry round "
                    f"{round_idx}: {len(missing)} seed(s) {missing}"
                )
            groups = [
                missing[i : i + num_worlds] for i in range(0, len(missing), num_worlds)
            ]
            for group_idx, group in enumerate(groups):
                pad = [
                    max_seed_exclusive + 1_000_000 + k
                    for k in range(num_worlds - len(group))
                ]
                _run_one_batch(
                    config, model, policy, env, results_path, done, new_results,
                    task_name=task_name, prompt=prompt, task_dir=task_dir,
                    execute_dim=execute_dim, prefix_len=prefix_len, max_chunks=max_chunks,
                    num_worlds=num_worlds, max_seed_exclusive=max_seed_exclusive,
                    batch_seed_base=list(group) + pad, eval_mode=eval_mode,
                    label=f"round {round_idx} batch {group_idx}",
                    injector_factory=injector_factory,
                    injection_attempts=injection_attempts,
                    max_injection_attempts=max_injection_attempts,
                )
    finally:
        env.close()

    return new_results


def run_eval(config: EvalConfig) -> dict[str, Any]:
    """Execute the full eval job matrix and write/print aggregated results."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    existing = load_jsonl(results_path)
    done = completed_keys(existing)
    if existing:
        print(f"Resuming: {len(existing)} episodes already recorded in {results_path}")

    for model in config.models:
        print(f"Model '{model.label}' @ {model.host}:{model.port}")
        from yam_sim.policy.openpi_policy import OpenPIPolicy

        policy = OpenPIPolicy(
            host=model.host,
            port=model.port,
            api_key=model.api_key,
            use_batch_infer=config.batched_inference,
        )
        for entry in config.tasks:
            _run_task_for_model(config, model, entry, policy, results_path, done)

    all_results = load_jsonl(results_path)
    summary = aggregate(all_results)
    json_path, csv_path = write_summary(output_dir, summary)
    print("\n" + format_table(summary))
    print(f"\nSummary written to {json_path} and {csv_path}")
    return summary
