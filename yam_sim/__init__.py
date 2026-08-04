"""yam-sim: Standalone MuJoCo simulation for the YAM bimanual robot."""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "MuJoCoYAMEnv":
        from yam_sim.env import MuJoCoYAMEnv

        return MuJoCoYAMEnv

    if name == "BatchedWarpYAMEnv":
        from yam_sim.batched_env import BatchedWarpYAMEnv

        return BatchedWarpYAMEnv

    if name in {"get_i2rt_sim_config", "get_i2rt_config", "RobotSystemConfig"}:
        from yam_sim.config import RobotSystemConfig, get_i2rt_config, get_i2rt_sim_config

        return {
            "get_i2rt_sim_config": get_i2rt_sim_config,
            "get_i2rt_config": get_i2rt_config,
            "RobotSystemConfig": RobotSystemConfig,
        }[name]

    if name in {"RandomizationState", "TASK_RANDOMIZERS"}:
        from yam_sim.randomization import RandomizationState, TASK_RANDOMIZERS

        return {
            "RandomizationState": RandomizationState,
            "TASK_RANDOMIZERS": TASK_RANDOMIZERS,
        }[name]

    if name in {"SimTaskSpec", "get_task_spec", "list_task_specs", "maybe_get_task_spec"}:
        from yam_sim.task_specs import (
            SimTaskSpec,
            get_task_spec,
            list_task_specs,
            maybe_get_task_spec,
        )

        return {
            "SimTaskSpec": SimTaskSpec,
            "get_task_spec": get_task_spec,
            "list_task_specs": list_task_specs,
            "maybe_get_task_spec": maybe_get_task_spec,
        }[name]

    if name in {"TaskEvalResult", "TaskEvaluator", "make_task_evaluator"}:
        from yam_sim.task_eval import TaskEvalResult, TaskEvaluator, make_task_evaluator

        return {
            "TaskEvalResult": TaskEvalResult,
            "TaskEvaluator": TaskEvaluator,
            "make_task_evaluator": make_task_evaluator,
        }[name]

    if name in {
        "CollectionTaskGroup",
        "DATA_COLLECTION_TASKS",
        "get_data_collection_task",
        "list_data_collection_tasks",
        "maybe_get_data_collection_task",
    }:
        from yam_sim.collection_tasks import (
            CollectionTaskGroup,
            DATA_COLLECTION_TASKS,
            get_data_collection_task,
            list_data_collection_tasks,
            maybe_get_data_collection_task,
        )

        return {
            "CollectionTaskGroup": CollectionTaskGroup,
            "DATA_COLLECTION_TASKS": DATA_COLLECTION_TASKS,
            "get_data_collection_task": get_data_collection_task,
            "list_data_collection_tasks": list_data_collection_tasks,
            "maybe_get_data_collection_task": maybe_get_data_collection_task,
        }[name]

    if name in {
        "DEFAULT_SCENE_XML",
        "ResolvedTask",
        "SCENE_XMLS",
        "SceneTaskSpec",
        "get_scene_task_spec",
        "get_task_physics_defaults",
        "get_task_randomizer",
        "get_task_scene_xml",
        "list_scene_task_names",
        "list_scene_tasks",
        "maybe_get_scene_task_spec",
        "resolve_env_task_name",
        "resolve_task",
    }:
        from yam_sim.task_registry import (
            DEFAULT_SCENE_XML,
            ResolvedTask,
            SCENE_XMLS,
            SceneTaskSpec,
            get_scene_task_spec,
            get_task_physics_defaults,
            get_task_randomizer,
            get_task_scene_xml,
            list_scene_task_names,
            list_scene_tasks,
            maybe_get_scene_task_spec,
            resolve_env_task_name,
            resolve_task,
        )

        return {
            "DEFAULT_SCENE_XML": DEFAULT_SCENE_XML,
            "ResolvedTask": ResolvedTask,
            "SCENE_XMLS": SCENE_XMLS,
            "SceneTaskSpec": SceneTaskSpec,
            "get_scene_task_spec": get_scene_task_spec,
            "get_task_physics_defaults": get_task_physics_defaults,
            "get_task_randomizer": get_task_randomizer,
            "get_task_scene_xml": get_task_scene_xml,
            "list_scene_task_names": list_scene_task_names,
            "list_scene_tasks": list_scene_tasks,
            "maybe_get_scene_task_spec": maybe_get_scene_task_spec,
            "resolve_env_task_name": resolve_env_task_name,
            "resolve_task": resolve_task,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_env(
    scene: str = "hybrid",
    task: str = "bottles",
    render_cameras: bool = True,
    camera_backend: str = "mujoco",
    camera_gpu_id: int | None = None,
    prompt: str | None = None,
    chunk_dim: int = 30,
    wrist_fov: float = 58.0,
    scene_xml: str | Path | None = None,
    scene_xml_string: str | None = None,
    scene_xml_transform_options=None,
    enable_task_randomizer: bool = True,
    **kwargs,
) -> MuJoCoYAMEnv:
    """Create a MuJoCo YAM bimanual environment.

    Args:
        scene: Scene variant — "eval", "training", or "hybrid".
        task: Task scene — "bottles" (default), "inhand_transfer", etc.
        render_cameras: Whether to render camera images in observations.
        prompt: Task prompt string included in observations. Defaults to the
            resolved task prompt when available.
        chunk_dim: Number of timesteps per action chunk.
        wrist_fov: Vertical field-of-view in degrees for the wrist cameras
            ("left" and "right"). Default 58 matches the real D405 mounting.
        scene_xml: Optional explicit XML file path for standard scene tasks.
        scene_xml_string: Optional in-memory XML override for standard scene
            tasks. Mutually exclusive with ``scene_xml``.
        scene_xml_transform_options: Optional runtime XML transform options for
            model-swapping tasks such as ``inhand_transfer``.
        enable_task_randomizer: Whether to attach and prepare the task
            randomizer. Replay/export callers with a fully assembled scene XML
            should disable this so the randomizer cannot reload the model.
        **kwargs: Additional kwargs passed to MuJoCoYAMEnv.

    Returns:
        Configured MuJoCoYAMEnv instance with scene variant applied.
    """
    import mujoco

    from yam_sim.config import get_i2rt_sim_config
    from yam_sim.env import MuJoCoYAMEnv
    from yam_sim.scene_variants import apply_scene_variant
    from yam_sim.task_registry import (
        get_task_physics_defaults,
        get_task_randomizer,
        get_task_scene_xml,
        list_scene_task_names,
        resolve_task,
    )

    if scene_xml is not None and scene_xml_string is not None:
        raise ValueError("scene_xml and scene_xml_string are mutually exclusive")

    resolved_task = resolve_task(task)
    env_task = resolved_task.env_task or task
    resolved_prompt = prompt
    if resolved_prompt is None:
        if resolved_task.task_spec is not None:
            resolved_prompt = resolved_task.task_spec.prompt
        elif isinstance(task, str) and task:
            resolved_prompt = task.replace("_", " ")
        else:
            resolved_prompt = "fold the towel"

    if env_task == "inhand_transfer":
        import numpy as _np

        from yam_sim.randomization import (
            InHandTransferRandomizer,
            _INHAND_CATEGORIES,
            _OBJ_Z,
            _X_MAX,
            _X_MIN,
            _Y_LEFT_MAX,
            _Y_LEFT_MIN,
            _inhand_apply_scene_transforms,
            _inhand_build_xml,
            _inhand_get_variants,
        )

        if scene_xml is not None or scene_xml_string is not None:
            raise ValueError("scene_xml overrides are not supported for task='inhand_transfer'")

        config = get_i2rt_sim_config()
        seed = kwargs.pop("seed", None)
        rng = _np.random.default_rng(seed)

        # Pick initial object and generate scene XML.
        categories = _INHAND_CATEGORIES
        category = categories[int(rng.integers(0, len(categories)))]
        variants = _inhand_get_variants(category)
        variant_dir = variants[int(rng.integers(0, len(variants)))]
        x = float(rng.uniform(_X_MIN, _X_MAX))
        y = float(rng.uniform(_Y_LEFT_MIN, _Y_LEFT_MAX))
        yaw = float(rng.uniform(-_np.pi, _np.pi))
        xml = _inhand_build_xml(category, variant_dir, x, y, _OBJ_Z, yaw)
        xml = _inhand_apply_scene_transforms(xml, scene_xml_transform_options)

        physics_kwargs = {**kwargs, **get_task_physics_defaults(env_task)}
        env = MuJoCoYAMEnv(
            config=config,
            render_cameras=render_cameras,
            prompt=resolved_prompt,
            chunk_dim=chunk_dim,
            scene_xml=None,
            **physics_kwargs,
        )
        env.reload_from_xml(xml)
        env.set_task(task)
        env._inhand_category = category
        env._inhand_variant = variant_dir.name

        # Bind a randomizer so XdofSimNode can call it on every reset.
        randomizer = InHandTransferRandomizer(
            scene_variant=scene,
            scene_xml_transform_options=scene_xml_transform_options,
        )
        randomizer.bind_env(env)
        randomizer._rng = rng
        env._task_randomizer = randomizer

        apply_scene_variant(env.model, scene)
        return env

    scene_xml_path = Path(scene_xml) if scene_xml is not None else get_task_scene_xml(env_task)
    if scene_xml_path is None:
        raise ValueError(
            f"Unknown task '{task}'. Available: {list(list_scene_task_names())}"
        )

    tmp_scene_xml_path: Path | None = None
    if scene_xml_string is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=f".{env_task}_",
            dir=scene_xml_path.parent,
            delete=False,
        ) as handle:
            handle.write(scene_xml_string)
            tmp_scene_xml_path = Path(handle.name)
        scene_xml_path = tmp_scene_xml_path

    config = get_i2rt_sim_config()
    physics_kwargs = {**kwargs, **get_task_physics_defaults(env_task)}
    try:
        env = MuJoCoYAMEnv(
            config=config,
            render_cameras=render_cameras,
            camera_backend=camera_backend,
            camera_gpu_id=camera_gpu_id,
            prompt=resolved_prompt,
            chunk_dim=chunk_dim,
            scene_xml=scene_xml_path,
            **physics_kwargs,
        )
    finally:
        if tmp_scene_xml_path is not None:
            tmp_scene_xml_path.unlink(missing_ok=True)
    apply_scene_variant(env.model, scene)
    env.set_task(task)
    env._scene_xml_transform_options = scene_xml_transform_options
    if scene_xml_string is not None:
        env._scene_xml_string = scene_xml_string

    if wrist_fov != 58.0:
        import math

        # Wider FOV causes the frustum edges to clip into the gripper geometry.
        tan_baseline = math.tan(math.radians(58.0 / 2))
        tan_new = math.tan(math.radians(wrist_fov / 2))
        clearance_offset = max(0.0, (tan_new - tan_baseline) / tan_baseline) * 0.025
        for cam_name in ("left", "right"):
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id >= 0:
                env.model.cam_fovy[cam_id] = wrist_fov
                env.model.cam_pos[cam_id][1] -= clearance_offset

    base_randomizer = get_task_randomizer(env_task) if enable_task_randomizer else None
    if base_randomizer is not None:
        randomizer = base_randomizer.clone()
        randomizer.bind_env(env, scene_variant=scene)
        env._task_randomizer = randomizer
        prepare_env = getattr(randomizer, "prepare_env", None)
        if callable(prepare_env):
            prepare_env()

    return env


def make_batched_env(
    scene: str = "hybrid",
    task: str = "bottles",
    render_cameras: bool = True,
    camera_backend: str = "mjwarp",
    camera_gpu_id: int | None = None,
    prompt: str | None = None,
    chunk_dim: int = 30,
    num_worlds: int = 1,
    wrist_fov: float = 58.0,
    mask_variable_count: bool = False,
    force_object_count: int | None = None,
    visual_seed: int | None = None,
    mug_inline: bool = False,
    **kwargs,
):
    """Create a batched Warp-based YAM environment for GPU rollout.

    ``camera_backend`` selects the OBSERVATION renderer (physics always runs on
    mjwarp): ``"mjwarp"``/``"madrona"`` render all worlds on-GPU in one batched
    call (fast); ``"mujoco"`` renders each world's cameras with MuJoCo-GL from its
    qpos -- slower but matches the MuJoCo-GL renderer used to make the training
    data, so eval observations are in-distribution.

    ``force_object_count`` pins the object count to a FIXED value K for every world
    (e.g. put_bottles=4 in eval, matching throw_bottles) with a canonical variant+
    scale, while POSES still randomize per seed. Because the count is constant the
    model layout (nq) is constant, so no parking/masking is needed -- the live-
    snapshot path loads each world's randomized poses into the shared warp model.
    This is the eval default for the count-varying tasks (count is a training-time
    augmentation; eval uses the canonical count).

    ``mask_variable_count`` (alternative) lets the object COUNT itself vary per world
    by building the model at MAX count and *parking* unused object slots far below
    the scene via qpos (the trick the sweep task uses); evaluators then score only
    the non-parked (active) objects via a z-height threshold. Variant/scale are
    fixed-per-batch in both modes (a shared-model limit).

    ``visual_seed`` enables one deterministic VISUAL config (mesh variant + scale +
    color) for the whole batch: the construction reset is seeded by ``visual_seed``
    with variant+scale+color randomization ON, baking that config into the shared
    model; per-world resets then keep it frozen (``randomize_variants/scales=False``)
    and only vary poses. Because the warp runtime + MuJoCo-GL renderer are built once
    from the construction model (and per-world scene reloads merely rebind a separate
    ``base_env.model`` whose qpos we snapshot), variant/scale/color auto-freeze to the
    construction config across all worlds. Combine with ``force_object_count`` to also
    pin the count; used by the harness's grouped visual-variety eval (one visual_seed
    per group of episodes). Mutually exclusive with ``mask_variable_count``.
    """
    from yam_sim.batched_env import BatchedWarpYAMEnv

    if mask_variable_count and force_object_count is not None:
        raise ValueError("mask_variable_count and force_object_count are mutually exclusive")
    if mask_variable_count and visual_seed is not None:
        raise ValueError("mask_variable_count and visual_seed are mutually exclusive")

    base_env = make_env(
        scene=scene,
        task=task,
        render_cameras=False,
        camera_backend="mujoco",
        camera_gpu_id=camera_gpu_id,
        prompt=prompt,
        chunk_dim=chunk_dim,
        wrist_fov=wrist_fov,
        **kwargs,
    )

    mask_reset_options: dict | None = None
    if mask_variable_count:
        randomizer = getattr(base_env, "_task_randomizer", None)
        max_count = None
        for attr in ("max_bottle_count", "max_plate_count", "max_mug_count"):
            if randomizer is not None and hasattr(randomizer, attr):
                max_count = int(getattr(randomizer, attr))
                break
        if max_count is None:
            raise ValueError(
                f"mask_variable_count=True but task {task!r} has no variable-count "
                "randomizer (expected max_bottle_count/max_plate_count/max_mug_count)"
            )
        # Construction: build the shared warp model at the MAX object count with a
        # canonical (fixed) variant+scale. Each randomizer reads only its own count
        # key from this dict and ignores the others.
        force_max = {
            "randomize_variants": False,
            "randomize_scales": False,
            "bottle_count": max_count,
            "plate_count": max_count,
            "mug_count": max_count,
        }
        base_env.reset(seed=0, options={"randomization": force_max}, randomize=True)
        # Per-world: keep object COUNT natural (seed-driven) and let variant vary
        # (its mesh is discarded -- warp uses the canonical model; only poses are
        # transferred). Pin scale off so canonical-scale poses transfer without
        # penetrating the canonical-scale warp model. NOTE: randomize_variants must
        # stay on here -- setting it False would freeze the count to MAX
        # (see _resolve_*_count), defeating count randomization.
        mask_reset_options = {"randomize_scales": False}

    fixed_reset_options: dict | None = None
    visual_config: dict | None = None
    if mug_inline:
        # Inline-mug mode (hang_mug): keep the scene's inline colored mugs (geoms directly
        # on mug_1/2/3) instead of reloading variant assets, so the geometry-based reward
        # -- calibrated on these mugs -- attributes tree contacts correctly. Construction +
        # per-world resets are pose-only on the inline scene (no variant reload, no color
        # randomization); the fixture is frozen so it's consistent across the shared model.
        base_env.reset(seed=0, options={"randomization": {"use_inline_mugs": True}}, randomize=True)
        fixed_reset_options = {"use_inline_mugs": True}
    elif force_object_count is not None or visual_seed is not None:
        # Construction reset bakes the per-batch config into the shared warp model.
        # With visual_seed: seed it and turn variant/scale/color randomization ON so a
        # deterministic VISUAL config is chosen (auto-frozen for all worlds, since the
        # warp runtime + renderer use this construction model). Without visual_seed:
        # canonical (variant/scale OFF) -- the original fixed-count behaviour.
        # force_object_count pins count=K either way. Per world, randomize_variants=
        # False freezes the count + variant to the construction value while poses are
        # still sampled per seed -> constant nq, randomized poses.
        if visual_seed is not None:
            con_seed = int(visual_seed)
            con = {"randomize_variants": True, "randomize_scales": True}
        else:
            con_seed = 0
            con = {"randomize_variants": False, "randomize_scales": False}
        if force_object_count is not None:
            k = int(force_object_count)
            con.update({"bottle_count": k, "plate_count": k, "mug_count": k})
        fixed_reset_options = {"randomize_variants": False, "randomize_scales": False}
        base_env.reset(seed=con_seed, options={"randomization": con}, randomize=True)
        if task == "mug_flip":
            # mug_flip places its mugs RELATIVE to the tray. If the tray is a MOCAP body
            # its pose lives in per-world data.mocap_pos, so each world can sample its own
            # tray (and the batched env carries it per-world): leave it UNPINNED so the
            # single 100-world batch gets 100 distinct tray placements with mugs-on-tray.
            # If the tray is still a static body (model.body_pos is shared across worlds),
            # fall back to pinning every world to the construction tray pose T_g so the
            # mugs are placed relative to the same tray the renderer/warp shows.
            import mujoco as _mujoco

            tid = _mujoco.mj_name2id(base_env.model, _mujoco.mjtObj.mjOBJ_BODY, "tray")
            tray_is_mocap = tid >= 0 and int(base_env.model.body_mocapid[tid]) >= 0
            if tid >= 0 and not tray_is_mocap:
                fixed_reset_options["tray_pose"] = (
                    *(float(x) for x in base_env.model.body_pos[tid]),
                    *(float(x) for x in base_env.model.body_quat[tid]),
                )
        if visual_seed is not None and getattr(base_env, "_last_randomization", None) is not None:
            rs = base_env._last_randomization
            visual_config = {
                "metadata": dict(getattr(rs, "metadata", {}) or {}),
                "scale_states": dict(getattr(rs, "scale_states", {}) or {}),
            }

    return BatchedWarpYAMEnv(
        base_env,
        num_worlds=num_worlds,
        camera_backend=camera_backend,
        camera_gpu_id=camera_gpu_id,
        render_cameras=render_cameras,
        mask_variable_count=mask_variable_count,
        mask_reset_options=mask_reset_options,
        fixed_reset_options=fixed_reset_options,
        visual_seed=visual_seed,
        visual_config=visual_config,
    )


__all__ = [
    "MuJoCoYAMEnv",
    "BatchedWarpYAMEnv",
    "make_env",
    "make_batched_env",
    "get_i2rt_sim_config",
    "get_i2rt_config",
    "RobotSystemConfig",
    "RandomizationState",
    "TASK_RANDOMIZERS",
    "SimTaskSpec",
    "get_task_spec",
    "list_task_specs",
    "maybe_get_task_spec",
    "TaskEvalResult",
    "TaskEvaluator",
    "make_task_evaluator",
    "CollectionTaskGroup",
    "DATA_COLLECTION_TASKS",
    "get_data_collection_task",
    "list_data_collection_tasks",
    "maybe_get_data_collection_task",
    "DEFAULT_SCENE_XML",
    "ResolvedTask",
    "SCENE_XMLS",
    "SceneTaskSpec",
    "get_scene_task_spec",
    "get_task_physics_defaults",
    "get_task_randomizer",
    "get_task_scene_xml",
    "list_scene_task_names",
    "list_scene_tasks",
    "maybe_get_scene_task_spec",
    "resolve_env_task_name",
    "resolve_task",
]
