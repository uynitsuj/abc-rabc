This directory is a task-scoped asset namespace for the `mug_flip` scene.

- `mug/mug_0/` is copied from the original plain mug asset at `assets/mug/`.
- `mug/mug_1/` onward are copied from
  `assets_robocasa/objaverse/objaverse/mug/`, sorted by source variant index.

The task randomizer samples these variants on reset. When `mug_0` is sampled,
the plain mug material is still recolored at runtime for additional diversity.
