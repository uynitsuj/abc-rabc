# Task-Local Dishrack Assets

This directory is a task-scoped asset namespace for the `dishrack` scene.

Layout:

- `plate/plate_0/`
  Default baked plate asset originally used by `yam_dishwasher_scene.xml`.
- `plate/plate_1/` through `plate/plate_n/`
  Robocasa plate variants copied from `assets_robocasa/objaverse/objaverse/plate/`.
- `dish_rack/dish_rack_0/`
  Default baked dish rack asset originally used by `yam_dishwasher_scene.xml`.
- `dish_rack/dish_rack_1/` through `dish_rack/dish_rack_n/`
  Robocasa dish-rack variants copied from
  `assets_robocasa/objects_lightwheel/lightwheel/dish_rack/`.

All task-local `model.xml` files carry their own visual mass and collision
parameters so the scene builder only has to copy assets and bodies into the
placeholder-based base scene.
