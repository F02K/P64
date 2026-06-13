# Scripting

P64 gameplay code is written as Python classes that usually inherit from
`p64.engine.scripting.UserScript`. Attach scripts through a `ScriptComponent` in
the editor inspector.

Scripts normally live in `assets/scripts/`. A script entry points to a Python file
and a class name.

## Lifecycle

- `on_start(self)` runs once when the script instance starts.
- `on_update(self, dt)` runs every runtime tick.
- Disabled script entries and inactive entities do not run.
- Play mode runs scripts against a runtime scene copy.

## Available Fields

Every `UserScript` instance receives these fields:

- `self.entity`: the entity that owns the script
- `self.transform`: shortcut to `self.entity.transform`
- `self.scene`: current scene
- `self.project`: current project
- `self.scene_manager`: scene switching API
- `self.input`: keyboard, mouse, and controller state
- `self.time`: current runtime time value passed by the engine
- `self.character_controller`: the entity's `CharacterController`, if present
- `self.entity_physics`: the entity's `EntityPhysics`, if present

## Rotate An Object

```python
from p64.engine.scripting import UserScript


class Spin(UserScript):
    speed = 90.0

    def on_update(self, dt):
        self.transform.rotation.y += self.speed * dt
```

## Keyboard Movement

```python
from p64.engine.scripting import UserScript


class KeyboardMove(UserScript):
    speed = 4.0

    def on_update(self, dt):
        direction = 0.0
        if self.input.is_key_down("d"):
            direction += 1.0
        if self.input.is_key_down("a"):
            direction -= 1.0
        self.transform.position.x += direction * self.speed * dt
```

Key names are lowercase strings such as `"w"`, `"a"`, `"space"`, `"escape"`,
`"left"`, `"right"`, `"shift"`, and `"ctrl"`.

## One-Frame Input

Use `was_key_pressed` and `was_key_released` for actions that should happen once.

```python
from p64.engine.scripting import UserScript


class JumpProbe(UserScript):
    def on_update(self, dt):
        if self.input.was_key_pressed("space"):
            self.transform.position.y += 1.0
        if self.input.was_key_released("space"):
            self.transform.position.y = 0.0
```

Mouse buttons use names such as `"left_mouse"`, `"right_mouse"`, and
`"middle_mouse"`. Mouse state also exposes `mouse_position`, `mouse_delta`, and
`wheel_delta`.

## Controller Input

Controller axes and buttons use normalized names. The first controller can use
names such as `"left_x"`, `"left_y"`, `"right_x"`, `"south"`, and `"start"`.

```python
from p64.engine.scripting import UserScript


class ControllerMove(UserScript):
    speed = 5.0

    def on_update(self, dt):
        x = self.input.get_axis("left_x")
        if self.input.is_button_down("south"):
            self.transform.position.y += self.speed * dt
        self.transform.position.x += x * self.speed * dt
```

Additional controllers are prefixed, for example `"controller1_left_x"`.

## Entity Physics

Add an `EntityPhysics` component to the same entity, then use
`self.entity_physics`.

```python
from p64.engine.math import Vec3
from p64.engine.scripting import UserScript


class Push(UserScript):
    def on_update(self, dt):
        if self.entity_physics is None:
            return
        self.entity_physics.add_force(Vec3(4.0, 0.0, 0.0))
        if self.input.was_key_pressed("space"):
            self.entity_physics.add_impulse(Vec3(0.0, 3.0, 0.0))
```

`EntityPhysics` also exposes `velocity`, `angular_velocity`, `add_torque`, and
`add_angular_impulse`.

## Character Controller Movement

Add a `CharacterController` component to the same entity, then call
`move_character`.

```python
from p64.engine.math import Vec3
from p64.engine.scripting import UserScript


class PlayerController(UserScript):
    speed = 5.0

    def on_update(self, dt):
        x = 0.0
        z = 0.0
        if self.input.is_key_down("d"):
            x += 1.0
        if self.input.is_key_down("a"):
            x -= 1.0
        if self.input.is_key_down("w"):
            z -= 1.0
        if self.input.is_key_down("s"):
            z += 1.0

        motion = Vec3(x * self.speed, 0.0, z * self.speed)
        self.move_character(motion, dt)
```

`move_character` returns the collision-adjusted motion as a `Vec3`.

## Scene Switching

Use `self.scene_manager.load_scene_by_name` to queue a scene switch. The switch is
applied after the current script tick.

```python
from p64.engine.scripting import UserScript


class Door(UserScript):
    target_scene = "second"
    spawn_id = "door_exit"

    def on_update(self, dt):
        if self.input.was_key_pressed("enter"):
            self.scene_manager.load_scene_by_name(self.target_scene, spawn_id=self.spawn_id)
```

You can also use `load_scene("assets/scenes/second.scenep64")` when you know the
scene path.

## Persistent Objects

Persistent entities are carried into the next scene. This is useful for players,
inventory managers, and global state objects.

```python
from p64.engine.scripting import UserScript


class PlayerPersistence(UserScript):
    def on_start(self):
        self.persistent()
```

Spawn points can reposition persistent entities during scene switches.

## Common Errors

- `Script entry has no script file`: the script entry is empty.
- `Script file not found`: the file is not in `assets/scripts/` or the legacy
  scripts folder.
- `Class 'Name' not found`: the class name in the Script component does not match
  the Python class.
- Syntax errors are reported by `python -m p64 validate <project>`.
