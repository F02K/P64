# Scripting

P64 gameplay code is written as Python classes that usually inherit from
`p64.engine.scripting.GameScript`. Attach scripts through a `ScriptComponent` in
the editor inspector.

Scripts normally live in `assets/scripts/`. A script entry points to a Python file
and a class name.

New projects include VSCode/Pylance support files automatically. For existing
projects, use `Project > Setup VSCode` in the editor or:

```powershell
python -m p64 vscode path\to\project
```

This refreshes `.vscode/` and regenerates
`packages/P64Generated/python/p64_project_api.py`, which exposes project
constants for scenes, assets, and common input names.

## Lifecycle

- `on_start(self)` runs once when the script instance starts.
- `on_update(self, dt)` runs every runtime tick.
- Disabled script entries and inactive entities do not run.
- Play mode runs scripts against a runtime scene copy.

## Available Fields

Every `GameScript` instance receives these fields:

- `self.entity`: the entity that owns the script
- `self.transform`: shortcut to `self.entity.transform`
- `self.scene`: current scene
- `self.project`: current project
- `self.scene_manager`: scene switching API
- `self.input`: keyboard, mouse, and controller state
- `self.time`: current runtime time value passed by the engine
- `self.character_controller`: the entity's `CharacterController`, if present
- `self.entity_physics`: the entity's `EntityPhysics`, if present
- `self.audio_source`: the entity's first `AudioSource`, if present
- `self.forward`, `self.right`, `self.up`: readonly world direction vectors from the entity transform

## Transform Direction Helpers

`self.transform.position`, `self.transform.rotation`, and `self.transform.scale`
are local to the parent. For root objects, those local values are also the world
values.

`self.transform.forward`, `self.transform.right`, and `self.transform.up` are
computed from the object's effective rotation. They are not constant global
vectors. A root object with `rotation.y = 90` has a different
`self.transform.forward` than `Vec3.forward()`, and a child object also includes
the rotation inherited from its parents.

Use `self.transform.local_forward`, `self.transform.local_right`, and
`self.transform.local_up` when you explicitly want directions from only the
object's local rotation.

`self.transform.scene_object` returns the entity that owns the transform.
`self.transform.sceneObject` is also available as a Unity-style alias.

Rotations are stored internally as normalized quaternions. The existing
`self.transform.rotation` field remains a live Euler-angle view in degrees, so
existing code such as `self.transform.rotation.y += 90` remains valid.

For quaternion-native rotation use:

```python
from p64.engine.math import Quaternion, Vec3

self.transform.local_quaternion = Quaternion.angle_axis(90, Vec3.up())
world_orientation = self.transform.world_quaternion
```

`Quaternion` also provides `from_euler`, `to_euler`, `look_rotation`, `inverse`,
`lerp`, and `slerp`. Parent and child orientations are combined using
quaternion multiplication.

## Rotate An Object

```python
from p64.engine.scripting import GameScript


class Spin(GameScript):
    speed = 90.0

    def on_update(self, dt: float) -> None:
        self.transform.rotation.y += self.speed * dt
```

## Keyboard Movement

```python
from p64.engine.scripting import GameScript


class KeyboardMove(GameScript):
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
from p64.engine.scripting import GameScript


class JumpProbe(GameScript):
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
from p64.engine.scripting import GameScript


class ControllerMove(GameScript):
    speed = 5.0

    def on_update(self, dt):
        x = self.input.get_axis("left_x")
        if self.input.is_button_down("south"):
            self.transform.position.y += self.speed * dt
        self.transform.position.x += x * self.speed * dt
```

Additional controllers are prefixed, for example `"controller1_left_x"`.

## Interactive UI

UI controls use `Canvas` and `RectTransform` for their screen-space bounds. Add
one interactive component to a UI entity:

- `UIButton`
- `UIToggle`
- `UISlider`
- `UIScrollView`

Mouse, keyboard, and controller input are handled automatically. Arrow keys,
WASD, the D-pad, and the left stick move focus. Enter, Space, or the controller
South button submit the focused control. Escape or controller East cancel it.

Attach a script to the same entity and implement only the callbacks it needs:

```python
from p64.engine.scripting import GameScript


class MainMenuButton(GameScript):
    def on_ui_focus(self):
        pass

    def on_ui_click(self):
        self.scene_manager.load_scene_by_name("main")

    def on_ui_cancel(self):
        pass
```

Toggle and Slider controls call `on_ui_value_changed(value)`. ScrollView calls
`on_ui_scroll_changed(x, y)`. Pointer transitions call
`on_ui_pointer_enter()` and `on_ui_pointer_exit()`, while controller or keyboard
focus calls `on_ui_focus()` and `on_ui_blur()`.

Every `GameScript` also receives `self.ui_control` plus the matching typed
shortcut: `self.ui_button`, `self.ui_toggle`, `self.ui_slider`, or
`self.ui_scroll_view`.

Navigation is selected geometrically by default. Inspector references can
override Up, Down, Left, and Right. A Canvas can provide an Initial Focus entity.
ScrollView content is clipped to the view rectangle and can be moved with the
wheel, pointer drag, or right stick.

## Entity Physics

Add an `EntityPhysics` component to the same entity, then use
`self.entity_physics`.

```python
from p64.engine.math import Vec3
from p64.engine.scripting import GameScript


class Push(GameScript):
    def on_update(self, dt):
        if self.entity_physics is None:
            return
        self.entity_physics.add_force(self.transform.forward * 4.0)
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
from p64.engine.scripting import GameScript


class PlayerController(GameScript):
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

## Raycasts

Gameplay raycasts test enabled Colliders on active entities. Renderer geometry
without a Collider is not included.

```python
from p64.engine.math import Vec3
from p64.engine.scripting import GameScript


class LookForWall(GameScript):
    def on_update(self, dt):
        hit = self.raycast(
            self.transform.world_position,
            self.transform.forward,
            max_distance=20.0,
            layer_mask="World",
        )
        if hit:
            print(hit.entity.name, hit.point, hit.normal, hit.distance)
```

`self.raycast` returns the nearest `RaycastHit` or `None`.
`self.raycast_all` returns all hits sorted by distance. Both support
`layer_mask`, `include_triggers`, and `ignore_entity`. Script shortcuts ignore
the script's own entity and its children by default. Use
`self.collision_world.raycast(...)` directly when no implicit ignore target is
wanted.

## Audio Playback

Add an `AudioSource` component to the same entity, assign an imported WAV
AudioClip, then call `play`, `stop`, `pause`, or `resume`. The scene must also
have an active `AudioListener` component, usually on the camera, or playback is
silent.

```python
from p64.engine.scripting import GameScript


class Footstep(GameScript):
    def on_update(self, dt):
        if self.audio_source and self.input.was_key_pressed("space"):
            self.audio_source.play()
```

WAV files under `assets/` are automatically imported as AudioClips. Runtime
copies are mono, 16-bit WAV files with a maximum sample rate of 22050 Hz.
Spatial playback is handled by the runtime relative to the active
`AudioListener` position and rotation.

## Scene Switching

Use `self.scene_manager.load_scene_by_name` to queue a scene switch. The switch is
applied after the current script tick.

```python
from p64.engine.scripting import GameScript


class Door(GameScript):
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
from p64.engine.scripting import GameScript


class PlayerPersistence(GameScript):
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
