# Building

P64 can validate projects, create runtime bundles, and build Windows desktop
executables through PyInstaller.

## Validate

```powershell
python -m p64 validate samples\FirstScene
```

Validation checks project settings, startup scene paths, script syntax, missing
assets, missing shaders, missing scripts, and invalid component references.

## Runtime Bundle

```powershell
python -m p64 bundle samples\FirstScene
```

This creates a fast bundle under:

```text
samples/FirstScene/build/bundle/
```

Bundles include game data, generated builtin packages, runtime support, and a
launcher script.

## Game Executable

```powershell
python -m p64 build samples\FirstScene
```

The default output is:

```text
samples/FirstScene/build/game/FirstScene/
```

Keep the executable beside its generated support files and folders.

## Build Settings

Build settings are available in the editor. They include executable name,
relative output folder, icon path, console/windowed mode, Python executable, and
build pipeline path.

The build pipeline lives under `libraries/P64Build/` by default and is refreshed
when a project opens.

## Hub Build

Build the portable Hub app:

```powershell
python -m p64 build-hub
```

Output:

```text
build/app/P64/P64Hub.exe
build/app/P64/_internal/
```

`P64Hub.exe` must stay beside the `_internal` folder.

## File Association

The Hub has a `File Association` button that prints the command needed to
associate `.p64` files with the current executable on Windows.
