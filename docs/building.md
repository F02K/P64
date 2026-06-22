# Building

P64 can validate projects, create runtime bundles, and build Windows desktop
executables through PyInstaller.

## Root Build Center

On Windows, use the root `build.bat` to build P64 tooling:

```powershell
.\build.bat
```

With no arguments it opens the Build Center dashboard. The dashboard builds the
portable P64 App/Hub, runs tests, or performs a full tooling build.

Available targets:

```text
ui        open the Build Center dashboard
app       build the portable P64 App/Hub executable
hub       alias for app
all       run verbose tests, then build the P64 App
test      run the verbose test suite only
```

Examples:

```powershell
.\build.bat app
.\build.bat all --skip-tests
.\build.bat hub --skip-pyinstaller
```

The Build Center shows total and current-step progress, elapsed time, per-step
durations, and structured log events. The `test` and `all` targets use verbose
unittest output, so the log and test progress indicator show individual test
names instead of only progress dots.

`build_all.bat` is kept as a deprecated compatibility wrapper for
`build.bat all`.

Project/game builds are handled from the editor project build settings. The
low-level project CLI remains available for automation:

```powershell
python -m p64 build path\to\Project
```

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
