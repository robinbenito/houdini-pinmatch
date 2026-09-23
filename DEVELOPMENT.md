# Developer guide / handoff

This file gets a developer (or an AI agent) productive on the Camera Pin Matcher. It covers
how the code is organised, the edit–build–test loop, and the Houdini 22 behaviours that cost
time to discover. User-facing docs are in [README.md](README.md), the original technical plan
in [PLAN.md](PLAN.md), and the measured results in [TESTLOG.md](TESTLOG.md).

## Status

* **Features:** everything the original spec asks for is implemented: locked camera view,
  2D pan/zoom, plate and mesh display modes, pin create/drag/select/delete/flags, the solver
  with locks, presets and minimal-change prior, per-frame pins, keys, copy and ghosts, HUD,
  undo, and safety checks. See the checklist in TESTLOG.md.
* **Verified:**
  * the camera model against Houdini's `worldTransform()` and `toNDC()`
  * the solver on synthetic ground truth
  * every tool action through the live viewer state in a GUI session (38 checks)
  * ground-truth recovery on the rendered test scene
* **Not verified: real OS mouse and keyboard input** (issue #1). Houdini's raw event dispatch
  can't be scripted, so the GUI test calls the state's handlers directly. This is the first
  thing to check by hand.
* **Tested on:** macOS 26 (arm64), Houdini 22.0.368, **Indie** license only.

## Layout

```
src/pinmatch.py          HDA PythonModule  - pure logic, importable from hython for tests
src/pinmatch_state.py    HDA ViewerStateModule - the interactive tool
build_hda.py             builds otls/camera_pin_matcher.hda[lc|nc] from src/ (build product is committed)
tests/test_core.py       hython: camera model, solver, locks, protected parms, asset buttons
tests/test_gui.py        GUI Houdini: drives the live state (see "Testing")
tests/make_test_scene.py builds scenes/pinmatch_test.hip* and renders scenes/plate (Karma)
dev/                     remote-control harness for a GUI Houdini (development only)
docs/                    README screenshots
```

### `src/pinmatch.py` (asset PythonModule)

The viewer state reaches it as `node.hdaModule()`, and tests reach it by `import pinmatch`.

* **`Rig(cam)`**: a snapshot of a camera. The seven solve values are
  `q = (tx, ty, tz, rx, ry, rz, focal)`; everything else is held constant (scale, pivots,
  pre-transform, parent, aperture, resolution, pixel aspect, screen window).
  * `world(q)` gives `buildTransform(...) · preTransform · parentAndSubnet`, with the 3×3
    orthonormalised by SVD. That is what Houdini does for cameras.
  * `project`, `unproject` and `pixel_errors` work on those matrices.
  * Row-vector convention throughout, like `hou.Matrix4`.
* **`solve(rig, P, UV, free, anchor, start, weight, lock_roll, focal_range, iters, polish)`**
  returns `(q, info)`. It runs Levenberg–Marquardt on the free subset of `q`, with focal as
  `log f` and a forward-difference Jacobian.
  * **Residuals:** the pixel errors of the pins, a minimal-change prior toward `anchor` in
    pixel equivalents, and optionally a stiff roll-lock term.
  * **Tuning constants** (see PLAN.md for the reasoning):
    * λ = 2e-4 × `weight`
    * stiffness: pan/tilt 1, roll 8, zoom 8, dolly 40, truck/pedestal 120
    * `polish` re-anchors the prior at the result; 2 passes on commit, 0 while dragging
  * **`info`:** `status` comes from the numerical rank of the data Jacobian against the free
    degrees of freedom. `info` also carries `rms`, per-pin `errors` (NaN for pins behind the
    camera), `n`, `behind`, `rank`, `dof` and `iterations`.
* **Pin store:** pins are JSON in the hidden `pins` parameter. The functions are
  `load_pins` / `save_pins` / `edit_pins` (one undo group), `pins_at`, `new_pin`,
  `pin_frames`, `neighbour_frames`, `nearest_pin_frame`, `copy_nearest_pins`, `set_all`,
  `delete_frame_pins` and `delete_all_pins`.
* **Camera IO:**
  * `camera_problems(cam)` returns `(fatal, {parm: reason})`. Parameters with an expression,
    a reference, a lock or a CHOP override come back as "protected". Look-at, constraints,
    non-perspective cameras and cameras in locked assets are fatal.
  * `write_camera` (live, `setPending`)
  * `key_camera` (`bezier()`, auto slopes)
  * `keyed_frames`, `delete_keys`
  * `solve_pins` / `solve_and_key` (used by the buttons and the K key)
  * `plate_path`, `match_resolution`
* **Button callbacks:** the `cb_*` functions. Houdini already wraps parameter callbacks in
  one undo group.

### `src/pinmatch_state.py` (viewer state)

* **View lock:** `_attach` looks through the asset's `view_proxy`, unlinked, and hides the
  reference object in this viewport through the visible-object mask.
  * `_sync_view` re-applies `setCamera(proxy)`, because the viewport picks up camera changes
    asynchronously.
  * `_watchdog` and the `CameraSwitched` viewport callback re-attach the view if something
    switches it.
  * `onExit` restores the viewport.
* **2D view:** hidden parameters `view_cx/view_cy/view_zoom` drive the proxy's screen window
  through expressions (see `add_view_proxy` in build_hda.py). `_set_view` writes them with
  undo disabled.
* **Mouse ↔ image mapping:** the map from uv (camera NDC) to viewport pixels is affine and
  doesn't depend on the camera pose. `_measure_frame` measures it in `onDraw`, where view and
  camera are in sync, and the handlers reuse it through `_screen`, `_uv` and `_mouse`. Don't
  derive the mouse position from `ui_event.ray()`: during a live solve the view lags the camera
  by one redraw. Pick rays come from the camera model (`_ray`).
* **Picking and snapping:** `_pick` grabs a pin within 12 px. `_hit_mesh` handles the three
  snap modes:
  * points: the nearest mesh point within 20 px, where hidden points only lose ties
  * edges: the nearest point on the edges of the face under the cursor
  * surface: the ray hit
* **Editing:** `_begin_drag` → `_drag_to` → `_end_drag`.
  * The drag keeps its own copy of the pins, `self.drag["data"]`.
  * It opens `beginStateUndo` on the first real movement, or at once when creating a pin.
  * It solves live with `polish=0`.
  * On release it saves the pins, solves with `polish=2`, keys if autokey is on, and closes
    the undo group.
* **Drawing:** `onDraw` draws the reference mesh (`_mesh`: Line drawable, or a Face drawable
  with baked N·L shading), then the plate (`_plate`: sprite fed by an ImageLayer at full
  resolution), then the pins (`_draw_pins`), then updates the HUD (`_update_hud`, which only
  pushes values that changed).
* **Menu and hotkeys:** `_menu()` builds the RMB menu. `ACTIONS` lists the actions that have
  hotkeys, registered through `viewerstate.utils.defineHotkey`.

### Asset internals (built by `build_hda.py`)

* **Parameters:** tabs Setup / Display / Pins / Solver. The stock OBJ Transform and Subnet
  tabs are hidden. Hidden parameters: `pins`, `view_cx`, `view_cy`, `view_zoom`.
* **`view_proxy` camera:**
  * transform: `origin("", chsop("../camera"), "TX".."RZ")`
  * lens: `ch(chsop("../camera") + "/focal")` and so on
  * screen window: composed with the 2D view parameters
* **Sections:** `PythonModule`, `ViewerStateModule`, `ViewerStateInstall` and
  `ViewerStateUninstall` (the latter two call `viewerstate.utils.register_pystate_embedded`),
  `DefaultState` and `Help`.
* **Names:** the type is `pinmatch::camera_pin_matcher::1.0`, and the state has the same name.
  Hotkey symbols are `h.pane.gview.state.obj.<encoded type name>.<action>`.

### Conventions

* **Coordinates:**
  * `uv` means camera NDC, the same as VEX `toNDC()`: 0–1, origin bottom-left, including the
    camera's screen window.
  * Viewport pixels have their origin at the bottom-left, as returned by `mapToScreen`.
  * `ui_event.device().mouseX()/mouseY()` are in view coordinates. Convert them with
    `windowToViewportTransform()`.
* **Dependencies:** only Houdini's Python and numpy. The tool uses no environment variables
  and no hard-coded paths.
* **Pin JSON:** it has a `"version"` field. Bump it and add a migration in `load_pins` if the
  format changes.
* **Build products:** `otls/` and `scenes/` are committed. Rebuild them and commit them
  together with the source changes.

## Edit, build, test

```bash
hython build_hda.py
```

Rebuilds `otls/camera_pin_matcher.hda(lc|nc)`. The extension follows the license.

```bash
hython tests/test_core.py
```

Headless checks. About 30 s, most of it Houdini start-up.

```bash
houdini -foreground tests/test_gui.py
```

Opens a fresh GUI session, runs the GUI checks, writes `tests/gui_test_log.txt` and exits.

```bash
hython tests/make_test_scene.py --no-render
```

Regenerates the scene. Drop `--no-render` to re-render the plate as well (about 3 min).

### Iterating inside one GUI session (dev/ harness)

Start Houdini once with an RPC server:

```bash
houdini -foreground dev/start_rpc.py
```

Then, from a terminal, rebuild, reload the asset and re-enter the tool on the test scene:

```bash
hython build_hda.py && hython dev/rpc.py dev/reload.py --scene
```

Run any snippet inside Houdini (`hou` is pre-imported):

```bash
hython dev/rpc.py -c "print(hou.ui.paneTabOfType(hou.paneTabType.SceneViewer).currentState())"
```

Run the GUI test in the open session:

```bash
hython dev/rpc.py tests/test_gui.py
```

Take a screenshot of the Houdini window:

```bash
hython dev/rpc.py dev/grab.py /tmp/houdini.png
```

Quit the session:

```bash
hython dev/rpc.py -c "import hdefereval; hdefereval.executeDeferred(lambda: hou.exit(suppress_save_prompt=True))"
```

`dev/rpc.py` runs the code on Houdini's main thread and returns what it prints. The server uses
port 18811, the `hrpyc` default. If another plugin (for example an MCP bridge) already uses that
port, change it in both `dev/` files.

## Testing notes

What I learned about testing a viewer state:

* **Synthetic Qt input doesn't reach the viewer state.** Events sent with `QTest` or
  `sendEvent` to the viewport's `QOpenGLWidget` ("RE_WindowDrawable") or its `QWindow` never
  arrive in `onMouseEvent`.
* **Handlers can't be monkeypatched.** Houdini binds a state's handlers when the template is
  registered. `tests/test_gui.py` therefore finds the live `State` instance with
  `gc.get_objects()` (`live_state(node)`) and calls its handlers directly with mock events.
* **The mock events have to match HOM where it matters:**
  * `curViewport()` must return a *fresh* wrapper each time. HOM viewport wrappers never
    compare equal, and this hid a bug that ignored every real mouse event; see commit 9d1d665.
  * `mouseX/mouseY` are in view coordinates.
  * `ray()` comes from `GeometryViewport.mapToWorld(x, y)`, which returns
    **(direction, origin)**.
  * A click may arrive as `Picked` alone or as `Start`, `Changed` and then `Picked`. The state
    handles both.
* **Screenshots:** `hou.qt.mainWindow().grab()` captures the GL viewport. Call
  `state._sync_view()` first: after script-driven camera or time changes, the viewport updates
  asynchronously.
* **The test scene doesn't load its asset on other machines.** The .hip records the asset
  library as `$JOB/...` with `JOB` saved as an absolute path, so the test installs
  `otls/camera_pin_matcher.hda*` before loading the scene (issue #2).
* **`__file__` isn't defined** when Houdini runs a script given on the command line. Use
  `sys._getframe().f_code.co_filename`.

## Houdini 22 behaviours worth knowing

(Also summarised in README "Houdini 22 API notes".)

**Viewport and camera view**
* "Lock Camera/Light Tumbling" and "Allow 2D Pan & Zoom" exist only as UI or hotkey actions
  (`h.pane.gview.camlocktumble`). There is no HOM for them and no way to trigger a hotkey action
  from Python.
* `viewtransform <viewport> window ( … )` sets the viewport's own 2D window, but an *unlinked*
  camera view detaches about 1 s later. `GeometryViewportCamera.setWindowOffset/Size` raise
  errors on unlinked cameras. `setCamera()` resets the viewport window. That's why the asset
  uses a proxy camera.
* A camera's world transform is **orthonormalised** (polar decomposition). Parent and own
  scale are removed, but the translation keeps the parent's scale.
* The viewport picks up camera parameter changes **asynchronously**. `vp.setCamera(cam)`
  applies them immediately (about 0.1 ms).
* `hou.GeometryViewport` wrappers are never equal (`vp == vp` is False). Compare `name()`.
  Nodes do compare equal.

**Drawables**
* **Sprite images:**
  * A sprite fed a file path is displayed at no more than **512 px**.
  * `max_resolution` accepts **int** sequences only; floats and `hou.Vector2` are rejected,
    although the docs say float.
  * Full resolution works with an **`hou.ImageLayer`** source plus `max_resolution` equal to
    the image size, set when the drawable is constructed.
  * Pixel-array (`{"data", "width", "height"}`) sources render white.
* **Plate colour:** `hou.loadImageDataFromFile(..., Float32 | Int8)` returns **linearised**
  values. Image layers are displayed as-is, so they need re-encoding to sRGB.
* **Other parameter types:**
  * `falloff_range` needs a plain tuple, not a `hou.Vector2`.
  * `TextDrawable` `color1` needs a `hou.Color` or a 4-tuple, and only `hou.Color` renders
    crisp.
  * Colour tuples must be floats; `(0, 0, 0, 1)` raises InvalidInput.
* **Point styles** such as `Locate`, `Ring` and `Cross` ignore `Cd`, so they are coloured
  through `color1`, one drawable per colour.
* Parameters passed to `drawable.draw(handle, params)` **persist** on the drawable.
* An exception anywhere in `onDraw` drops the draws that were queued before it. Nothing is
  logged to stdout.

**Assets, scenes and the rest**
* Embedded HDA states need the `ViewerStateInstall` and `ViewerStateUninstall` sections, or
  they never register.
* OBJ assets are wrapped in the standard Transform/Subnet switcher. Hide those tabs and append
  your own tabs to the same switcher (see `build()`).
* Keyframes created through HOM default to `bezier()` with non-auto slopes. Call
  `setSlopeAuto(True)` to match keys set in the UI.
* `attribwrangle` has no SOP verb. `OpNode.asCode()` has no `save_keyframes` argument.
* Houdini saves Limited Commercial content (HouLC headers) under Indie whatever the extension.
  The build scripts pick `.hdalc`/`.hiplc` accordingly.

## Open work

See the GitHub issues. The most important ones:

* #1: verify with real mouse and keyboard input
* #2: test scene portability
* #3: Space and Alt navigation lock under real input

Everything else is enhancements, grouped by the labels `solver`, `viewer-state`, `performance`
and `testing`.
