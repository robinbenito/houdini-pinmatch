# Camera Pin Matcher – technical plan

Target: Houdini 22.0.368 (Python 3.13, numpy 2.3). Everything below marked ✅ was
checked against the local H22 docs (`$HFS/houdini/help/hom.zip`) and/or a live
hython / GUI session before writing code. For the current code map, dev loop and
Houdini gotchas see [DEVELOPMENT.md](DEVELOPMENT.md).

## Architecture

```
src/pinmatch.py        -> HDA "PythonModule"      (camera model, LM solver, pin store,
                                                    camera read/write/keys, parm callbacks)
src/pinmatch_state.py  -> HDA "ViewerStateModule" (viewer state: view lock, 2D pan/zoom,
                                                    drawables, pin interaction, HUD, menu, hotkeys)
build_hda.py           -> hython script, assembles the OBJ HDA from src/ (the .hda is a build product)
tests/                 -> hython tests (core + GT), GUI smoke test driver, test-scene generator
```

* OBJ-level HDA `pinmatch::camera_pin_matcher::1.0` (subnet, no inputs, no geometry of its own
  besides the view proxy camera). Its default state is the embedded Python viewer state. Select
  the node, Enter in the viewport (or press the "Enter Tool" button).
* The state only *reads* the reference object (`displayNode().geometry()`, `worldTransform()`).
  The mesh is hidden **in this viewport only** via the viewport's visible-object mask and redrawn
  by drawables, so no flag/parm of the mesh is ever touched.
* All persistent data lives on the HDA node (parms) and on the camera (keys) → saved with the .hip.

## Pin JSON (hidden string parm `pins`)

```json
{"version": 1, "next_id": 7,
 "frames": {"1": [{"id": 1, "p": [x, y, z], "uv": [u, v], "on": true, "lock": false}]}}
```
* `p`  – world-space 3D anchor, fixed at creation.
* `uv` – 2D target in camera NDC, `[0,1]²`, origin bottom-left (same convention as VEX `toNDC`),
  so it survives resolution changes. Pixels = `uv * (resx, resy)`.
* `id` – stable per feature; copying pins to another frame keeps ids (ghosts/labels match).
* Frames are integer frame numbers (as strings, JSON keys).

## Camera model (must equal Houdini exactly)

* `M = buildTransform(t, r, s, p, pr; xOrd, rOrd) · preTransform · parentWorld`
* ✅ Houdini cameras orthonormalise this: `W = polar(M₃ₓ₃)` (SVD `U·Vᵀ`), translation kept
  (parent/own scale is removed). Verified numerically (1e-8).
* `Pc = Pw · W⁻¹`, camera looks down −Z.
  `u = ((f/A)·Xc/−Zc − winx)/winsizex + ½`, `v = ((f/Av)·Yc/−Zc − winy)/winsizey + ½`,
  `Av = A·resy/(resx·pixelaspect)`. ✅ Matches `toNDC()` to 1e-6 for random rigs.
* Solve variables are the camera's own local parms `tx ty tz rx ry rz focal` → locks are exact
  (locked parms are simply not variables) and keys stay Euler-continuous. Parent/pre transform are
  constants of the solve, so the result is written straight back into local space.

## Solver (numpy Levenberg–Marquardt)

Residual vector, minimised with LM (Marquardt diagonal damping, forward-difference Jacobian,
focal as `log f`, clamped to `[fmin, fmax]`):

1. **Data**: for each active pin `(uᵢ(q) − uᵢ*)·resx`, `(vᵢ(q) − vᵢ*)·resy` → pixels.
2. **Minimal change prior** toward the anchor camera `a` (camera at drag start), expressed in
   *pixel-equivalents* so every term is comparable to the data term:
   `λ·[F·ω_tilt, F·ω_pan, 8R·ω_roll, 8R·log(f/fa), 120F·δx/D, 120F·δy/D, 40R·δz/D]`
   with `ω = log(Raᵀ R)` (camera frame), `δ = Raᵀ(c − ca)`, `F` focal in px, `R` half image
   diagonal in px, `D` median pin depth, `λ = 2e-4`. Stiffness order pan/tilt < roll ≈ zoom < dolly < truck/pedestal gives:
   1 pin → pan/tilt; 2 pins → + roll + zoom (dolly if focal locked); 3+ → translation as needed.
   `λ` is small (fit dominates whenever the pins determine the camera).
3. **Lock roll** (geometric, independent of rotate order): stiff residual on
   `atan2(Xcam·up, Ycam·up)`.

Guards: steps that push an active pin behind the camera are rejected; focal clamped; NaN → keep
previous camera. Status from the numerical rank of the data Jacobian vs free DOF:
`rank < dof` under-constrained, `2n == dof` solved (exact), `2n > dof` over-constrained (LSQ).

## Interaction / display

* Enter: remember viewport camera, link flag and visible-object mask; look through the asset's
  **view proxy camera**, unlinked, and hide the reference object in this viewport only.
* 2D pan/zoom: ❌ `viewtransform <viewport> window (...)` keeps the window but an *unlinked*
  camera view detaches ~1 s later; ❌ `GeometryViewportCamera.setWindow*` refuses unlinked
  cameras; the camera's own screen window must not be touched. ✅ Solution: `view_proxy` inside the
  asset mirrors the target camera by expression (`origin()` for the world transform, `ch()` for
  the lens) and carries the 2D window in its own `winx/winsize`. Wheel = zoom about cursor,
  MMB = pan, `H` = reset. The viewport picks up proxy changes asynchronously, so the state calls
  `setCamera(proxy)` after its own edits and when `onDraw` sees the view lag the camera.
* Navigation lock: H22's "Lock Camera/Light Tumbling" exists only as a UI action
  (`h.pane.gview.camlocktumble`, no HOM) → the state consumes wheel/MMB/Alt/Space, and a
  watchdog (`CameraSwitched` viewport callback, mouse events, resume) re-attaches the view.
* Mouse ↔ image mapping: uv → viewport pixels is affine and pose independent; measured in
  `onDraw` (view and camera in sync) and reused by the handlers, so live solves never read a view
  one redraw behind the camera. Pick rays come from the camera model.
* Drawables: plate = `GeometryDrawable` **Sprite**, fed an `ImageLayer` at full resolution
  (file-path sprites are capped at 512 px; `max_resolution` must be ints, set at construction;
  `loadImageDataFromFile` returns linear values → re-encoded to sRGB); mesh = Line drawable (wire)
  or Face drawable with baked N·L shading; pins = per-state Locate markers + anchor dots + residual
  lines; labels = `TextDrawable` with `hou.Color`; HUD = `SceneViewer.hudInfo`.
* Undo: `hou.undos.group` for discrete edits; `SceneViewer.beginStateUndo/endStateUndo` around a
  create/drag (opened on the first real movement, so plain clicks leave no undo entry).
* Keys: HOM keys default to `bezier()` with non-auto slopes → auto slopes set explicitly (the UI
  default). Live drag writes `Parm.setPending` (= plain set on un-animated parms).

## Verification status

All stages implemented and tested (see TESTLOG.md). Verified in H22.0.368: camera model vs
`worldTransform`/`toNDC`, sprite/ImageLayer plate, drawable parameter types, `TextDrawable`
multi-draw per `onDraw`, visible-object mask (`"* ^/obj/room"`), state undo grouping, hotkey
registration (`PluginHotkeyDefinitions` + `viewerstate.utils.defineHotkey`), embedded-state
install hooks. Not verifiable here: raw OS input dispatch (Space+drag, real Ctrl/⌘+click) –
synthetic Qt events don't reach viewer states and computer-use access was not granted.

## Stages

1. ✅ HDA skeleton + state: camera lock, 2D pan/zoom, plate + mesh display.
2. ✅ Pins: create (snap), drag, select, delete, flags, drawing, JSON storage, undo.
3. ✅ Solver with locks/presets/regularisation, live solve while dragging.
4. ✅ Timeline: keys on commit, copy pins, ghosts, per-frame delete, keyed-frame list.
5. ✅ HUD, safety checks, README, test scene + GT test log.
