# Camera Pin Matcher – technical plan

Target: Houdini 22.0.368 (Python 3.13, numpy 2.3). Everything below marked ✅ was
checked against the local H22 docs (`$HFS/houdini/help/hom.zip`) and/or a live
hython / GUI session before writing code.

## Architecture

```
src/pinmatch.py        -> HDA "PythonModule"      (camera model, LM solver, pin store,
                                                    camera read/write/keys, parm callbacks)
src/pinmatch_state.py  -> HDA "ViewerStateModule" (viewer state: view lock, 2D pan/zoom,
                                                    drawables, pin interaction, HUD, menu, hotkeys)
build_hda.py           -> hython script, assembles the OBJ HDA from src/ (the .hda is a build product)
tests/                 -> hython tests (core + GT), GUI smoke test driver, test-scene generator
```

* OBJ-level HDA `pinmatch::camera_pin_matcher::1.0` (subnet, no inputs, no geometry of its own).
  Its default state is the embedded Python viewer state. Select the node, Enter in the viewport
  (or press the "Enter Tool" button).
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
  `Av = A·resy/(resx·pixelaspect)`. To be verified against `toNDC()` for random cameras.
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

* Enter: remember viewport camera, link flag, 2D window, visible-object mask; look through the
  camera, unlink (`lockCameraToView(False)`) so view changes can never write the camera.
* 2D pan/zoom: ✅ `viewtransform <viewport> window ( xmin xmax ymin ymax )` changes only the
  viewport's own window (camera `winx/winsize` untouched, still looking through the camera).
  Wheel = zoom about cursor, MMB = pan, `H` = reset.
* Navigation lock: ✅ H22 has "Lock Camera/Light Tumbling" but only as a UI action
  (`h.pane.gview.camlocktumble`, no HOM) → the state consumes wheel/MMB/Alt/Space events, and a
  watchdog (every event/draw + `CameraSwitched` viewport callback) re-attaches camera + window.
* Drawables: plate = `GeometryDrawable` **Sprite** (image file, world-space billboard filling the
  frustum; far = background, near + alpha = foreground); mesh = Line drawable (wire) or Face
  drawable with baked N·L shading (mode 2); pins = Point/Line drawables; labels = `TextDrawable`;
  HUD = `SceneViewer.hudInfo`.
* Undo: ✅ `hou.undos.group` for discrete edits; `SceneViewer.beginStateUndo/endStateUndo` around a
  drag (one step). Parm-button callbacks are already grouped by Houdini.
* Keys: ✅ HOM keys default to `bezier()` with non-auto slopes → we set auto slopes (UI default).
  Live drag writes `Parm.setPending` (✅ = plain set on un-animated parms); commit writes keys.

## H22 APIs still to verify in the GUI (stage 1–2)

* Sprite drawable size/aspect semantics (`pscale`, `spritescale`), `images`/`max_resolution`,
  per-frame image switching, `Alpha`, depth vs scene geometry (`fade_factor`).
* `TextDrawable` drawn several times per `onDraw` with different params.
* Whether Space/Alt navigation reaches `onKeyEvent`/`onMouseEvent` (can it be consumed?).
* `GeometryViewportSettings.setVisibleObjects` mask syntax for excluding one object.
* `beginStateUndo` + many parm writes during a drag = one undo entry.

## Stages

1. HDA skeleton + state: camera lock, 2D pan/zoom, plate + mesh display.
2. Pins: create (snap), drag, select, delete, flags, drawing, JSON storage, undo.
3. Solver with locks/presets/regularisation, live solve while dragging.
4. Timeline: keys on commit, copy pins, ghosts, per-frame delete, keyed-frame list.
5. HUD, safety checks, README, test scene + GT test log.
