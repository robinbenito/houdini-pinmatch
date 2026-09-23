# Test log – Camera Pin Matcher

Environment: Houdini 22.0.368 (Indie license), macOS 26.3.1 on arm64, Python 3.13.10, numpy 2.3.2.

To reproduce:

```bash
hython tests/test_core.py
```

```bash
houdini -foreground tests/test_gui.py
```

The GUI test writes its output to `tests/gui_test_log.txt`.

## Ground-truth test (test scene, GUI, full tool path)

`scenes/pinmatch_test.hiplc` contains the following:

* `/obj/room`: the reference mesh, a room with a cabinet, a table, a cube and a pillar.
* `/obj/gt_cam`: the known camera, 1280×720, 36 mm aperture, animated over frames 1–24.
  `scenes/plate/plate.$F4.jpg` was rendered from it with Karma.
* `/obj/cam1`: the camera to solve, starting from a deliberately wrong pose and focal.

The test enters the tool and places six pins one at a time, as a user would. For each pin it
**Ctrl+drags** from a mesh corner, clicking 3.6 px off the corner so point snapping has to catch
it, to the corner's position on the plate. That position comes from Houdini's own projection
through `gt_cam`, plus Gaussian noise with σ = 0.5 px per axis to simulate human placement. The
drag is solved live, and the camera is solved and keyed on release. The pin corners are chosen
visible in the plate and spread across the image.

**Frame 1: accuracy as pins are added** (position in scene units ≈ m; the room is 8 × 10 m):

| pins | status | position error | rotation error | focal error | RMS |
|---|---|---|---|---|---|
| 3 | under-constrained | 1.3839 | 1.611° | 20.034 % | 0.000 px |
| 4 | over-constrained | 0.0326 | 0.146° | 0.409 % | 0.285 px |
| 5 | over-constrained | 0.0587 | 0.181° | 0.792 % | 0.470 px |
| **6** | **over-constrained** | **0.0173** | **0.171°** | **0.129 %** | 0.485 px |

**Frame 1: solved vs. ground truth** after 6 pins and Solve & Key:

| parm | ground truth | solved | start |
|---|---|---|---|
| tx | 1.4000 | 1.3952 | 2.2000 |
| ty | 1.6500 | 1.6342 | 1.3000 |
| tz | 3.6000 | 3.5948 | 4.4000 |
| rx | −9.0000 | −8.8531 | −4.0000 |
| ry | 14.0000 | 13.9914 | 24.0000 |
| rz | 0.0000 | 0.1287 | −3.0000 |
| focal | 30.0000 | **29.9613** (−0.13 %) | 42.0000 |

**Frame 12: rough matchmove.** The six pins were copied from frame 1 with **C**, dragged onto
the frame-12 plate positions without noise, then solved with Solve & Key. Position, rotation and
focal errors are 0.0000 / 0.000° / 0.000 %. The camera is keyed at 1 and 12 and interpolates in
between.

With 3 pins the problem is under-constrained: 6 equations for 7 unknowns. The minimal-change
prior picks the smallest plausible camera change, so the focal is still far off, which is
expected. From 4 pins on, the pins determine the camera. The residual error of 0.1–0.8 % focal
comes from the injected 0.5 px click noise; the noise-free frame 12 solve is exact.

## Headless solver tests (`tests/test_core.py`)

```
camera model: max |W - worldTransform| = 2.42e-08, max |uv - toNDC| = 9.38e-07
cheirality: min depth 0.031, focal 5.00, status over-constrained
1 pin : err 0.0000 px, camera moved 0.00011, focal 30.000 -> 29.981, rot change [ 3.001  5.601 -0.345]  [under-constrained]
2 pins focal free  : pinA err 0.0004 px, pinB err 0.0004 px, focal 33.35, move (x,y,z) [ 0.0017  0.0012 -0.0202]  [under-constrained]
2 pins focal locked: pinA err 0.0058 px, pinB err 0.0057 px, focal 30.00, move (x,y,z) [ 0.0266  0.0075 -0.3775]  [under-constrained]
GT  6 pins noise 0.0px: dpos 0.0000  drot 0.000 deg  dfocal 0.000%  rms 0.000px  over-constrained  3 it  5.6 ms
GT  6 pins noise 0.5px: dpos 0.0075  drot 0.032 deg  dfocal 0.066%  rms 0.453px  over-constrained  3 it  5.1 ms
GT 12 pins noise 0.5px: dpos 0.0056  drot 0.020 deg  dfocal 0.045%  rms 0.583px  over-constrained  2 it  5.9 ms
GT 30 pins noise 0.5px: dpos 0.0028  drot 0.028 deg  dfocal 0.005%  rms 0.675px  over-constrained  2 it  5.3 ms
hda: resolution matched, Solve & Key = 1 undo step, presets / copy / deactivate / delete buttons ok
locks: locked values bit-identical, roll lock holds (1.9e-08 rad), focal clamped to 30.000
protected parms: ['ry', 'tx'] treated as locked, look-at disables solving
OK
```

* **Camera model:** 12 random rigs with non-uniformly scaled parents, pivots, pivot rotation,
  pre-transforms, every transform and rotate order, screen windows and pixel aspects. They are
  compared against Houdini's `worldTransform()` and VEX `toNDC()`, which works in float32.
* **1 pin:** pan/tilt only. The camera moves 0.1 mm and the focal changes 0.06 %.
* **2 pins:** pin A stays within 0.0004 px. With focal free the camera zooms; with focal
  locked it dollies (−0.38 along the view axis).
* **Performance:** a 30-pin solve takes about 5 ms. In the GUI, a live drag event with 30 pins
  (solve plus camera parameter writes) takes about 2 ms, and a release with polish and keys about
  4 ms.

## Acceptance criteria

| Criterion | Result | Evidence |
|---|---|---|
| Viewport can't leave the camera view while the tool is active | PASS for everything but real input, see gaps | GUI test: forced camera switch is re-attached; the wheel zooms 2D without moving the camera; Space is consumed |
| Mesh transform and geometry unchanged | PASS | SHA-1 over `asCode(recurse)` of the object network, point positions, topology and world transform, before and after all operations |
| One aligned pin stays in place while a second is dragged | PASS | Maximum drift 0.0009 px over the whole drag (GUI); 0.0004 px (headless) |
| Locked parameters never change | PASS | Bit-identical values for every lock combination (headless); Lock Focal and Nodal presets during GUI drags |
| Unlock all / delete all / per-frame delete work and are undoable | PASS | GUI test; asset buttons tested headless |
| Pins and keys survive save and reopen | PASS | GUI test saves, clears, reloads and compares the pin JSON and every keyframe |
| 6 good pins recover focal within about 1 % and position within a small tolerance | PASS | 0.13 % and 1.7 cm with 0.5 px click noise; exact without noise |

Other checks in the GUI test, all PASS (38 in total):

* Each create+drag and each drag is exactly one undo step, and one undo reverts pins and
  camera together.
* A Ctrl+click delivered as press, release and then `Picked` creates exactly one pin, and pin
  creation is undoable.
* The toggle active, toggle lock, locked-pin-can't-drag and Backspace-delete actions work, and
  each is undoable.
* Copying pins keeps the ids.
* Ghost frames are correct.
* Keys use `bezier()` and interpolate.
* Delete keys works and is undoable.
* Leaving the tool restores the viewport.

## Known gaps

* **Real OS mouse and keyboard input wasn't exercised.** Synthetic Qt events don't reach
  Houdini's viewer-state dispatch, and a computer-use check was declined. The GUI test calls the
  live state's handlers directly, with rays from the real viewport. Its mock events mimic HOM
  where that matters: `curViewport()` returns a fresh wrapper, and mouse coordinates are in
  view space. So the raw dispatch still needs a check by hand:
  * Ctrl/⌘ + click
  * Space + drag
  * the hotkeys, which are registered and assigned (`A K H C` checked)
