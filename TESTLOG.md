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
| 3 | under-constrained | 1.3641 | 1.549° | 19.754 % | 0.000 px |
| 4 | over-constrained | 0.0206 | 0.094° | 0.258 % | 0.187 px |
| 5 | over-constrained | 0.0378 | 0.117° | 0.510 % | 0.306 px |
| **6** | **over-constrained** | **0.0107** | **0.110°** | **0.069 %** | 0.316 px |

**Frame 1: solved vs. ground truth** after 6 pins and Solve & Key:

| parm | ground truth | solved | start |
|---|---|---|---|
| tx | 1.4000 | 1.3972 | 2.2000 |
| ty | 1.6500 | 1.6400 | 1.3000 |
| tz | 3.6000 | 3.5976 | 4.4000 |
| rx | −9.0000 | −8.9052 | −4.0000 |
| ry | 14.0000 | 13.9953 | 24.0000 |
| rz | 0.0000 | 0.0834 | −3.0000 |
| focal | 30.0000 | **29.9792** (−0.07 %) | 42.0000 |

**Frame 12: rough matchmove.** The six pins were copied from frame 1 with **C**, dragged onto
the frame-12 plate positions without noise, then solved with Solve & Key. Position, rotation and
focal errors are 0.0000 / 0.000° / 0.000 %. The camera is keyed at 1 and 12 and interpolates in
between.

With 3 pins the problem is under-constrained: 6 equations for 7 unknowns. The minimal-change
prior picks the smallest plausible camera change, so the focal is still far off, which is
expected. From 4 pins on, the pins determine the camera. The residual error of 0.07–0.5 % focal
comes from the injected 0.5 px click noise; the noise-free frame 12 solve is exact.

**Frame 1: a 7th pin on the wrong plate feature** (Robust Solve, the default). After the six pins
above, the test Ctrl+drags a 7th corner 75 px past its plate position:

| | position error | rotation error | focal error |
|---|---|---|---|
| 6 good pins | 0.0107 | 0.110° | 0.069 % |
| 6 good pins + the wrong one | 0.0097 | 0.110° | 0.026 % |

The HUD lists it as the only outlier, `7 (55.3 px)` (its anchor snapped to a mesh point near the
corner). It is drawn red, residual line and label included, which the test checks on the
rendered pixels. **Deactivate Worst Pin** picks it, and both steps are undoable.

## Reference display: scans and Gaussian splats (GUI)

The test scene also has the room as a dense scan (`/obj/scan`: bilinear subdivision,
triangulated, ±2 mm noise; 983 040 triangles) and as Gaussian splats (`/obj/splat`: 500 000
flat splats on the surfaces with Houdini's GSplat attributes, plus 400 faint floaters). At frame
20, with no pins on screen:

| Check | Result |
|---|---|
| Hidden-line removal, on the rendered viewport pixels | 11 visible and 6 hidden room edges are all drawn in Wireframe (All Edges). In Hidden Line, Hidden Line Ghost and Shaded, all visible edges are drawn and none of the hidden ones |
| Tool-view near clip | fitted to 0.0101 for the room (camera near clip stays 0.001) |
| W hotkey | cycles hidden line → ghost → shaded → wireframe |
| Dense scan: set as reference and first drawn | 0.26 s; first shaded draw (shading baked) 0.15 s; a redraw with framebuffer readback 21 ms |
| Dense scan: Ctrl+click on a corner, Points snapping | pin on a vertex 8 mm from the corner (vertices there are 3–80 mm apart), 52 ms including pin creation |
| Dense scan, Hidden Line: Ctrl+click on a hidden corner | snaps to a visible point, never to the hidden corner |
| Splats: recognised by `GS_Alpha`, left to the viewport | yes, 500 400 splats, not in the visible-object mask |
| Splats: Ctrl+click on 4 face centres, Free Surface Hit | within 0.2 mm of the true surface point |
| Splats: the same, Points snapping | on a splat centre, within 2.7 cm |
| Splat pick time | 13 ms for 500 400 splats |
| Scan and splat objects unchanged | checksums equal |

Hidden-line depth precision was also checked by hand with the room scaled ×100 (surfaces
500–1000 m away): with the fitted near clip (0.5) the hidden-line display stays clean. Before
the near clip was fitted (the camera's 0.001), a view of surfaces about 1 km away showed broken
edges and hidden edges.

## Headless solver tests (`tests/test_core.py`)

```
camera model: max |W - worldTransform| = 2.42e-08, max |uv - toNDC| = 9.38e-07
cheirality: min depth 0.031, focal 5.00, status over-constrained
1 pin : err 0.0000 px, camera moved 0.00011, focal 30.000 -> 29.981, rot change [ 3.001  5.601 -0.345]  [under-constrained]
2 pins focal free  : pinA err 0.0004 px, pinB err 0.0004 px, focal 33.35, move (x,y,z) [ 0.0017  0.0012 -0.0202]  [under-constrained]
2 pins focal locked: pinA err 0.0058 px, pinB err 0.0057 px, focal 30.00, move (x,y,z) [ 0.0266  0.0075 -0.3775]  [under-constrained]
GT  6 pins noise 0.0px: dpos 0.0000  drot 0.000 deg  dfocal 0.000%  rms 0.000px  over-constrained  3 it  5.8 ms
GT  6 pins noise 0.5px: dpos 0.0075  drot 0.032 deg  dfocal 0.066%  rms 0.453px  over-constrained  3 it  5.6 ms
GT 12 pins noise 0.5px: dpos 0.0056  drot 0.020 deg  dfocal 0.045%  rms 0.583px  over-constrained  2 it  6.1 ms
GT 30 pins noise 0.5px: dpos 0.0028  drot 0.028 deg  dfocal 0.005%  rms 0.675px  over-constrained  2 it  5.7 ms
hda: resolution matched, Solve & Key = 1 undo step, presets / copy / deactivate / delete buttons ok
locks: locked values bit-identical, roll lock exact (1.3e-15 rad) and off within 5 deg of vertical, focal clamped to 30.000
outlier: least squares dpos 1.1541 dfocal 19.69% flags [] | robust dpos 0.0545 dfocal 1.047% flags [6] | 6 good pins alone dpos 0.0564 dfocal 1.115%
polygons: packed and polysoup references are converted for the drawables
protected parms: ['ry', 'tx'] treated as locked, look-at disables solving
splats: depth error 0.0001 (splats) / 0.0052 (plain points), 2409000 splats picked in 69 ms
OK
```

* **Camera model:** 12 random rigs with non-uniformly scaled parents, pivots, pivot rotation,
  pre-transforms, every transform and rotate order, screen windows and pixel aspects. They are
  compared against Houdini's `worldTransform()` and VEX `toNDC()`, which works in float32.
* **1 pin:** pan/tilt only. The camera moves 0.1 mm and the focal changes 0.06 %.
* **2 pins:** pin A stays within 0.0004 px. With focal free the camera zooms; with focal
  locked it dollies (−0.38 along the view axis).
* **Lock Roll:** exact to 1.3e-15 rad with nothing locked, focal locked, position locked, RX
  locked, and RY and focal locked, and the rotation still changes in each case. Looking 88° down
  the lock is off and says so; looking 80° down it holds exactly.
* **Outlier:** 6 good pins (0.5 px noise) and 1 pin 86 px off. Least squares ends 19.7 % off in
  focal and flags nothing. The robust solve flags only the wrong pin and lands within 0.2 % focal
  and 1 cm of what the 6 good pins give alone. On pins that agree, it gives the least-squares
  camera bit for bit.
* **Splats:** two walls of flat splats (5 m and 8 m away) with faint floaters in front. 30
  rays hit the front wall within 0.1 mm, and the splat each one snaps to is on that wall. An
  opaque floater on the ray is hit instead, as it should be. The same scene as a plain point
  cloud (no splat attributes, 6 px point spacing) is hit within 4.5 mm. `_quat_matrices`
  matches `hou.Quaternion`.
* **Performance:** a 30-pin solve takes about 5 ms. In the GUI, a live drag event with 30 pins
  (solve plus camera parameter writes) takes about 2 ms, and a release with polish and keys about
  4 ms. When the robust solve has a pin to down-weight: 5–10 ms per drag event (7 to 100 pins)
  and 16–20 ms for the solve on release.

## Robust solve: random trials

Random scenes (`scene_points`), 0.5 px noise, one extra pin 20–200 px off in a random direction,
a start camera up to 0.4 units, 4° and 13 mm focal away, all 7 parameters free, 60 trials each.
Flagged means that pin and no other one:

| | median focal error | 90 % | worst | trials > 1 % | wrong pin flagged |
|---|---|---|---|---|---|
| 6 + 1 pins, least squares | 8.0 % | 27 % | 88 % | 53 | 3 |
| 6 + 1 pins, robust | 0.23 % | 0.55 % | 2.4 % | 3 | **60** |
| 10 + 1 pins, least squares | 3.1 % | 11 % | 30 % | 50 | 33 |
| 10 + 1 pins, robust | 0.07 % | 0.20 % | 0.43 % | 0 | **60** |

With 6 + 1 pins, the robust result is within 0.014 % focal (median) of the 6 good pins solved
alone, 0.86 % at worst; the 3 trials over 1 % are ones where the 6 good pins alone are that far
off too (up to 2.2 %). On the same trials without the wrong pin, robust and least squares give the
same camera.

Designs that did not make it, same trials (6 + 1 pins, 40 of them): Huber weights, from the
least-squares fit, with a threshold of 3 × median error: 28 trials over 1 %; Huber with a fixed
2 px threshold: 16–18; Cauchy weights from the least-squares fit: 15. With 7 pins, least squares
spreads the wrong pin's error over all of them. Starting Cauchy IRLS from the fit without the
pin with the largest leave-one-out statistic brought it to 2. The weights that ship,
min(1, (c/e)²), do as well (3 of 60 above) and leave pins that agree at least squares exactly.

## Redraw cost (#19)

Forced redraws (the 2D view nudged, then the framebuffer grabbed) with the 1M-triangle scan as
the reference in Hidden Line, 100 pins on the frame, 3 of them outliers:

| | redraw | `onDraw` (Python) | of that: pins / HUD |
|---|---|---|---|
| no pins | 19–22 ms | | |
| 100 pins, before | 50 ms | 10.8 ms | 7.1 / 1.6 ms |
| 100 pins, now | 51 ms | 9.2 ms | 6.1 / 0.9 ms |
| 100 pins, now, labels not drawn (experiment) | 24.5 ms | | |

The HUD solve is cached (a HUD update takes 0.56 ms instead of 0.95 ms), and the residual lines
are built with four bulk calls instead of about 900 per-point HOM calls. The 100 pin labels
(100 text draws) cost about 25 ms per redraw, almost all on the render side; the markers,
anchor dots and residual lines about 5 ms.

## Acceptance criteria

| Criterion | Result | Evidence |
|---|---|---|
| Viewport can't leave the camera view while the tool is active | PASS for everything but real input, see gaps | GUI test: forced camera switch is re-attached; the wheel zooms 2D without moving the camera; Space is consumed |
| Mesh transform and geometry unchanged | PASS | SHA-1 over `asCode(recurse)` of the object network, point positions, topology and world transform, before and after all operations |
| One aligned pin stays in place while a second is dragged | PASS | Maximum drift 0.0009 px over the whole drag (GUI); 0.0004 px (headless) |
| Locked parameters never change | PASS | Bit-identical values for every lock combination (headless); Lock Focal and Nodal presets during GUI drags |
| Unlock all / delete all / per-frame delete work and are undoable | PASS | GUI test; asset buttons tested headless |
| Pins and keys survive save and reopen | PASS | GUI test saves, clears, reloads and compares the pin JSON and every keyframe |
| 6 good pins recover focal within about 1 % and position within a small tolerance | PASS | 0.07 % and 1.1 cm with 0.5 px click noise; exact without noise |
| One wrong pin among 7 doesn't drag the camera and is flagged | PASS | GUI: 0.026 % focal, only that pin flagged; headless: as good as the 6 good pins alone, and 60 of 60 random trials flag only the wrong pin |
| Lock Roll is exact | PASS | 1.3e-15 rad for five lock combinations; off (and shown in the HUD) within 5° of a straight up/down view |

Other checks in the GUI test, all PASS (57 in total):

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
* The reference display checks listed above.
* Leaving the tool restores the camera and visible-object mask, also after the deferred work the
  exit queued has run (an earlier version re-attached the view and hid the reference again).
* Exclusions left behind by an earlier session are cleaned up, whether or not the node kept the
  user's mask.
* The HUD hint rows name the tool's hotkey symbols, and every one of them has a key assigned
  (`sv.hotkeyAssignments`); Delete Pin is bound to Del and Backspace.
* The HUD's solver status is cached between redraws.
* The HUD shows the roll lock as off for a camera looking straight down.
* A flagged pin is drawn to the end of `onDraw` (red residual line, label), checked on the
  rendered pixels. Exceptions in `onDraw` are silent: during this work a `TypeError` there went
  unnoticed by every other check.

## Known gaps

* **Real OS mouse and keyboard input wasn't exercised** (issues #1 and #3). Synthetic Qt events don't reach
  Houdini's viewer-state dispatch, and a computer-use check was declined. The GUI test calls the
  live state's handlers directly, with rays from the real viewport. Its mock events mimic HOM
  where that matters: `curViewport()` returns a fresh wrapper, and mouse coordinates are in
  view space. So the raw dispatch still needs a check by hand:
  * Ctrl/⌘ + click
  * Space + drag
  * the hotkeys, which are registered and assigned (all checked with `hotkeyAssignments`). Delete
    Pin is now a hotkey action as well; Del and Backspace still reach `onKeyEvent` first
