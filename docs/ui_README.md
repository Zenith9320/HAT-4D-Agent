# HAT-4D Annotation Tool UI Guide

[Chinese](ui_README_ch.md) | [Back to HAT-4D](README.md) | [Installation and launch guide](annotation_tool/README_eng.md)

This guide explains how to initialize 3D objects, propagate a 4D Gaussian
sequence, and repair failed intermediate frames in the Viser frontend after all
services are running. The frontend listens on `http://127.0.0.1:8897` by
default.

## SAM3D Initialization

### When to use it

- Initialize every target object at the first frame.
- Rebuild a later frame when it has severe shape deformation, position drift,
  incorrect depth, or another large unexplained error.
- Before entering SAM3D, select a frame where every object to be rebuilt is
  clearly visible in the `input` view. SAM3D cannot reliably generate an object
  that is fully occluded or outside the image. This is especially important
  when using `Total cover`, because that mode replaces all 3D content in the
  current frame.

### Enter generation mode

1. Expand the `SAM 3D Gen` panel.
2. Click `Begin SAM3D Gen`.
3. Enable `Pixel Pick Mode` to enter SAM point-prompt segmentation mode.

> **Full replacement of an intermediate frame:** enable
> `SAM 3D Gen > Total cover (clear session PLY on begin)` **before** clicking
> `Begin SAM3D Gen`. This clears the existing session PLY for the current frame,
> so every visible object in that frame must then be regenerated. Do not enable
> it for first-frame initialization or when existing objects must be retained.

### Segment and generate the first object

1. Leave `Negative Point` and `Eraser Point` disabled, then click the target
   object in the image to add positive points.
2. To exclude an incorrect region, enable `Negative Point` and click the region
   that should not belong to the object.
3. To remove a prompt, enable `Eraser Point` and click that prompt. You can also
   use `Undo Point` to revert the latest prompt or `Clear Points` to remove all
   prompts for the current object.
4. Confirm that the mask covers the complete target without including another
   object, then click `Generate 3D (Current Mask)`.

<p align="center">
  <img src="annotation_tool/docs/assets/ui/sam3d-point-prompts.png" width="92%" alt="SAM3D point prompts and current mask">
</p>

Generation and the default pose optimization typically take about 10 seconds,
depending on the GPU, model cache, and input. Keep `Pose optimize after GEN`
enabled, and wait for both generation and pose-optimization completion messages
before continuing.

<p align="center">
  <img src="annotation_tool/docs/assets/ui/sam3d-generated-object.png" width="92%" alt="Generated SAM3D object after pose optimization">
</p>

### Correct pose and depth

After generation, return to the `input` view and inspect the silhouette, scale,
depth, and contact relationships between objects.

1. If the object is far from `x=0` along the world X axis, drag its transform
   controls to make a coarse correction.
2. Expand `SAM 3D Gen > Pose Opt`.
3. Enable `Pose opt hard lock world X` to preserve the manually adjusted X
   position.
4. Click `Optimize Pose (Mask Align)` and wait for mask-alignment optimization
   to finish.
5. Check silhouette alignment and depth again in the `input` view. If they are
   still incorrect, adjust the transform or other axis-lock options and rerun
   optimization.

<p align="center">
  <img src="annotation_tool/docs/assets/ui/sam3d-pose-opt-and-propagate.png" width="92%" alt="SAM3D pose optimization and L4GM propagation controls">
</p>

### Generate the remaining objects and save

1. Click `New Object` to create the next object slot.
2. Repeat point-prompt segmentation, 3D generation, inspection, and pose
   optimization for each independently moving object.
3. After all objects have been generated and their relative positions are
   correct, click `Add All Objects & Exit`.

`Add All Objects & Exit` writes the objects in the current SAM3D session into
the current frame and exits segmentation mode. Do not click it while an object
is still missing.

## 4D Propagation

### Generate the next segment

1. Finish all objects in the current frame and exit SAM3D mode.
2. Expand `L4GM_4DGEN`.
3. Click `Generate next segment` to generate the next Gaussian sequence from
   the current frame.

By default, one segment contains the next 16 frames. It usually takes about 10
seconds on the reference machine, but the actual time depends on the GPU and
cache state. After generation, use `Play / Pause` at the top of the page to
inspect silhouettes, geometry, relative depth, object contact, and motion
continuity.

We recommend pausing after every 16 frames. Click `Generate next segment` to
start generation. When the label changes to `Pause Generate`, click it again so
the process pauses after completing the current segment. Continue only after
that segment has been inspected.

### Repair a failed intermediate frame

When an intermediate frame has severe deformation, drift, or incorrect depth:

1. Pause 4D propagation and navigate to the failed frame.
2. Confirm that every object to be rebuilt is visible in the `input` view.
3. Enable `Total cover` **before** clicking `Begin SAM3D Gen`.
4. Click `Begin SAM3D Gen` and reinitialize every visible object in the frame.
5. Click `Add All Objects & Exit` to save the complete replacement.
6. Verify the repaired frame, then continue with `Generate next segment`.

<p align="center">
  <img src="annotation_tool/docs/assets/ui/sam3d-total-cover.png" width="92%" alt="SAM3D total cover workflow for replacing a failed frame">
</p>

> **Do not omit `Total cover`:** without it, newly generated objects can be
> mixed with the old PLY for the frame, causing duplicate objects, residual
> geometry, or incorrect occlusion.

## Segment Checklist

- Every object silhouette matches the `input` image without obvious missing or
  extra regions.
- Object scale, depth ordering, and contact relationships are plausible.
- Playback has no sudden stretching, collapse, rotation jumps, or position
  drift.
- A repaired frame contains no residual old PLY and omits no required object.
- Continue propagation only after the current segment is correct, so errors do
  not accumulate in later segments.
