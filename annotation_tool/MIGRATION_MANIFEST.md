# Migration Manifest

The annotation runtime was selectively copied from source commit
`4d3558173541370300892ad24217140fd640c7b5`. The local source location is not
part of this distribution.

## Included

| Source-relative path | Purpose |
| --- | --- |
| `viser_frontend.py` | Viser browser frontend and workspace orchestration |
| `roi_backend.py` | ROI selection, L4GM/SV4D generation, and repair API |
| `mvtap_backend.py` | Optional multi-view point-tracking API adapter |
| `good_optimizer.py` | Gaussian repair and optimization implementation |
| `headless_3d_gen.py` | Non-interactive 3D Human-Agent entry point |
| `headless_4dgen.py` | Non-interactive 4D Human-Agent entry point |
| `core/gs.py` | Gaussian renderer used by the frontend and optimizers |
| `utils/*.py` | Camera, geometry, image, and PLY utilities |
| `metric/evaluator.py`, `metric/utils/*.py` | Online metric adapters used by the frontend and optimizer |
| `model4dgen/*.py` | L4GM and SV4D backend adapters |
| `viewer_plugins/manual_edit/*.py` | Manual Gaussian editing plugin |
| `viewer_plugins/record_ops/*.py` | Annotation operation recording plugin |
| `viewer_plugins/sam3d_gen/` Python files | SAM and SAM3D frontend/backend integration |
| `viewer_plugins/vlm_assistor/` Python files | VLM agents, operators, UI, and backend bridge |
| `model4dgen/L4GM_MoGA/` | Modified L4GM/MoGA inference source required by the ROI backend |
| `model4dgen/generative-models/` | Modified SV4D2 inference source and HAT-4D encoder configuration |
| `requirements.txt`, `requirements_roi.txt` | Base and ROI/L4GM Python dependencies |
| `LICENSE` | License shipped with the source runtime |
| `model4dgen/*/MODIFICATIONS.md` | Upstream revision and HAT-4D modification notices |

The copy deliberately preserves module locations and public interfaces to keep
the frontend/backend protocol compatible with the source-tested workflow.

## Publication Additions

The following files were written in this repository rather than copied from
the migration source:

| Path | Purpose |
| --- | --- |
| `README.md`, `README_eng.md`, `README_ch.md` | Bilingual installation, launch, annotation, and recovery guides |
| `docs/qwen3_vl_8xh200_test.md` | External Qwen3-VL GPU verification protocol |
| `external_dependencies.json` | Publicly fetchable source URLs and immutable revisions |
| `external_dependencies.unavailable.json` | Source provenance that cannot be publicly installed |
| `requirements-dev.txt` | Test and lint dependencies |
| `scripts/install_external_dependencies.py` | Safe clone and revision verification |
| `scripts/check_publication.py` | Artifact, path, credential, and comment scan |
| `scripts/create_demo_data.py` | Tiny deterministic generated input |
| `scripts/smoke_test_services.py` | Model-independent service checks |
| `tests/` | Distribution, safety, protocol, demo, and documentation tests |

These annotation-tool additions contain no copied dataset samples, model
weights, or generated inference results. The repository root separately
includes two reviewed demo subsets; their provenance and data license are
documented in `asset/README.md`.

## External Code

Public external repositories are installed separately at pinned revisions.
Their source trees, Git histories, caches, and weights were not copied from
local checkouts.

| Dependency | Revision |
| --- | --- |
| Official SAM3D Objects | `f91db411c50efee93d8db7aeb323885650f6f722` |
| Segment Anything | `6fdee8f2727f4506cfbbe553e23b895e27956588` |
| Qwen3-VL | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` |

The migration source recorded `261434711d6db969819cd122573ed603cdc0f4db` for
L4GM and `7a2e81f16e6cca202e9f43e9842a9abef6aa2431` for generative-models. These
identifiers are not fetchable from the named public upstream repositories and
are provenance identifiers, not publicly reproducible Git revisions. Required
modified runtime files are therefore bundled with upstream licenses and HAT-4D
change notices. The optional VLM toolkit at
`250b7a4d97edb211e2d8373d6db34c2e26887102` remains provenance-only in
`external_dependencies.unavailable.json`; the direct HTTP Qwen backend does not
require it. MV-TAP is not in the lock because the migration source did not
provide a publishable implementation and revision for that optional adapter.

## Excluded

- The full dataset, input collections other than the two reviewed demo subsets
  under the repository root `asset/` directory, model checkpoints, and Hugging
  Face caches.
- Workspaces, generated PLY files, masks, renderings, videos, metrics, logs,
  spreadsheets, JSONL experiment output, and temporary debug files.
- Batch experiment launchers, ablations, result rendering, and score
  aggregation scripts.
- Source repository history, bytecode, editor settings, local assistant files,
  machine-specific configuration, and all symbolic-link targets.
- Training entry points and experiment-only tests that are unrelated to the
  published inference runtime.
- Obsolete `models_abandon.py`, `models_viz_tmp_1215_synced.py`, and the
  non-runnable Objaverse and Memory4D training providers.
- Training-only data modules, including CIFAR10 and MNIST examples, and the
  former L4GM `train_utils.py`; inference keeps only a small runtime seed helper.
