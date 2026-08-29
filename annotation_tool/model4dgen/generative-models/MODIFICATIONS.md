# HAT-4D Modifications

This subtree is derived from Stability AI Generative Models and has been
modified for HAT-4D. The migration source recorded
`7a2e81f16e6cca202e9f43e9842a9abef6aa2431` as its baseline identifier, but that
identifier is not fetchable from the public upstream repository and must not be
treated as a publicly reproducible revision.

The changes add the SV4D2 first-stage encoder configuration and HAT-4D
inference helpers used for multi-view initialization and temporal inpainting.
Unrelated training and application components are excluded where possible.

The upstream MIT license is retained in `LICENSE-CODE`. Model checkpoints are
downloaded separately and are not licensed by `LICENSE-CODE`. See
`../../MIGRATION_MANIFEST.md` for the distribution-level provenance record.
