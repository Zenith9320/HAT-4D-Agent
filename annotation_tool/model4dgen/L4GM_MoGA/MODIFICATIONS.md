# HAT-4D Modifications

This subtree is derived from NVIDIA Toronto AI Lab's public L4GM source and has
been modified for HAT-4D. The migration source recorded
`261434711d6db969819cd122573ed603cdc0f4db` as its baseline identifier, but that
identifier is not fetchable from the public upstream repository and must not be
treated as a publicly reproducible revision.

The changes include multi-object Gaussian generation, MoGA routing and
attention, memory-aware inference, camera and pose alignment, and integration
with the HAT-4D ROI service. Training-only and obsolete local experiment files
are excluded from this distribution.

The upstream Apache License 2.0 is retained in `LICENSE`. See
`../../MIGRATION_MANIFEST.md` for the distribution-level provenance record.
