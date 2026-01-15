# Data & Manifest

Manifest structure (beauty_all-TORCH-001-ML-15.json):
- First line: `"record_type": "set"` with `output_root` and `output_rel` base info.
- Following lines: `"record_type": "sample"` with `output_rel` pointing to TIFFs and `metadata.split` (`train` or `val`).
- Loader uses `output_root` + `output_rel` to read images. Missing files are skipped with a warning.

Conditions:
- Default spec: `temperature:1, wind:3, wind_mag:1`.
- `wind_mag` is auto-computed from `wind` if absent; any missing field is zero-filled.
- Extendable via `--condition_spec name:dim,...` (use `none` for unconditional).
- At train time, `cond_dropout_prob` zeroes some condition vectors for classifier-free style training.

Transforms:
- Resize to target `resolution`, center-crop or random-crop, optional random flip.
- If `--preserve_input_precision` is set, images stay float32 (no forced 8-bit) and channels are clamped to 3.
- Otherwise, standard `ToTensor` + normalize to [-1, 1].

DataLoader notes:
- Uses PyTorch DataLoader with `num_workers` configurable. A picklable no-op flip is used when flips are disabled.
- Batches yield `pixel_values` and `cond` (empty tensor when conditioning is disabled or not requested).
