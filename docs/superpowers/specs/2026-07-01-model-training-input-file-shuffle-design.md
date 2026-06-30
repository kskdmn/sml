# Model Training Input File Shuffle Design

## Context

Model training currently discovers input shards in `v1/src/train_sml.py` by
matching configured file names, skipping hidden files, and returning matches
sorted by file name. Tokenizer training has a separate discovery path in
`v1/src/train_tokenizer.py` and must keep its existing sorted order.

The requested change is to shuffle input files for model training only.

## Design

Keep `discover_input_files()` deterministic and sorted. It remains responsible
only for finding matching files in a stable order.

Add a small model-training helper that accepts the discovered file tuple and a
seed, copies the files into a list, shuffles that list with a local
`random.Random(seed)`, and returns a tuple. `train_model()` calls this helper
after discovery and before building dataloaders.

Add `shuffle_input_files: bool = True` to `TrainingConfig` so model training
uses shuffled input files by default while tests and callers can opt out.

## Data Flow

1. `train_model()` calls `discover_input_files()` and receives sorted matching
   paths.
2. If `training_config.shuffle_input_files` is true, `train_model()` replaces
   that tuple with the deterministic shuffled tuple.
3. Each epoch builds its dataloader from the same shuffled tuple.
4. `iter_texts()` and progress logging continue to report the actual file being
   read.

## Error Handling

Existing discovery errors stay unchanged. Missing input directories still raise
`FileNotFoundError`, and empty matched file sets still raise `FileNotFoundError`.
The shuffle helper has no new failure mode because it only reorders an already
validated sequence.

## Testing

Add tests in `v1/tests/test_train.py` for the model-training shuffle helper:

- the same seed produces the same shuffled order;
- a different seed can produce a different order for a multi-file input set;
- the helper returns a tuple and does not mutate the caller's sequence.

Keep existing discovery tests asserting sorted order. Tokenizer training code is
not changed.
