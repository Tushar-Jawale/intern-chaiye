# Business Entity Resolution — reproducible pipeline

Blocking (sparse IDF key matching per country) → pairwise string features →
LightGBM pair classifier → macro-F0.5-tuned threshold → `matching_results.tsv`
+ `candidate_pairs.tsv`.

No external data, APIs or pretrained models are used. Country is treated as an
open set: blocking runs separately for every country label present in the data
(France on test is just another label).

## Setup

```bash
pip install -r requirements.txt
```

Put the challenge data at `dataset/train/*.tsv` and `dataset/test/*.tsv`.

## Run (from this folder)

```bash
# 1. Preprocess all six source files + ground truth to parquet (~15 min)
python src/run_preprocess.py dataset preprocessed

# 2. Block a 100k train sample, train the matcher, block test, predict
python src/run_basic.py --preprocessed preprocessed --output output

# 3. Validate
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Step 2 skips any stage whose output already exists in `output/`
(`sample_candidate_pairs.tsv`, `matcher.txt`, `candidate_pairs.tsv`); delete the
file to redo that stage. To reuse a trained model:
`--model output/matcher.txt --meta output/matcher_meta.json`.

## What each stage prints

- Blocking (train sample): per-country and overall `BLOCKING RECALL`, candidates
  per entity, share of rows that hit `top_k`, and the rank at which true matches
  were found.
- Training: entities are split 70/15/15 into fit / threshold / report. The
  threshold is chosen on the threshold split only; `REPORT MACRO F0.5` (with
  macro precision, recall, singleton false-positive rate, and F0.5 by true-match
  count and country) is measured on the untouched report split and saved in
  `output/matcher_meta.json`.

## Files

| File | Role |
|---|---|
| `src/preprocess.py` | normalization, legal-suffix/abbreviation tables, Indic transliteration |
| `src/run_preprocess.py` | applies preprocessing to all files, writes parquet |
| `src/blocking.py` | candidate generation, recall measurement, `candidate_pairs.tsv` |
| `src/features.py` | pair features and the macro-F0.5 metric |
| `src/train.py` | LightGBM training, threshold selection, report |
| `src/predict.py` | scores `candidate_pairs.tsv`, writes `matching_results.tsv` |
| `src/run_basic.py` | end-to-end driver (`--self-test` runs on a toy dataset) |
