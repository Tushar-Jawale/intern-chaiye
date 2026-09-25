# Business Entity Resolution — reproducible pipeline

## v3 pipeline (current)

Record-centric: every Source 2/3 record picks at most one Source-1 owner.

1. `norm3.py` learns an Indic→Latin token dictionary from train pairs (plus a
   phonetic fallback that works for any Indic script), and normalizes names,
   addresses, units, states and numbers per country label.
2. `retrieve3.py` finds the top-10 Source-1 records for every Source 2/3 record
   with weighted sparse keys (words, word pairs, phonetic skeletons, numbers,
   units, n-grams).
3. Stage 1: pair model on all 10 candidates (two query folds, out-of-fold scores).
4. Stage 2: top-3 candidates per record with context (runner-up gap, how
   contested the Source-1 entity is, similarity to the records already clustered
   on it), scored by a LightGBM + XGBoost + CatBoost blend (weights tuned on
   held-out entities; add `rf` via `--s2-models lgb,xgb,cat,rf`).
5. Each record goes to its best candidate; the keep/drop rule (threshold, or
   per-entity expected-F0.5) is tuned on held-out entities.

Source-1 entities are hashed into report 4% / tune 4% / train 92%. `REPORT
MACRO F0.5` is macro F0.5 over the report entities (singletons included) with
all records of the country in play; nothing from the report entities is used
for fitting or tuning.

### Kaggle (GPU T4, "Save & Run All" so it survives the browser closing)

Upload `student_resource/src` and `student_resource/utils` as one dataset and
the challenge data as another, then:

```bash
!pip install -q "rapidfuzz>=3.6"
!cp -r /kaggle/input/<code>/src /kaggle/input/<code>/utils /kaggle/working/
%cd /kaggle/working
!python src/pipeline3.py all --data /kaggle/input/<data>/dataset --work work3 --output output
!python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir /kaggle/input/<data>/dataset/test --check-ids
```

Every step skips outputs that already exist under `--work` (`norm/`, `ret/`,
`models/`, `p1/`, `s2data/`), so a session can be resumed by attaching the
previous version's `/kaggle/working` output. `--force` redoes a step.
Steps can also be run one at a time: `prep`, `retrieve`, `stage1`, `stage2`,
`train2` (re-tune stage 2 from saved features), `predict`.

Quick local check on a 10% hash slice of train (no test outputs):

```bash
python src/norm3.py learn --data dataset --work work3
python src/norm3.py build --data dataset --work work3
python src/pipeline3.py retrieve --work work3_dev --dev-frac 0.1 --skip-test   # work3_dev/norm -> work3/norm
python src/pipeline3.py stage1 --work work3_dev --skip-test --s1-fit-queries 80000
python src/pipeline3.py stage2 --work work3_dev --skip-test --s2-fit-queries 200000
```

On the 10% slice this gave REPORT MACRO F0.5 = 0.992 (India 0.992, US 0.992).
The full world has ~10x more look-alike Source-1 records per query, so expect
the full-scale number to be somewhat lower.

## v1 pipeline (baseline, 0.826)

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
