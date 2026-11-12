# Corpus-passage inference runner

`run_inference.py` is separate from the annotated evaluation CLI. It accepts a
prepared passage-level corpus CSV, creates a compact runtime `llm_input.csv`,
sends only the `text` field to the existing full-codebook prompt, and writes
inference outputs without gold-label metrics.

The checked-in default input,
`data/frozen/corpus_inference/llm_input_armstrong.csv`, contains only the 29-book
Armstrong/core subset. The corresponding frozen predictions are
`data/frozen/corpus_inference/passage_predictions_armstrong.csv`. Neither file
represents the complete 1,348-book corpus. Pass `--data` with another prepared
passage-level CSV to run the same pipeline on a different corpus.

Run the commands below from `workflows/gemini_inference/`.

## Prepare and validate files without API calls

```bash
python run_inference.py \
  --data ../../data/frozen/corpus_inference/llm_input_armstrong.csv \
  --output-dir inference_results \
  --model gemini-3.5-flash \
  --prepare-only
```

A prepared directory contains checkpoint state. Continue it with `--resume`
and the exact same model/prompt configuration, or choose a new output directory
for a fresh run.

## Pilot then full inference

```bash
export GEMINI_API_KEY="your-api-key"
python run_inference.py \
  --data ../../data/frozen/corpus_inference/llm_input_armstrong.csv \
  --output-dir inference_results \
  --model gemini-3.5-flash \
  --backend developer \
  --resume
```

The first five passages are an automatic pilot. Their valid predictions remain
in the checkpoint and are skipped by the full phase. Run only the pilot with
`--pilot-only`.

Resume an interrupted run with the command printed in `run_config.json`, or:

```bash
python run_inference.py \
  --data ../../data/frozen/corpus_inference/llm_input_armstrong.csv \
  --output-dir inference_results \
  --model gemini-3.5-flash \
  --backend developer \
  --resume
```

Resume is refused when the model, prompt digest/version, compact input digest,
backend, thinking level, valid-label set, or output-token setting differs.
Successful passages are skipped; failed and invalid passages are attempted
again. The checkpoint is appended and synced after every passage.

Progress is printed every `--status-every N` passages (default: `100`). Each
progress line reports passages run as a count and percentage, successful,
failed, invalid, and remaining counts, cumulative runtime, and estimated cost.
At the same interval, the final CSV files, validation report, and
`run_config.json` are refreshed; the JSONL checkpoint is still synced after
every passage.

For `gemini-3.5-flash`, estimated cost uses $1.50 per 1 million input tokens and
$9.00 per 1 million output tokens. Gemini thinking tokens are already included
in the output-token total returned by the existing client.

## Vertex AI

Install the requirements and authenticate with Application Default
Credentials:

```bash
pip install -r requirements.txt
gcloud auth application-default login
```

Use a fresh output directory because checkpoints are tied to their backend,
model, and prompt configuration:

```bash
python run_inference.py \
  --data ../../data/frozen/corpus_inference/llm_input_armstrong.csv \
  --output-dir inference_results_vertex \
  --model gemini-3.5-flash \
  --backend vertex \
  --project YOUR_GOOGLE_CLOUD_PROJECT \
  --location global
```

The command automatically runs the pilot before starting full inference. Add
`--pilot-only` to stop after the pilot, then rerun the same command with
`--resume` (and without `--pilot-only`) to continue.

## Outputs

The runtime output directory contains:

- `llm_input.csv` (a generated compact copy for that run, distinct from the committed frozen Armstrong input)
- `checkpoint.jsonl`
- `passage_predictions.csv`
- `book_summary.csv`
- `run_config.json`
- `validation_report.md`

`book_summary.csv` includes `is_core`, derived at book level from whether any
input passage has `is_core_extra=True`. Every book has a `base` row; only core
books also have an `expanded` row.
