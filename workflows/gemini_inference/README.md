# Local Gemini Interiority Evaluation

This folder runs Gemini-only interiority classification tests on an annotation CSV. It runs two evaluation groups by default:

- Diagnostic tests: legal-label format, codebook example recovery, codebook order invariance, generic-label consistency, and full-codebook dev performance.
- Final zero-shot test: full-codebook prediction on the held-out test split, with item-level outputs, summary metrics, model comparison, and confusion matrices.

The CLI accepts one or more Gemini model names and can run each model at one or
more thinking levels. It writes predictions, metrics, checkpoints, failures,
usage, and run configuration to local CSV/JSON/JSONL files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required Inputs

You must provide:

- Gemini access method:
  - `--backend developer` with `GEMINI_API_KEY`, or
  - `--backend vertex` with Google Cloud credentials, `--project`, and `--location`
- At least one model name with `--models`

The default data file is `df_annotation_final.csv`. If your data file has a different path, pass it with `--data`.

The input CSV must include these columns:

- `split`
- `paragraph`
- `gold_label`

Optional but useful columns:

- `title`
- `index`

## Auth

### Gemini Developer API

Set your API key in the terminal:

```bash
export GEMINI_API_KEY="your-api-key"
```

Then run with:

```bash
python -m gemini_interiority.cli \
  --data df_annotation_final.csv \
  --output-dir results/local_gemini \
  --models gemini-3.5-flash \
  --backend developer
```

### Vertex AI

Authenticate with Google Cloud:

```bash
gcloud auth application-default login
```

Then run with:

```bash
python -m gemini_interiority.cli \
  --data df_annotation_final.csv \
  --output-dir results/local_gemini \
  --models gemini-3.5-flash \
  --backend vertex \
  --project your-google-cloud-project \
  --location global
```

Use `--backend vertex` for Google Cloud / Vertex AI access, including Gemini Enterprise Agent Platform environments that expose Gemini through Vertex AI credentials.

## Common Runs

Run diagnostics and final zero-shot. Both run by default:

```bash
python -m gemini_interiority.cli \
  --data df_annotation_final.csv \
  --output-dir results/local_gemini \
  --models gemini-3.5-flash \
  --backend developer
```

Run only the held-out test split and skip diagnostics:

```bash
python -m gemini_interiority.cli \
  --data df_annotation_final.csv \
  --output-dir results/local_gemini_test_only \
  --models gemini-3-flash-preview \
  --backend developer \
  --run final
```

Compare one model across multiple thinking levels in a single run:

```bash
python -m gemini_interiority.cli \
  --data df_annotation_final.csv \
  --output-dir results/gemini-3.5-flash-thinking-comparison \
  --models gemini-3.5-flash \
  --thinking-levels minimal low medium high \
  --max-output-tokens 4096 \
  --backend developer \
  --run final
```

`--thinking-levels` takes precedence over `--thinking-level`. The runner creates
the cross-product of every value passed to `--models` and every requested
thinking level. Each variant has a separate prediction cache. The final model
comparison includes a `thinking_level` column and names such as
`gemini-3.5-flash [thinking=medium]`.

Resume an interrupted final run from its per-row checkpoint:

```bash
python -m gemini_interiority.cli \
  --data df_annotation_final.csv \
  --output-dir results/gemini-3.5-flash-thinking-comparison \
  --models gemini-3.5-flash \
  --thinking-levels minimal low medium high \
  --max-output-tokens 4096 \
  --backend developer \
  --run final \
  --resume
```

The runner also prints a complete resume command if it stops after consecutive
API failures.

Run only a small sample. Diagnostics use a sampled dev subset; final zero-shot
uses the first `--pilot-n` test rows:

```bash
python -m gemini_interiority.cli \
  --data df_annotation_final.csv \
  --output-dir results/local_gemini_pilot \
  --models gemini-3.5-flash \
  --backend developer \
  --pilot \
  --pilot-n 5
```

For preview/pro models with low requests-per-minute limits, add a delay between requests:

```bash
python -m gemini_interiority.cli \
  --data df_annotation_final.csv \
  --output-dir results/local_gemini_pro_pilot \
  --models gemini-3.1-pro-preview \
  --backend developer \
  --pilot \
  --pilot-n 5 \
  --request-delay-seconds 3 \
  --max-retries 5
```

## Pilot And Checkpoint Behavior

Before a normal final run, each uncached model/thinking-level variant runs an
automatic pilot check on the first `--pilot-n` test rows. If any pilot output is
invalid or any request ends in an error, that variant's full dataset is not
started and the command exits with `Pilot check failed`.

`--pilot` has a different purpose: it turns the requested run itself into a
small-sample run. In this mode, final zero-shot evaluates only the first
`--pilot-n` rows instead of running the automatic pilot followed by all test
rows.

During a full run, the prediction JSONL cache and `resume_state.json` are updated
after every completed row. Progress, token usage, estimated cost, item-level CSV,
summary files, `run_config.json`, and `failed_items.csv` when needed are refreshed
every `--status-every` rows. The default status interval is 30.

Transient `429`, `500`, `502`, `503`, `504`, timeout, and connection failures are
retried with exponential backoff and jitter. `--max-retries 5` means up to six
attempts including the initial request. A request that still fails is recorded
in `failures.json`, and the full runner continues. After three consecutive
requests fail after retry handling, the runner checkpoints, stops cleanly, and
prints a resume command.

## Configurable Options

These options have defaults but can be changed:

- `--data`: input CSV path. Default: `df_annotation_final.csv`
- `--output-dir`: output folder. Default: `results/local_gemini`
- `--run`: evaluation group to run: `all`, `diagnostics`, or `final`. Default: `all`
- `--backend`: `developer` or `vertex`. Default: `developer`
- Sampling temperature is not sent in API requests; every model always uses its default temperature
- `--max-output-tokens`: requested output token budget. Default: `4096`
- `--thinking-level`: one classification thinking level. Default: `high`
- `--thinking-levels`: run every model at multiple thinking levels in one invocation; takes precedence over `--thinking-level`
- `--request-timeout-seconds`: timeout for each API attempt. Use `0` to disable. Default: `60`
- `--status-every` / `--status_every`: print progress and refresh final CSV/JSON checkpoints every N rows. Default: `30`
- `--resume`: resume a final run from `resume_state.json`
- `--seed`: sampling seed for diagnostic pilot mode. Default: `42`
- `--pilot`: run a small local test instead of full dev/test sets. Default: off
- `--pilot-n`: pilot size. Default: `5`
- `--request-delay-seconds`: delay after each request. Default: `0.0`
- `--max-retries`: retries after the initial request for transient failures. Default: `5`
- `--input-cost-per-1m-tokens`: override the input-token price used for estimates
- `--output-cost-per-1m-tokens`: override the output-token price used for estimates; thinking tokens are included in output usage
- `--api-key-env`: environment variable name for Developer API key. Default: `GEMINI_API_KEY`
- `--api-key`: pass a Developer API key directly. Prefer the environment variable for normal use.

Thinking-level support is model-dependent. The current classification path sends
`thinking_level` directly, so use it with Gemini models that support that field.

## Cost Estimates

Cost reporting uses token usage returned by Gemini/Vertex. `output_tokens`
includes both candidate output and thinking tokens. Built-in paid-tier estimates
are:

- `gemini-3-flash-preview`: $0.50 per 1M input tokens and $3.00 per 1M output tokens
- `gemini-3.5-flash`: $1.50 per 1M input tokens and $9.00 per 1M output tokens
- `gemini-3.1-pro-preview`: $2.00 per 1M input tokens and $12.00 per 1M output tokens

For other models, pass both cost override flags. These are estimates only; actual
billing and free-tier eligibility depend on the account and backend.

## Outputs

Outputs are written under `--output-dir`:

- `predictions/*.jsonl`: cached raw diagnostic predictions
- `behavioral_results/all_item_level_predictions.csv`
- `behavioral_results/all_summary_metrics.csv`
- `behavioral_results/model_comparison_master.csv`
- `final_test_results/predictions/*.jsonl`: cached final-test predictions
- `final_test_results/pilot_item_level_predictions.csv`: most recently saved automatic pilot predictions
- `final_test_results/item_level_predictions.csv`
- `final_test_results/final_test_item_level_predictions.csv`
- `final_test_results/final_test_summary_metrics.csv`
- `final_test_results/final_test_model_comparison.csv`
- `final_test_results/final_test_confusion_matrices.csv`
- `final_test_results/failed_items.csv`: created when invalid/error rows exist
- `final_test_results/failures.json`: exhausted API request failures
- `final_test_results/resume_state.json`: per-row final-run checkpoint and resume command
- `final_test_results/run_config.json`: settings, status, usage, cost, and output paths

Use `--resume` to continue a partial final-run cache without repeating completed
rows. A complete cache is reused automatically. Without `--resume`, a partial
final cache is not treated as the continuation point. Delete the relevant JSONL
cache file, or use a new output directory, to force a fresh run.
