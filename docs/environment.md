# Environments and dependencies

The repository intentionally uses separate environments for current analysis,
API workflows, and historical provenance notebooks. The root environment is
small and supports only the active publication analyses; it does not attempt to
combine incompatible model-processing environments.

## Active analysis environment

The root [`requirements.txt`](../requirements.txt) supports:

- [`10_book_level_data_and_descriptive_analysis.ipynb`](../notebooks/05_analysis/10_book_level_data_and_descriptive_analysis.ipynb)
- [`11_hypothesis_analysis_and_paper_figures.ipynb`](../notebooks/05_analysis/11_hypothesis_analysis_and_paper_figures.ipynb)
- [`12_canonical_author_historical_analysis.ipynb`](../notebooks/05_analysis/12_canonical_author_historical_analysis.ipynb)
- [`src/analysis_utils.py`](../src/analysis_utils.py)

The notebook metadata records Python 3.12, which is the recommended version for
this environment. Create an isolated environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

The direct dependencies are NumPy and pandas for tabular analysis and
bootstrap calculations, Matplotlib and Seaborn for figures, and
IPython/IPykernel/JupyterLab for notebook execution. The active notebooks do
not import SciPy, scikit-learn, statsmodels, Krippendorff, Torch, Transformers,
Datasets, Accelerate, BookNLP, spaCy, or an API client.

## Gemini evaluation environment

[`workflows/gemini_evaluation/requirements.txt`](../workflows/gemini_evaluation/requirements.txt)
declares the dependencies used by the current evaluation code:

- `google-genai>=1.51.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `tqdm>=4.66.0`

Install and run this workflow from `workflows/gemini_evaluation/` as described
in its [README](../workflows/gemini_evaluation/README.md). The Developer API
uses `GEMINI_API_KEY`. The Vertex AI backend uses Application Default
Credentials plus a project and location supplied through command-line options
or `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.

## Gemini inference environment

[`workflows/gemini_inference/requirements.txt`](../workflows/gemini_inference/requirements.txt)
declares:

- `google-genai>=2.0.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `tqdm>=4.66.0`

Install this workflow in its own environment from
`workflows/gemini_inference/`. See the workflow
[README](../workflows/gemini_inference/README.md) and
[inference guide](../workflows/gemini_inference/INFERENCE_README.md). It uses
the same Developer API and Vertex AI credential mechanisms as the evaluation
workflow. Preparing and validating an input with `--prepare-only` does not
require an API call.

The two Gemini requirement files remain separate because they specify different
minimum `google-genai` versions. They should not be replaced by the root
analysis requirements.

## Historical encoder environment

[`03_encoder_model_evaluation.ipynb`](../notebooks/02_model_evaluation/03_encoder_model_evaluation.ipynb)
is the preserved Colab/GPU pipeline for BERT, RoBERTa, DistilBERT, and
ModernBERT. Its install cell requires:

- `transformers==4.48.3`
- `tokenizers`
- `accelerate`
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`

The notebook also imports PyTorch and asserts Transformers 4.48.3. PyTorch was
provided by the CUDA-enabled Colab runtime and is not version-pinned in the
notebook, so this is historical provenance rather than a portable locked
environment. Reproducing training requires a compatible GPU/CUDA PyTorch
installation and the original external model downloads and persistent output
locations.

## Historical BookNLP environment

[`09a_booknlp_sharded_batch.ipynb`](../notebooks/04_booknlp/09a_booknlp_sharded_batch.ipynb)
and
[`09b_booknlp_additional_42_book_batch.ipynb`](../notebooks/04_booknlp/09b_booknlp_additional_42_book_batch.ipynb)
create an isolated Python 3.10 environment inside a Python 3.12 Colab runtime.
The recorded installation resolved to:

- Python 3.10.12
- BookNLP 1.0.8
- PyTorch 2.4.1
- Transformers 4.30.0
- spaCy 3.7.5
- NumPy 1.26.4 under the notebook's `numpy<2` constraint
- `tokenizers<0.14`
- `huggingface_hub<0.20`
- `setuptools<82`
- spaCy model `en_core_web_sm` 3.7.1

These versions should remain separate from the encoder environment, which uses
Transformers 4.48.3. The BookNLP notebooks also depend on Colab, Google Drive,
downloaded BookNLP models, and historical input manifests.

## Historical cross-family LLM environment

[`04_cross_family_llm_evaluation.ipynb`](../notebooks/02_model_evaluation/04_cross_family_llm_evaluation.ipynb)
installs `openai`, `google-genai`, `groq`, `tqdm`, and `jsonlines` without
version pins and also uses pandas, NumPy, and Matplotlib. It is retained as the
original cross-family benchmark workflow, not as a current API compatibility
guarantee. It expects `OPENAI_API_KEY`, `GOOGLE_API_KEY`, and `GROQ_API_KEY`
through Colab Secrets.

## Other provenance notebooks

Notebooks 01–09 preserve the original annotation, model evaluation, corpus
construction, metadata recovery, and BookNLP workflows. Depending on the
notebook, additional historical packages include Krippendorff, SciPy,
scikit-learn, RapidFuzz, Datasets, Requests, tqdm, and `huggingface_hub`.
Several notebooks require Colab, Google Drive paths, external downloads,
historical intermediate files, GPUs, or third-party APIs. They are not
guaranteed to run from a fresh clone without those original inputs and services.

## Credential handling

No credentials are required for the active analysis notebooks. For API-backed
workflows:

- Gemini Developer API: set `GEMINI_API_KEY`.
- Gemini on Vertex AI: use Application Default Credentials and provide a Google
  Cloud project and location.
- Historical cross-family LLM notebook: provide `OPENAI_API_KEY`,
  `GOOGLE_API_KEY`, and `GROQ_API_KEY` through Colab Secrets.

Do not commit credentials or `.env` files. The repository's `.gitignore`
excludes `.env` and local virtual environments. Although the Gemini command-line
tools accept a direct `--api-key` argument, environment variables or Vertex
Application Default Credentials are preferred so keys do not enter shell
history or process listings.
