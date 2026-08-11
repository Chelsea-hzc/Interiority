# Interiority in British Fiction

This project studies literary interiority—textual access to characters' inner
experience—across a large corpus of British fiction. It combines human passage
annotation, evaluation of Gemini predictions against human gold labels,
Gemini-based classification of unlabeled corpus passages, and book-level
historical analysis.

The repository contains the human-annotation evaluation workflow, the corpus
inference workflow, a finalized dataset of 1,348 books, descriptive analyses,
H1–H4 hypothesis analyses, canonical manuscript figures, and appendix and
supporting numerical outputs. The research workflow is:

`human annotation → Gemini evaluation against gold labels → selected Gemini interiority classifier → corpus inference on unlabeled passages → finalized book-level dataset → descriptive and H1–H4 analyses → manuscript and appendix outputs`

## Repository Structure

```text
Interiority/
├── Gemini_evaluation/
├── Gemini_interiority/
├── analysis/
│   ├── book_level_data_and_descriptive_analysis.ipynb
│   ├── hypothesis_analysis_and_paper_figures.ipynb
│   ├── figures/
│   └── outputs/
│       ├── descriptive/
│       │   └── tables/
│       └── hypotheses/
│           ├── figures/
│           └── tables/
├── analysis_utils.py
├── book_summary_analysis.csv
└── book_summary_analysis_schema.csv
```

### `Gemini_evaluation/`

This workflow evaluates Gemini predictions against human gold-label
annotations. It contains the annotated evaluation data, evaluation code,
metrics, tests, requirements, and detailed documentation.

### `Gemini_interiority/`

This is the corpus-inference workflow. It applies the selected Gemini
interiority classifier to prepared corpus passages for which human gold labels
are unavailable, producing passage-level predictions and book-level
aggregations. It is distinct from the gold-label evaluation workflow.

### `analysis/`

The two publication notebooks perform the descriptive and H1–H4 analyses.
`analysis_utils.py` provides their shared plotting constants, deterministic
book-level bootstrap helpers, and figure-export utilities.

`analysis/figures/` is the canonical publication-figure directory. It contains
manuscript Figures 1–7 and the H2 and H3 appendix robustness figures, each in
PDF and PNG format.

`analysis/outputs/` contains supporting and reproducibility outputs:

- `analysis/outputs/descriptive/tables/` contains numerical results underlying
  the descriptive manuscript figures.
- `analysis/outputs/hypotheses/figures/` contains supporting H1–H4 figures that
  are not canonical manuscript figures.
- `analysis/outputs/hypotheses/tables/` contains appendix estimate tables and
  detailed H1–H4 numerical outputs.

## Data

### Human annotations

`Gemini_evaluation/df_annotation_final.csv` is the human-annotated passage
dataset used to evaluate the Gemini classifier. It contains 717 passages: 357
in the training split, 120 in the development split, and 240 in the test split.

### Final book-level analysis dataset

`book_summary_analysis.csv` is the finalized one-row-per-book dataset used for
the manuscript analyses. It contains 1,348 rows representing 1,348 unique
books. `book_summary_analysis_schema.csv` documents the columns, data types,
missingness, and unique-value counts in that dataset.

Manuscript-level analysis normally begins from the included finalized
book-level dataset. Earlier corpus-preparation stages are outside the primary
publication-analysis workflow documented here; not every upstream raw-data
artifact is included.

## Gemini Evaluation Against Human Annotations

`Gemini_evaluation/` implements the sequence:

`human-labeled passages → Gemini predictions → comparison with gold labels → evaluation metrics`

From that directory, install its dependencies, supply a Gemini Developer API
key through `GEMINI_API_KEY`, and run the current evaluation CLI:

```bash
cd Gemini_evaluation
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export GEMINI_API_KEY="<your-api-key>"
python -m gemini_interiority.cli \
  --data df_annotation_final.csv \
  --output-dir results/local_gemini \
  --models gemini-3.5-flash \
  --backend developer
```

The workflow runs diagnostic evaluations and a held-out zero-shot test by
default. Vertex AI authentication is also supported. See
`Gemini_evaluation/README.md` for backend configuration, CLI options,
checkpoint behavior, and output definitions.

## Gemini Corpus Inference

`Gemini_interiority/` implements the publication's inference workflow:

`prepared corpus passages → Gemini interiority classification → passage-level predictions → book-level aggregation`

The runner accepts a prepared passage-level CSV with the documented schema and
applies the selected interiority classifier without human gold labels or
gold-label metrics. A concise Developer API run is:

```bash
cd Gemini_interiority
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export GEMINI_API_KEY="<your-api-key>"
python run_inference.py \
  --data prepared_passages.csv \
  --output-dir results/corpus_inference \
  --model gemini-3.5-flash \
  --backend developer
```

The input filename above represents a user-supplied, schema-compatible corpus
file. The framework operates on prepared passage inputs and supports compatible
Gemini model configurations; this release documents the Gemini workflow used
for the publication. See `Gemini_interiority/INFERENCE_README.md` for required
columns, Vertex AI setup, checkpointing, resume behavior, and outputs.

## Analysis

### Book-Level Data and Descriptive Analysis

`analysis/book_level_data_and_descriptive_analysis.ipynb` documents the
finalized book-level dataset workflow and the descriptive analyses used in the
manuscript. Routine publication analysis begins from
`book_summary_analysis.csv`; the notebook's upstream reconstruction section is
only needed when intentionally rebuilding that dataset from additional
upstream inputs.

The notebook produces:

- Figure 2: indicator-group counts and average interiority by historical period.
- Figure 7: book-level publication-year patterns and rolling trends.

Their supporting numerical outputs are:

- `analysis/outputs/descriptive/tables/figure_2_indicator_group_period_summary.csv`
- `analysis/outputs/descriptive/tables/broad_group_domestic_non_domestic_rolling_trends.csv`

### H1–H4 Hypothesis Analysis

`analysis/hypothesis_analysis_and_paper_figures.ipynb` loads the finalized
book-level dataset and performs the manuscript's confirmatory analyses:

- **H1:** domestic versus non-domestic fiction across historical periods.
- **H2:** protagonist gender and historical change.
- **H3:** author gender in relation to protagonist gender.
- **H4:** narration-versus-quotation interiority in third-person fiction across
  historical periods, comparing domestic and non-domestic books.

The primary manuscript mapping is Figure 1 for H1, Figure 3 for H2, Figures
4–5 for H3, and Figure 6 for H4. The notebook also produces an H2
genre-standardized robustness figure, an H3 protagonist-standardized
robustness figure, H1–H4 appendix estimate tables, supporting contrast and
diagnostic figures, and detailed numerical CSVs.

Canonical figures are written to `analysis/figures/`; supporting hypothesis
outputs are written to `analysis/outputs/hypotheses/`.

## Reproducing Manuscript Outputs

The recommended publication-analysis workflow is:

1. Begin with the included `book_summary_analysis.csv`.
2. Run `analysis/book_level_data_and_descriptive_analysis.ipynb` from the
   finalized-data loading section for the descriptive analyses.
3. Run `analysis/hypothesis_analysis_and_paper_figures.ipynb` for the H1–H4
   analyses.

Canonical publication figures are written to `analysis/figures/`. Descriptive
supporting tables are written to `analysis/outputs/descriptive/tables/`, while
hypothesis supporting figures and tables are written under
`analysis/outputs/hypotheses/`.

| Output | Analysis source |
| --- | --- |
| Figure 1 | H1, hypothesis notebook |
| Figure 2 | Descriptive notebook |
| Figure 3 | H2, hypothesis notebook |
| Figure 4 | H3, hypothesis notebook |
| Figure 5 | H3 author × protagonist analysis, hypothesis notebook |
| Figure 6 | H4, hypothesis notebook |
| Figure 7 | Descriptive notebook |
| H2 robustness figure | Hypothesis notebook |
| H3 robustness figure | Hypothesis notebook |
| H1–H4 appendix estimate tables | Hypothesis notebook |

## Setup

### Analysis environment

The publication notebooks were prepared using Python 3.12; the repository does
not formally pin a Python version. Their principal dependencies are pandas,
NumPy, Matplotlib, seaborn, Jupyter, and IPython. A minimal interactive setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy matplotlib seaborn jupyter ipython
jupyter lab
```

### Gemini environment

`Gemini_evaluation/requirements.txt` and
`Gemini_interiority/requirements.txt` define the dependencies for their
respective workflows. The preferred Developer API authentication mechanism is
the `GEMINI_API_KEY` environment variable. Both workflows also support Vertex
AI through Google Cloud credentials; consult the corresponding subdirectory
documentation for setup details.

## Citation

Citation information will be added upon publication.
