# PDF table extraction experiment

Author: Nikita Riswan Kusuma Putri, student ID 81828.

This folder is the reproduction package for the paper "Improving PDF Table Extraction Accuracy Using Layout-Aware Pre-Processing and Schema Validation."

## Contents

- `run_pdf_table_experiment.py` generates the benchmark, runs the three pipelines, the ablation, and the tolerance sweep.
- `benchmark_data/` holds the 20 synthetic PDFs, five for each scenario A-D.
- `results/experiment_detailed_results.csv` holds one row per document and pipeline.
- `results/experiment_summary_results.csv` holds the means and standard deviations reported in the paper.
- `results/ablation_results.csv` holds the component ablation at a tolerance of 7 px.
- `results/sensitivity_analysis_results.csv` holds the tolerance sweep at 1, 3, 5, and 7 px.

## Reproduce

From this folder, with Python, pdfplumber, ReportLab, pandas, and NumPy installed:

```
python run_pdf_table_experiment.py
```

The script rewrites the PDFs in `benchmark_data/` and the four CSV files in `results/`. The first of the seven timed runs on each document is a warm-up and is not included in the mean. The default tolerance is 3 px. The ablation uses 7 px.
