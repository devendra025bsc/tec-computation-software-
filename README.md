# Magnetosphere–Ionosphere Coupling During Geomagnetic Storms
### A Multi-Event, Multi-Station Analysis of dB/dt–ROTI Response Delays and Longitudinal Variability

**Author:** Devendra Bhetuwal  
**Institution:** St. Xavier's College, Maitighar, Kathmandu, Nepal

---

## Overview

This repository contains the full analysis pipeline, data-collection scripts,
and figure-generation code for the manuscript
*"Magnetosphere–Ionosphere Coupling During Geomagnetic Storms: A Multi-Event,
Multi-Station Analysis of dB/dt–ROTI Response Delays and Longitudinal
Variability"*.

The study quantifies the lagged cross-correlation between the geomagnetic
field time derivative (|dB/dt|) and the Rate of TEC Index (ROTI) across five
geomagnetic storm events (2018–2024) and five co-located geomagnetic–GNSS
station pairs spanning equatorial to sub-auroral latitudes.

---

## Repository Structure

```
.
├── scripts/                      # Data collection (run before analysis)
│   ├── download_geomagnetic_data.py     # 1-min geomag data from USGS
│   ├── compute_roti_from_rinex.py       # ROTI from NOAA CORS RINEX files
│   ├── fetch_symh_index.py              # SYM-H from NASA OMNI via pyspedas
│   ├── fetch_omni_solar_wind.py         # Hourly OMNI solar-wind parameters
│   ├── fetch_stec_vtec.py               # STEC/VTEC multi-station plots
│   └── cross_correlation_preliminary.py # Exploratory CCF (independent check)
│
├── src/                          # Core analysis pipeline
│   ├── cross_correlation_pipeline.py    # Main analysis (CCF, surrogates,
│   │                                    #   bootstrap, FDR, master table)
│   └── generate_figures.py              # All manuscript figures (Figs 1–18)
│
├── data/                         # Input data directory (not tracked in git)
│   ├── geomag/
│   ├── roti/
│   ├── symh/
│   └── omni/
│
├── results/                      # Output tables (not tracked in git)
│   └── master_table.csv
│
├── figures/                      # Output figures (not tracked in git)
│
├── latex/                        # Overleaf-ready manuscript source
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── config_example.json           # Example config for the analysis pipeline
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect data

Run each data-collection script for each storm event and station pair.
Example for the May 2024 event:

```bash
# Geomagnetic data for all five observatories
for STN in BOU TUC GUA HON FRD; do
    python scripts/download_geomagnetic_data.py \
        --station $STN --start 2024-05-08 --end 2024-05-15 \
        --outdir data/geomag
done

# ROTI from CORS RINEX for all five GNSS stations
for STN in nist cot1 guam hnlc loy8; do
    python scripts/compute_roti_from_rinex.py \
        --station $STN --start 2024-05-08 --end 2024-05-15 \
        --outdir data/roti
done

# SYM-H index
python scripts/fetch_symh_index.py \
    --start 2024-05-08 --end 2024-05-15 --outdir data/symh

# Hourly OMNI solar-wind parameters
python scripts/fetch_omni_solar_wind.py \
    --start 20240508 --end 20240515 --outdir data/omni

# STEC/VTEC multi-station comparison plots
python scripts/fetch_stec_vtec.py \
    --stations NIST GUAM LOY8 HNLC COT1 \
    --start 2024-05-08 --end 2024-05-14 --outdir data/stec_vtec
```

Repeat for the other four events:
- 2018-08-23 to 2018-08-29
- 2021-11-02 to 2021-11-08
- 2023-11-03 to 2023-11-09
- 2024-03-22 to 2024-03-28

### 3. Run the analysis

Edit `config_example.json` to point to your actual data files, then:

```bash
python src/cross_correlation_pipeline.py \
    --config config_example.json \
    --outdir results
```

This produces `results/master_table.csv` — the 175-row (or 118-row after
QC correction) table that underlies all manuscript statistics.

### 4. Generate all figures

```bash
python src/generate_figures.py \
    --master-csv results/master_table.csv \
    --omni-csv data/omni/omni_data_20240508_20240514.csv \
    --outdir figures
```

### 5. Compile the manuscript

The `latex/` directory contains the complete Overleaf-ready source.
Open in Overleaf (File → Import → Upload) or compile locally:

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Methods Summary

| Step | Tool | Description |
|------|------|-------------|
| Data collection | `scripts/` | USGS, NOAA CORS, NASA OMNI |
| |dB/dt| | `download_geomagnetic_data.py` | Central-difference derivative of H-component |
| ROTI | `compute_roti_from_rinex.py` | Carrier-phase L1/L2 → STEC → 5-min ROTI |
| SYM-H phase classification | `cross_correlation_pipeline.py` | Daily-min SYM-H, pre-registered before CCF |
| Cross-correlation | `cross_correlation_pipeline.py` | τ ∈ [−180, +180] min, 5-min step |
| Significance | `cross_correlation_pipeline.py` | Circular-shift surrogates (n=500, ≥30-min shift) |
| Multiple comparisons | `cross_correlation_pipeline.py` | Benjamini–Hochberg FDR (α=0.05), n=142 tests |
| Uncertainty on r | `cross_correlation_pipeline.py` | Moving-block bootstrap (30-min blocks, n=500) |
| Uncertainty on τ | `cross_correlation_pipeline.py` | Full-lag-scan bootstrap (30-min blocks, n=200) |
| Figures | `src/generate_figures.py` | Matplotlib, Okabe-Ito palette, 300 dpi |

---

## Data Quality Note

During analysis, the NIST GNSS station's ROTI files were found to be
byte-identical across four of five storm events (100% value match,
n=2016 samples per comparison). This is documented in the manuscript
(Section 2.3 / data-quality audit) and resolved by excluding BOU+NIST from
the four affected events, retaining it only for May 2024. Other stations
(COT1, GUAM, HNLC, LOY8) show no such duplication. The pipeline in
`src/cross_correlation_pipeline.py` applies this correction automatically
when `"qc_exclude": true` is set in the config for the affected cases.

---

## Reproducing Key Results

| Result | Value | Location in code |
|--------|-------|-----------------|
| Strongest case | r=0.800, τ=+5 min, FRD–LOY8, 2024-05-10 | `results/master_table.csv` |
| Nominal significant | 12/142 (8.5%) | `significant_nominal` column |
| FDR significant | 0/142 (min q=0.094) | `significant_fdr` column |
| Storm vs quiet |r| | p=0.009, rank-biserial=−0.34 | `src/cross_correlation_pipeline.py` |
| AE predictor (mixed model) | p=0.039 | Reported in manuscript Section 3.9 |

---

## Citation

If you use this code or data, please cite:

```
Bhetuwal, D. (2024). Magnetosphere–Ionosphere Coupling During Geomagnetic Storms:
A Multi-Event, Multi-Station Analysis of dB/dt–ROTI Response Delays and
Longitudinal Variability. St. Xavier's College, Maitighar, Kathmandu, Nepal.
```

---

## License

MIT License — see LICENSE file.
