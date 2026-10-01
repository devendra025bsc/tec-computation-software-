# Compiling the Manuscript

## Overleaf (recommended)

1. Go to [overleaf.com](https://www.overleaf.com) and sign in.
2. Click **New Project → Upload Project**.
3. Upload `Bhetuwal_dBdt_ROTI_LaTeX_Package.zip` (the entire `latex/` folder zipped).
4. Set the compiler to **pdfLaTeX** (Project → Compiler).
5. Click **Compile** — Overleaf handles the BibTeX pass automatically.

The full manuscript should compile to 30 pages with all 18 figures embedded.

---

## Local compilation

Requires a standard TeX Live or MiKTeX installation with `pdflatex` and `bibtex`.

```bash
cd latex/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Three `pdflatex` passes are needed to resolve all cross-references,
figure/table numbers, and citations correctly.

---

## File structure expected by main.tex

```
latex/
├── main.tex
├── references.bib
└── figures/
    ├── Fig1_station_map.png
    ├── Fig2_SYMH_overview.png
    ├── Fig5_dbdt_roti_timeseries.png
    ├── Fig6_lag_heatmap.png
    ├── Fig7_correlation_heatmap.png
    ├── Fig8_lag_vs_maglat.png
    ├── Fig9_r_vs_maglat.png
    ├── Fig11_lag_vs_mlt_polar.png
    ├── Fig12_lag_vs_symh.png
    ├── Fig13_phase_comparison.png
    ├── Fig13b_r_vs_dbdt.png
    ├── Fig14_cross_event_summary.png
    ├── Fig15_propagation_separation.png
    ├── Fig16_quiet_vs_storm_control.png
    ├── Fig_ccf_loy8_may2024.png
    ├── Fig_solarwind_may2024.png
    ├── Fig_stec_vtec_nist.png
    └── Fig_vtec_multistation.png
```

All figure paths in `main.tex` are relative to the `latex/` directory,
so the structure above must be preserved exactly.

---

## Packages required

All packages are included in standard TeX Live (2021+):

- `geometry`, `times`, `amsmath`, `amssymb`
- `graphicx`, `booktabs`, `array`, `longtable`
- `hyperref`, `natbib`, `caption`, `authblk`
- `fancyhdr`, `titlesec`, `xcolor`, `microtype`
