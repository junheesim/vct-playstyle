# VCT playstyle

Measuring playstyle in professional VALORANT, from public match data.

**→ [Read the report](REPORT.md)**

Includes a case study testing the long-running fan claim that **aspas** is a
stat-padder: the factual premise holds (15th-percentile entry rate across four
seasons) but the stated mechanism does not, because entry fragging turns out to be
K/D-neutral — it costs deaths and earns damage in roughly equal measure.

One robust style dimension and a second, marginal one. Riot's four-way role taxonomy
recovers exactly one of them — whether you are a duelist — and among the 572
non-duelist player-seasons it explains 10% of where a player sits. Discrete archetypes
do not exist: playstyle is continuous.

**[Explore the data interactively →](https://YOURUSERNAME.github.io/vct-playstyle/)**

775 player-seasons · 400 players · 2023–2026 franchised VCT · **data snapshot
2026-09-19** (the 2026 season is still running and contains no Champions event, so all
2026 figures will change if the source is refreshed).

Source: Kaggle mirror of vlr.gg (`ryanluong1/valorant-champion-tour-2021-2023-data`).

## Layout

```
index.html     the site — interactive explorer plus the full report
REPORT.md      the same findings, as markdown
LOGIC.md       how the pieces fit — the phase map
METHODS.md     what each diagnostic computes
decisions/     one record per analytical choice: the question, the diagnostic,
               the measured cost, the decision, and what would overturn it
src/           the pipeline, plus one runnable diagnostic per decision
figures/       every figure in light and dark variants
```

## The site

`index.html` is served by GitHub Pages from the repository root. Settings → Pages →
*Deploy from a branch*, branch `main`, folder `/ (root)`. No build step or workflow.

It is regenerated with `python src/build_site.py`, which wraps the page in a real HTML
document, points the figures at `figures/` rather than inlining them, and keeps the
player data inline so the file also works opened straight from disk.

`decisions/` is the point of this repo. Every number in the report traces to a file
there that says why the choice was made and what would change it — including the
corrections, which are kept visible rather than edited away.

## Running it

```bash
git clone <this repo> && cd vct-playstyle

python src/data.py             # scoping, grain, identity
python src/features.py         # the eight behavioural measures
python src/style.py            # quality adjustment -> style matrix
python src/components.py       # parallel analysis -> component retention
python src/clusters.py         # archetypes vs continuum
python src/validate.py         # held-out 2026
python src/sensitivity.py      # do the judgement calls change the answer?
python src/report_numbers.py   # every figure the report quotes
python src/figures.py          # figures
```

Requires pandas, numpy, statsmodels, scikit-learn, matplotlib.
Raw data is not committed; `data/raw/` expects the Kaggle dataset unpacked by year.
