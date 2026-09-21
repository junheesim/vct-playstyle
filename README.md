# VCT playstyle

Measuring playstyle in professional VALORANT, from public match data.

**→ [Read the report](https://junheesim.github.io/vct-playstyle/#report)**

Includes a case study testing the long-running fan claim that **aspas** is a
stat-padder: the factual premise holds (15th-percentile entry rate across four
seasons) but the stated mechanism does not, because entry fragging turns out to be
K/D-neutral — it costs deaths and earns damage in roughly equal measure.

One robust style dimension and a second, marginal one. Riot's four-way role taxonomy
recovers exactly one of them — whether you are a duelist — and among the 570
non-duelist player-seasons it explains 16% of where a player sits. Discrete archetypes
do not exist: playstyle is continuous.

**[Explore the data interactively →](https://junheesim.github.io/vct-playstyle/)**

775 player-seasons · 400 players · 2023–2026 franchised VCT · **data snapshot
2026-09-19** (the 2026 season is still running and contains no Champions event, so all
2026 figures will change if the source is refreshed).

Source: Kaggle mirror of vlr.gg (`ryanluong1/valorant-champion-tour-2021-2023-data`).

## Layout

```
index.html     the built site — interactive explorer plus the full report
site/          what it is built FROM: the page parts, and data.json
LOGIC.md       how the pieces fit — the phase map
METHODS.md     what each diagnostic computes
decisions/     one record per analytical choice: the question, the diagnostic,
               the measured cost, the decision, and what would overturn it
src/           the pipeline
src/diagnostics/  one named check per decision
tests/         the claims the report makes, pinned
figures/       every figure in light and dark variants
```

Inside `src/`, the pipeline and the diagnostics are separate things.

```
paths -> data -> features -> roles -> style -> pca -> components
                                                   -> clusters | validate | robustness
```

`src/diagnostics/` holds the one-off checks behind individual decisions. Nothing
imports them; they print, and they are read next to the decision file that cites
them. Each is a named subcommand:

```bash
python src/diagnostics/scope.py --list        # grain, era, identity      (01-03)
python src/diagnostics/screening.py --list    # feature selection         (04-07)
python src/diagnostics/role_labels.py --list  # how good is the role tag? (07, 11, 14)
```

## The site

`index.html` is served by GitHub Pages from the repository root. Settings → Pages →
*Deploy from a branch*, branch `main`, folder `/ (root)`. No build step or workflow.

`index.html` is **generated** — edit `site/`, not the built file:

```bash
python src/publish.py data    # site/data.json, from the pipeline
python src/publish.py page     # site/* + figures/ -> index.html
```

`src/publish.py page` wraps the parts in a real HTML document, expands each `FIG_*`
placeholder into a light/dark `<img>` pair pointing at `figures/`, and inlines
`data.json` so the page also works opened straight from disk — fetching a sibling
JSON over `file://` is blocked by CORS.

The report lives on the site and nowhere else. It was kept in parallel as a markdown
file until 2026-09-20; the two copies drifted twice — once on which figures they showed,
once on nine numbers and a claim about named players — so the duplicate was removed
rather than maintained.

`decisions/` is the point of this repo. Every number in the report traces to a file
there that says why the choice was made and what would change it — including the
corrections, which are kept visible rather than edited away.

## Running it

```bash
git clone <this repo> && cd vct-playstyle
python -m venv .venv && .venv/bin/pip install -r requirements.txt

python src/data.py             # scoping, grain, identity
python src/features.py         # the seven behavioural measures
python src/style.py            # quality adjustment -> style matrix
python src/components.py       # parallel analysis -> component retention
python src/clusters.py         # archetypes vs continuum
python src/validate.py         # held-out 2026
python src/robustness.py       # do the judgement calls change the answer?
python src/quoted_numbers.py   # every number the prose quotes
python src/figures.py          # figures  (FIG_THEME=dark for the dark set)
python src/publish.py          # site/data.json, then index.html
pytest                         # the claims above, pinned
```

Every script runs from any working directory and each one prints its own result;
`src/quoted_numbers.py report` is the single source of truth for every number in the prose.

Raw data is not committed. `data/raw/` expects the Kaggle dataset unpacked by year,
so that `data/raw/vct_2023/matches/overview.csv` exists. In this working copy it is a
symlink to a local copy of the dump; `python src/data.py` says so plainly if the link
is missing or broken.
