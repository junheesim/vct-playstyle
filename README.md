# VCT Playstyle

[![tests](https://github.com/junheesim/vct-playstyle/actions/workflows/tests.yml/badge.svg)](https://github.com/junheesim/vct-playstyle/actions/workflows/tests.yml)

**Measuring playstyle in professional VALORANT, and separating it from how good a
player is.** Six behaviors, drawn from 117,000 rows of public match data, reduce to two
style dimensions that persist across seasons and are not a restatement of Riot's agent
roles.

**[Read the report →](https://junheesim.github.io/vct-playstyle/#report)**  ·
**[Explore 775 player-seasons →](https://junheesim.github.io/vct-playstyle/)**

775 player-seasons · 400 players · 1,532 matches · 2023–2026 franchised VCT

![What each axis is made of](figures/3-axes.png)

---

## What it found

**Two style dimensions, one of them solid.** Six behaviors reduce to two components,
kept only if they beat a Monte Carlo null rather than a variance threshold. *Aggression*
clears it comfortably at 45.4% of the variation; *isolation* clears it by 0.071, fails
among role specialists, and is reported as marginal throughout.

**Riot's four named roles are three distinguishable levels.** A duelist / not-duelist
binary gets R² .523 against the full four-way label's .586 — almost everything the
taxonomy knows is one bit. Duelists barely overlap anyone else, and controllers and
initiators are the same role played twice (Cohen's d = 0.32).

**Playstyle persists, more strongly than performance does.** From one season to the
next a player's aggression correlates .731 — higher than their own kill-death ratio
(.634) and far higher than whether their team wins (.399). That is what a disposition
looks like rather than a result.

**Discrete archetypes do not exist.** Separation among the real players is 0.369,
against 0.320 ± 0.010 for structureless clouds and 0.527 for simulated data that really
does contain three types. The data sits three times closer to no grouping than to real
types. Playstyle is continuous.

**Case study — is aspas a stat-padder?** He does take the round's first fight far less
than other duelists: 8th percentile on aggression among them. But twenty-two
duelist-seasons enter even less than he does, and their kill-death ratio averages 1.05
against 1.06 for duelists as a whole. His is 1.28. Entering less simply does not come
with a better ratio, so it cannot be where his numbers come from.

---

## How the repository works

```
index.html            the deployed site: interactive explorer + the full report.
                      GENERATED — edit site/, never this.

site/                 what index.html is built from
  head.html             styles and the page shell
  explore.html          the interactive scatter of all 775 player-seasons
  report.html           the report itself: every finding, figure and table
  app.js.html           explorer behavior — filtering, hover, the detail panel
  data.json             one record per player-season, written by the pipeline

src/                  the pipeline: raw CSVs in, the style space out
  paths.py              where things live; nothing else hard-codes a path
  data.py               loading and scoping — grain, era, player identity
  features.py           the six behavioral measures, per player-season
  roles.py              agent → Riot role, and the season's role label
  style.py              adjust each behavior for quality → the style matrix
  pca.py                standardize, rotate, fix the sign — fitted in ONE place
  components.py         how many dimensions beat a Monte Carlo null
  clusters.py           archetypes or a continuum
  validate.py           refit on 2023–25, project 2026 cold
  robustness.py         do the judgment calls change the answer?
  examples.py           named players who are extreme for their role
  figures.py            every figure, in light and dark
  quoted_numbers.py     EVERY number the site quotes, computed from the data
  publish.py            site/ + figures/ → index.html

  diagnostics/          one-off checks; nothing imports them, they print
    scope.py              grain, era, player identity
    screening.py          which behaviors measure style rather than quality
    role_labels.py        how good the role label is, and what it costs

tests/                the claims the report makes, pinned
figures/              every figure, light and dark variant
.github/              CI: the checks that need no raw data, run on every push
LOGIC.md              how the phases fit together
METHODS.md            what each diagnostic computes
```

### The pipeline

```
paths → data → features → roles → style → pca → components
                                                 ├→ clusters     archetypes or continuum
                                                 ├→ validate     does 2026 hold up
                                                 └→ robustness   floor, quality measure, per-round
```

Every script runs from any working directory and prints its own result. Raw data is not
committed; `data/raw/` expects the Kaggle dump unpacked by year, and `data.py` says so
plainly if it is missing.

### The one rule this repo is built around

**No number in the prose is hand-copied.** `quoted_numbers.py` computes every figure the
site states, and `tests/test_site.py` fails the build when the site quotes a value the
pipeline does not produce — naming each one that drifted. The report and the analysis
cannot disagree without the test suite saying so.

The same idea runs through the code. The PCA is fitted in exactly one function, because
the sign convention was once applied to the loadings and not the scores. The agent→role
lookup raises on an unknown agent instead of silently dropping it. Corrections are kept
visible in comments rather than edited away: each says what was wrong, how it surfaced,
and what now prevents it.

### The site

`index.html` is served by GitHub Pages from the repository root — no build step. It is
generated from `site/`, which `publish.py` wraps in a document, expands `FIG_*`
placeholders into light/dark image pairs, and inlines `data.json` so the page works
opened straight from disk as well as served.

---

## Data

[**VALORANT Champion Tour data**](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data)
on Kaggle, a mirror of public [vlr.gg](https://www.vlr.gg) match records. The dataset's
name says 2021–2023; it is maintained past that and carries 2023–2026, which is the
window used here — the franchised era, chosen so that every team plays every other and
strength of schedule stops being a confounder.

**Snapshot 2026-09-19.** The 2026 season was still running and contains no Champions
event, so every 2026 figure will move if the source is refreshed.
