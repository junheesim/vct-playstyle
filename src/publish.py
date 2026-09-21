"""Build the site: the player data, then index.html.

    python src/publish.py data    site/data.json, from the pipeline
    python src/publish.py page    site/* + figures/ -> index.html
    python src/publish.py         both

(Named publish, not site: `site` is a stdlib module that Python imports at startup,
so a `site.py` next to it is the same trap as `numbers.py`.)

`index.html` is GENERATED. Edit `site/`, not the built file. Two things happen in
`page` that the parts cannot do for themselves:

  1. A real document wrapper. The parts are a fragment, because the hosted-artifact
     build supplies <!doctype html>, <html>, <head> and <body> itself. A file served
     by Pages has to carry its own.

  2. Figure placeholders expand to a light/dark <img> pair pointing at figures/.
     Referencing rather than inlining keeps index.html near 220 KB instead of 1.7 MB,
     gives readable diffs, and lets the browser cache the images.

The player data stays INLINE (~210 KB) so the page also works opened straight from
disk -- fetching a sibling JSON over file:// is blocked by CORS.
"""
import argparse
import json
import re
import sys

import pandas as pd

import features as F
import paths
from examples import guarded


DATA = paths.SITE / "data.json"

# page key -> pipeline column
COLS = {"fe": "first_engagement", "d": "deaths", "as": "assists", "hs": "hs",
        "pl": "plants", "cl": "clutch_att", "kd": "kd_ratio", "adr": "adr"}


def agent_mix(d: pd.DataFrame) -> pd.Series:
    """[[agent, maps], ...] per player-season, most-played first."""
    b = d[d.Side == "both"]
    c = b.groupby(["player_id", "year"]).Agents.value_counts()
    return (c.rename("n").reset_index()
             .groupby(["player_id", "year"])
             .apply(lambda g: [[a, int(n)] for a, n in
                               zip(g.Agents.str.lower(), g.n)], include_groups=False)
             .rename("mx"))


def build_data() -> list[dict]:
    d = F.build()
    G = (guarded()
         .merge(F.seasons(d), on=["player_id", "year"], how="left", suffixes=("", "_r"))
         .merge(agent_mix(d), on=["player_id", "year"], how="left"))
    out = []
    for _, r in G.iterrows():
        rec = {"h": r.handle, "y": int(r.year), "t": r.team, "g": r.region, "r": r.role,
               "x": round(float(r.PC1), 2), "z": round(float(r.PC2), 2)}
        rec |= {k: round(float(r[c]), 2) for k, c in COLS.items()}
        # `rc` is the map COUNT in the labelled role. The page has no agent-to-role
        # lookup, so it cannot count these itself, and recovering it from the rounded
        # `rs` would be luck rather than arithmetic.
        rec |= {"m": int(r.maps),
                "rc": int(round(float(r.role_share) * int(r.maps))),
                "ok": bool(r.label_ok), "mx": r.mx}
        out.append(rec)
    return out


SITE = paths.SITE
OUT  = paths.ROOT / "index.html"

FIG = {"FIG_3axes": "3-axes", "FIG_1broles": "1b-roles",
       "FIG_2heldout": "2-held-out", "FIG_4kd": "4-kd-hides"}

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="description" content="Measuring playstyle in professional VALORANT: 775 VCT
player-seasons on two measured style dimensions, with the full research report.">
<style>img{max-width:100%}[hidden]{display:none!important}</style>
"""


def part(name: str) -> str:
    p = SITE / name
    if not p.exists():
        sys.exit(f"missing {p.relative_to(paths.ROOT)}"
                 + ("  -- run: python src/publish.py data" if name == "data.json" else ""))
    return p.read_text()


def expand_figures(report: str) -> str:
    def pair(m):
        n, alt = FIG[m.group(1)], m.group(2)
        return (f'<img class="fig-light" src="figures/{n}.png" alt="{alt}" loading="lazy">'
                f'<img class="fig-dark" src="figures/dark-{n}.png" alt="{alt}" loading="lazy">')
    out, n = re.subn(r'<img src="(FIG_\w+)" alt="([^"]*)">', pair, report)
    assert n == len(FIG), f"expanded {n} figure placeholders, expected {len(FIG)}"
    assert "FIG_" not in out, "unreplaced figure placeholder"
    return out


def build_page():
    body = (part("head.html") + part("explore.html") + expand_figures(part("report.html"))
            + "\n<script>const DATA=" + part("data.json").strip() + ";</script>\n"
            + part("app.js.html"))
    OUT.write_text(HEAD + "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    print(f"wrote index.html  {OUT.stat().st_size/1024:.0f} KB")

    missing = [p for n in FIG.values() for p in (f"figures/{n}.png", f"figures/dark-{n}.png")
               if not (paths.ROOT / p).exists()]
    if missing:
        sys.exit("missing figures: " + ", ".join(missing) + "\n  -- run: python src/figures.py")
    print(f"referenced {len(FIG)*2} figure files, all present")


def export_data():
    recs = build_data()
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(recs, separators=(",", ":")))
    print(f"wrote {DATA.relative_to(paths.ROOT)}  "
          f"{len(recs)} player-seasons, {DATA.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", nargs="*", default=[], choices=["data", "page", []],
                    help="default: both")
    a = ap.parse_args()
    for name in (a.step or ["data", "page"]):
        {"data": export_data, "page": build_page}[name]()
