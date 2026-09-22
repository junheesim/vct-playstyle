"""The site is built from site/, and its data file is regenerable from the pipeline."""
import json

import paths


def test_every_site_part_is_present():
    for name in ("head.html", "explore.html", "report.html", "app.js.html", "data.json"):
        assert (paths.SITE / name).exists(), f"site/{name} is missing"


def test_report_placeholders_match_the_builder():
    import re
    import publish
    found = set(re.findall(r'<img src="(FIG_\w+)"', (paths.SITE / "report.html").read_text()))
    assert found == set(publish.FIG)


def _referenced() -> set:
    """Figure stems named by anything a reader actually opens.

    This used to union in REPORT.md's references. That file was the report kept as a
    second markdown copy; it was removed once the two drifted, so the site is the only
    reader left."""
    import publish
    return set(publish.FIG.values())


def test_every_referenced_figure_exists():
    for n in _referenced():
        for f in (f"{n}.png", f"dark-{n}.png"):
            assert (paths.FIGURES / f).exists(), f"figures/{f} is referenced but missing"


def test_no_figure_is_generated_and_never_used():
    """The other direction. A figure rebuilt by every run and opened by nobody is
    either dead or a reference someone forgot to add -- say which."""
    built = {p.stem.removeprefix("dark-") for p in paths.FIGURES.glob("*.png")}
    assert built <= _referenced(), f"generated but referenced nowhere: {sorted(built - _referenced())}"


def test_agent_mix_accounts_for_every_map():
    """The mix was truncated to six agents, while the page divides by its total to
    size the bars -- so 246 of 775 players had every bar drawn too wide."""
    for r in json.loads((paths.SITE / "data.json").read_text()):
        assert sum(n for _, n in r["mx"]) == r["m"], f"{r['h']} {r['y']}"


def test_data_json_matches_the_page_contract():
    recs = json.loads((paths.SITE / "data.json").read_text())
    assert len(recs) == 775
    assert set(recs[0]) == {"h","y","t","g","r","x","z","fe","d","as","hs","pl",
                            "cl","kd","adr","m","rc","ok","mx"}


def test_the_role_count_is_consistent_with_the_share_and_the_guard():
    """The card states the derivation as `<role> on rc of m (rs%)`, so those three
    have to agree with each other and with the >=70% guard."""
    for r in json.loads((paths.SITE / "data.json").read_text()):
        assert 0 < r["rc"] <= r["m"]
        # The card floors this share; the guard is >=70%. They must agree exactly,
        # including at the boundary -- iamgrq 2026 is 69.57%, which ROUNDS to 70%
        # but must not read as passing.
        assert r["ok"] == (r["rc"] / r["m"] * 100 // 1 >= 70), f'{r["h"]} {r["y"]}'


def test_the_site_quotes_the_numbers_the_pipeline_computes():
    """Every pipeline-derived number the prose states, checked against the pipeline.

    This is the guard the project was missing. `README` explains that REPORT.md was
    deleted because a second copy of a document full of measured numbers has no way
    to stay correct -- and then the site itself shipped three hand-copied numbers
    that had drifted from their source:

      * agent retention quoted as 34% stayers / 30% movers, when the diagnostic
        said 35% / 35% -- the two halves reversed, on a card whose whole point is
        that the gap between them is zero;
      * a clutch median of 13, when the analysis sample's median is 12;
      * the explorer's landing line still quoting the R2 that decision 17 superseded,
        so the first sentence a visitor read contradicted Finding 2.

    Presence, not position: the claim has to appear SOMEWHERE in site/. If a number
    moves, the string stops appearing and this names the key that broke. The weakness
    of matching on presence is a short string that occurs elsewhere by chance, so the
    formats here are the distinctive ones the page uses -- "15th", not "15%".
    """
    from quoted_numbers import site_claims
    text = "".join((paths.SITE / n).read_text()
                   for n in ("report.html", "explore.html"))
    for ent, ch in (("&minus;", "\u2212"), ("&nbsp;", " "), ("&ndash;", "\u2013")):
        text = text.replace(ent, ch)
    missing = {k: v for k, v in site_claims().items() if v not in text}
    assert not missing, "site/ does not quote the computed value for:\n" + "\n".join(
        f"  {k:22s} computed {v!r}" for k, v in sorted(missing.items()))


def test_index_html_is_in_sync_with_the_parts_it_is_built_from():
    """`index.html` is GENERATED. Editing it directly, or editing site/ and not
    rebuilding, puts the deployed page and its source out of step -- and the
    deployed one is what a reader sees."""
    import publish
    built = (paths.ROOT / "index.html").read_text()
    for name in ("head.html", "explore.html", "app.js.html"):
        assert (paths.SITE / name).read_text() in built, f"index.html is stale vs site/{name}"
    assert publish.expand_figures((paths.SITE / "report.html").read_text()) in built, \
        "index.html is stale vs site/report.html -- run: python src/publish.py page"
