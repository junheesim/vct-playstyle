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
