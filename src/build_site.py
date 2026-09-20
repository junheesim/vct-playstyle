"""Build index.html for GitHub Pages from the same parts as the hosted artifact.

Two differences from the artifact build:

  1. A real document wrapper. The artifact platform supplies <!doctype html>, <html>,
     <head> and <body> itself, so the source fragment starts at <title>. A file served
     by Pages has to carry its own.

  2. Figures are referenced, not inlined. The artifact had to be one portable file;
     a repo already has figures/ committed, so <img src="figures/..."> keeps index.html
     around 220 KB instead of 1.7 MB, gives readable diffs, and lets the browser cache
     the images.

Player data stays inline (~200 KB) so the page works opened straight from disk --
fetching a sibling JSON over file:// is blocked by CORS.
"""
import pathlib, re, sys

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   "/private/tmp/claude-501/-Users-junheesim-Documents-Internship-Prep/"
                   "9bbe92f0-1581-4dcd-8e44-88d8930e71f4/scratchpad")
ROOT = pathlib.Path(__file__).resolve().parent.parent

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


def main():
    def pair(m):
        n, alt = FIG[m.group(1)], m.group(2)
        return (f'<img class="fig-light" src="figures/{n}.png" alt="{alt}" loading="lazy">'
                f'<img class="fig-dark" src="figures/dark-{n}.png" alt="{alt}" loading="lazy">')
    report = re.sub(r'<img src="(FIG_\w+)" alt="([^"]*)">', pair,
                    (SRC / "rp_report.html").read_text())
    assert "FIG_" not in report, "unreplaced figure placeholder"

    body = ((SRC / "rp_head.html").read_text()
            + (SRC / "rp_explore.html").read_text()
            + report
            + "\n<script>const DATA=" + pathlib.Path("/tmp/style_data.json").read_text()
            + ";</script>\n" + (SRC / "rp_js.html").read_text())

    out = HEAD + "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    (ROOT / "index.html").write_text(out)
    print(f"wrote index.html  {len(out)/1024:.0f} KB")
    for n in FIG.values():
        for p in (f"figures/{n}.png", f"figures/dark-{n}.png"):
            assert (ROOT / p).exists(), f"missing {p}"
    print(f"referenced {len(FIG)*2} figure files, all present")


if __name__ == "__main__":
    main()
