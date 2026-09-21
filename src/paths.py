"""Where things live.

Every script in src/ used to start with `sys.path.insert(0, "src")` and then read and
write relative paths like `data/raw` and `figures/`, so nothing ran unless the working
directory happened to be the repository root. These constants are anchored to this
file instead, and scripts run from anywhere.

The sys.path line below is NOT what made the imports work -- Python already puts a
script's own directory first when you run `python src/anything.py`. It is here so the
same modules import cleanly from a notebook or REPL started at the repository root.
"""

import pathlib
import sys

SRC     = pathlib.Path(__file__).resolve().parent
ROOT    = SRC.parent
RAW     = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
FIGURES = ROOT / "figures"
SITE    = ROOT / "site"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
