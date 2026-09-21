"""Shared plumbing for the diagnostics.

Importing this puts `src/` on the path, so a diagnostic can `import paths` and
`import features` while living one directory down. Import it FIRST for that reason.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def rule(title, width=74):
    print(f"\n{'='*width}\n{title}\n{'='*width}")


def main(checks: dict, description: str):
    """`checks` maps a subcommand name to (one-line help, function)."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("check", nargs="*", default=[],
                    help="which check(s) to run; default is all of them")
    ap.add_argument("--list", action="store_true", help="list the checks and exit")
    a = ap.parse_args()
    if a.list:
        for name, (help_, _) in checks.items():
            print(f"  {name:<16}{help_}")
        return
    unknown = [c for c in a.check if c not in checks]
    if unknown:
        sys.exit(f"unknown check(s): {', '.join(unknown)}\n"
                 f"available: {', '.join(checks)}")
    for name in (a.check or checks):
        checks[name][1]()
