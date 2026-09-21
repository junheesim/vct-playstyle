"""The repository's own shape."""
import sys

import paths


def test_no_module_shadows_the_standard_library():
    """Twice now: a `numbers.py` broke numpy's import, and a `site.py` shadowed the
    module Python loads at startup. src/ goes on sys.path, so every name here is a
    name the stdlib cannot have."""
    stdlib = sys.stdlib_module_names
    ours = {p.stem for p in paths.SRC.glob("*.py")} | {p.stem for p in (paths.SRC / "diagnostics").glob("*.py")}
    clash = sorted(n for n in ours if n in stdlib)
    assert not clash, f"shadows the standard library: {clash}"


def test_the_diagnostics_are_not_imported_by_the_pipeline():
    """They print and are read by hand. If the pipeline started depending on one, it
    would be library code sitting in the wrong place -- which is how the role taxonomy
    ended up inside a file called step11_role_independence."""
    import ast
    names = {p.stem for p in (paths.SRC / "diagnostics").glob("*.py")} | {"diagnostics"}
    for p in paths.SRC.glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Import):
                got = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                got = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (got & names), f"{p.name} imports {got & names}"


def test_every_decision_cites_something_that_exists():
    """`decisions/05` cited `src/step06_reliability.py`, which never existed."""
    import re
    bad = []
    for md in sorted((paths.ROOT / "decisions").glob("*.md")):
        for ref in re.findall(r"`src/([\w/]+\.py)([^`]*)`", md.read_text()):
            path, sub = paths.SRC / ref[0], ref[1].strip()
            if not path.exists():
                bad.append(f"{md.name} -> src/{ref[0]} (missing)")
            elif sub and f'"{sub}"' not in path.read_text():
                bad.append(f"{md.name} -> src/{ref[0]} has no check {sub!r}")
    assert not bad, "\n".join(bad)
