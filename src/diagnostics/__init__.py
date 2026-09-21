"""One-off diagnostics, one per analytical decision.

These were fifteen files named after the order they were run in -- `step01b_probe`,
`step07_q14`, `step08_yardstick`. That is a transcript, not a structure: the names
said when a check happened rather than what it asks, and a citation in `decisions/05`
pointed at a file that did not exist.

They are three modules now, grouped by what they interrogate, each a set of named
subcommands:

    python src/diagnostics/scope.py         grain | duplicates | era | cost | identity
    python src/diagnostics/screening.py     --list
    python src/diagnostics/role_labels.py   --list

Nothing imports these. They print, and they are read next to the decision file that
cites them.
"""
