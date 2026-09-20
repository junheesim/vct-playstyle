"""
Step 1 -- audit the raw data. No decisions, no filtering, no cleaning.

The only goal is to answer one question: what is one row?

Every later step depends on that answer. If you aggregate or join before you know
the grain of the table, the mistake is invisible and it propagates.
"""
import pandas as pd, pathlib, warnings
warnings.filterwarnings("ignore")

RAW = pathlib.Path("data/raw")
YEARS = range(2021, 2027)


def load(kind: str) -> pd.DataFrame:
    frames = []
    for yr in YEARS:
        p = RAW / f"vct_{yr}/matches/{kind}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False).assign(year=yr))
    return pd.concat(frames, ignore_index=True)


def rule(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


if __name__ == "__main__":
    ov = load("overview")

    rule("A. Size")
    print(ov.groupby("year").size().rename("rows").to_frame())

    rule("B. What does one row describe?")
    # A candidate key is a set of columns that should identify a row uniquely.
    # If it does not, either the key is wrong or the table has repeats.
    for name, key in [
        ("match + map + player",        ["Tournament","Stage","Match Type","Match Name","Map","Player","year"]),
        ("match + map + player + side", ["Tournament","Stage","Match Type","Match Name","Map","Player","Side","year"]),
    ]:
        dup = ov.duplicated(key).sum()
        print(f"  {name:<30} duplicate rows: {dup:>7,}   {'UNIQUE' if dup == 0 else 'NOT unique'}")

    rule("C. Column values that are not what you would guess")
    print("  Side:", ov.Side.value_counts().to_dict())
    print("\n  Map -- 12 most common:")
    print(ov.Map.value_counts().head(12).to_string())

    rule("D. Is 'Match Name' a match id?")
    n_names = ov["Match Name"].nunique()
    n_real  = ov.groupby(["Tournament","Stage","Match Type","Match Name","year"]).ngroups
    print(f"  distinct 'Match Name' values           {n_names:>7,}")
    print(f"  distinct tournament+stage+type+name    {n_real:>7,}")

    rule("E. Pick one player-match and print every row for it")
    g = (ov[ov.Side == "both"]
           .groupby(["Tournament","Stage","Match Type","Match Name","Player","year"])
           .size())
    probe = g[g == 3].index[0]                      # a player with 3 rows in one match
    cols = ["Map","Side","Kills","Deaths","Assists","First Kills","Average Combat Score"]
    sel = ov.set_index(["Tournament","Stage","Match Type","Match Name","Player","year"]).loc[[probe]]
    print(f"  {probe[4]} -- {probe[3]} ({probe[0]})\n")
    print(sel[cols].to_string(index=False))
