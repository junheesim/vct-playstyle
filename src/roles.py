"""Riot's agent -> role taxonomy, the season role label, and how much of a
season a player really spent in it.

The lookup used to live inside a file named after an exploration step, which six
modules imported. The diagnostics that ask whether roles are redundant, and what the
weak agent label costs, are in `diagnostics/role_labels.py`; the lookup is here.

Two things are deliberate:

  ONE mapping function. `role_of` was previously written twice, once splitting the
  agent cell on a comma and once not, so a multi-agent cell resolved to a role in one
  code path and to NaN in the other -- silently failing the label guard in the second.

  A LOUD failure on an unknown agent. Riot adds agents; an unmapped one used to map
  to NaN and disappear from every role-based result without complaint. `role_of`
  raises unless the caller passes `strict=False`.
"""
import pandas as pd

import data as D
import features as F

ROLE = {**{a: "duelist"    for a in ["jett","raze","reyna","phoenix","yoru","neon","iso","waylay"]},
        **{a: "initiator"  for a in ["sova","breach","skye","kayo","fade","gekko","tejo"]},
        **{a: "controller" for a in ["omen","brimstone","viper","astra","harbor","clove","miks"]},
        **{a: "sentinel"   for a in ["killjoy","cypher","sage","chamber","deadlock","vyse","veto"]}}

# Agent names that appear in the data but are deliberately given no role. Empty: every
# agent in the source is now placed. The escape hatch stays, so that a future name can
# be excluded ON PURPOSE rather than by being forgotten -- `role_of` raises otherwise.
NOT_AN_AGENT: set[str] = set()

ROLES = ["duelist", "initiator", "controller", "sentinel"]


def role_of(agents: pd.Series, strict: bool = True) -> pd.Series:
    """Agent cell -> Riot role. Takes the first agent if the cell lists several."""
    a = agents.astype(str).str.lower().str.split(",").str[0].str.strip()
    out = a.map(ROLE)
    unknown = sorted(set(a[out.isna() & a.ne("nan")]) - NOT_AN_AGENT)
    if unknown and strict:
        raise KeyError(f"agents missing from ROLE: {unknown}. Add them, or add them to "
                       "NOT_AN_AGENT if they should stay unmapped.")
    return out


def with_role(ps: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Attach each player-season's role, its share of the season, and its modal agent.

    `role` is the MODAL ROLE: every map is mapped to a role and the most common one
    wins. It used to be the role of the single most-played AGENT, which is a much
    weaker thing -- the modal agent covers a median of 43% of a player's maps while
    the modal role covers 81%. A player who splits 40% Jett / 35% Sova / 25% Fade is
    an initiator for 60% of their maps, and the old label called them a duelist.

    `main_agent` is kept, but as a description only. Nothing models it.

    A tie is broken by the order of ROLES, which is arbitrary; `role_share` is carried
    alongside so a reader can see when the label is thin.
    """
    b = d[d.Side == "both"]
    counts = (b.assign(_r=role_of(b.Agents))
                .groupby(["player_id", "year"])._r.value_counts()
                .unstack().reindex(columns=ROLES).fillna(0.0))
    lab = pd.DataFrame({"role": counts.idxmax(axis=1),
                        "role_share": counts.max(axis=1) / counts.sum(axis=1)})
    ag = (b.groupby(["player_id", "year"]).Agents
            .agg(lambda s: s.mode().iat[0]).rename("main_agent"))
    return ps.merge(ag, on=["player_id", "year"]).merge(lab, on=["player_id", "year"])


# ---- how much of a season was actually spent in the labeled role ----


def shares(d: pd.DataFrame = None, ps: pd.DataFrame = None) -> pd.DataFrame:
    """Per player-season: fraction of maps in each Riot role, plus the modal agent."""
    d = F.build() if d is None else d
    ps = F.seasons(d) if ps is None else ps
    b = d[d.Side == "both"].copy()
    b["r"] = role_of(b.Agents)
    keep = pd.MultiIndex.from_frame(ps[["player_id", "year"]])
    b = b[pd.MultiIndex.from_frame(b[["player_id", "year"]]).isin(keep)]
    # Denominator is ALL of the player's maps, not just the ones whose agent maps to
    # a role. A map on an agent outside the taxonomy (a new release, or a bad row) is
    # not a map in the labeled role, so dropping it from the denominator OVERSTATES
    # how much of the season was spent in role -- by up to 28% of a season, enough to
    # push four player-seasons over the 70% label guard they should fail.
    counts = (b.groupby(["player_id", "year"]).r.value_counts()
                .unstack().reindex(columns=ROLES).fillna(0.0))
    sh = counts.div(b.groupby(["player_id", "year"]).size(), axis=0)
    lab = b.groupby(["player_id", "year"]).agg(
        handle=("handle", "last"),
        main=("Agents", lambda s: s.mode().iat[0]),
        agent_share=("Agents", lambda s: s.value_counts(normalize=True).iat[0]),
        n_agents=("Agents", "nunique"))
    return ps.merge(sh, on=["player_id", "year"]).merge(lab, on=["player_id", "year"])


# ---- does a player's agent travel with them? (decision 08) ----


def ownership_pairs() -> pd.DataFrame:
    """One row per year-over-year pair: did they move, and did they keep the agent?

    Two corrections against the first version of this diagnostic, both of which the
    rest of the project had already made and this file had not:

      FLOOR. It filtered at 20 maps, the floor decision 13 retired. Everything else
      in the repo is quoted at `D.MIN_MAPS`.

      TEAM KEY. It keyed "did the player move?" on the DISPLAYED team name, so the
      four franchises renamed mid-window (GIANTX, TALON, KIWOOM DRX, and NRG's
      mislabelled slot) read as roster moves that never happened. `data.py` carries
      `org` for exactly this, and decisions/07 already uses it; this did not.
    """
    d  = F.build(); b = d[d.Side=="both"]
    ps = (b.groupby(["player_id","year"])
            .agg(main=("Agents", lambda s: s.mode().iat[0]),
                 org=("org",     lambda s: s.mode().iat[0]),
                 maps=("Map","size")).reset_index())
    ps = ps[ps.maps >= D.MIN_MAPS]
    pool = b.groupby(["player_id","year"]).Agents.apply(lambda s: set(s.str.lower()))
    ps = ps.merge(pool.rename("pool"), on=["player_id","year"])
    # decision 16: the season's role is the modal ROLE across maps, not the role of
    # the modal agent. `with_role` is the one place that is computed.
    lab = with_role(F.seasons(d), d)[["player_id","year","role"]]
    ps = ps.merge(lab, on=["player_id","year"], how="left")

    rows = []
    for y in (2023, 2024, 2025):
        a = ps[ps.year == y].set_index("player_id")
        c = ps[ps.year == y+1].set_index("player_id")
        for pid in a.index.intersection(c.index):
            x, z = a.loc[pid], c.loc[pid]
            j = len(x["pool"] & z["pool"]) / len(x["pool"] | z["pool"])
            rows.append({"moved": x.org != z.org, "same_main": x["main"] == z["main"],
                         "pool_overlap": j, "role": x.role, "same_role": x.role == z.role})
    return pd.DataFrame(rows)


def agent_retention() -> tuple:
    """(stayers, movers) share keeping their most-played agent.

    The site quotes both. They live here, in the pipeline, because nothing may import
    `diagnostics/` -- those modules print and are read beside a decision file. A number
    the site states has to be computable without running a diagnostic by hand."""
    g = ownership_pairs().groupby("moved").same_main.mean()
    return float(g[False]), float(g[True])
