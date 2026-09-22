"""Phase 3 -- the style matrix.  Implements decisions 08 and 10.

Two layers, deliberately:

  ANALYSIS layer   each feature residualized on quality (kd_ratio).
                   Baseline = a player of equal SKILL. Role is NOT controlled for
                   (decision 08) so that cross-role structure can be SHOWN rather
                   than engineered.

  DESCRIPTIVE layer  each feature also expressed as a within-role z-score.
                   Baseline = the average player in the same Riot role. Never an
                   input to the model -- it exists so results are readable:
                   "a duelist who plays like a sentinel".
"""

import numpy as np, statsmodels.api as sm
import pandas as pd

import features as F
import paths
from roles import with_role


def residualize(ps: pd.DataFrame, cols, covariate: str = None) -> pd.DataFrame:
    """Each feature minus what the covariate predicts. Standardized afterwards so
    every feature enters the style space on the same scale."""
    covariate = covariate or F.QUALITY
    out = {}
    for f in cols:
        ok = ps[f].notna() & ps[covariate].notna()
        r = pd.Series(np.nan, index=ps.index)
        r[ok] = sm.OLS(ps.loc[ok, f], sm.add_constant(ps.loc[ok, covariate])).fit().resid
        out[f] = (r - r.mean()) / r.std()
    return pd.DataFrame(out, index=ps.index)


def within_role(ps: pd.DataFrame, cols) -> pd.DataFrame:
    """Descriptive only. Position within the player's own Riot role, in sd units."""
    g = ps.groupby("role")
    return pd.DataFrame({f: (ps[f] - g[f].transform("mean")) / g[f].transform("std")
                         for f in cols}, index=ps.index)


def build(feature_set=None) -> pd.DataFrame:
    cols = feature_set or F.STYLE
    d  = F.build()
    ps = with_role(F.seasons(d), d)
    S  = residualize(ps, cols)
    W  = within_role(ps, cols).add_suffix("_inrole")
    lab = (d[d.Side == "both"].groupby(["player_id", "year"])
             .agg(handle=("handle", "last"),
                  team=("Team", lambda s: s.mode().iat[0]),
                  region=("region", lambda s: s.mode().iat[0])))
    ps = ps.merge(lab, on=["player_id", "year"], how="left")
    keys = ["player_id", "handle", "year", "team", "region", "role", "role_share",
            "main_agent", "maps", F.QUALITY, "team_win"]
    return pd.concat([ps[keys], S, W], axis=1).dropna(subset=cols)


if __name__ == "__main__":
    for name, cols in [("STYLE", F.STYLE)]:
        S = build(cols)
        print(f"\n{name}: {S.shape[0]} player-seasons x {len(cols)} features")
        q = S[cols].corrwith(S[F.QUALITY]).abs()
        print(f"  residual |r| with quality: mean {q.mean():.3f}  max {q.max():.3f} ({q.idxmax()})")
        if name == "STYLE":
            paths.INTERIM.mkdir(parents=True, exist_ok=True)
            S.to_parquet(paths.INTERIM / "style_matrix.parquet")
            print(f"  wrote {paths.INTERIM/'style_matrix.parquet'}\n")
            show = ["handle","year","team","role","main_agent",
                    "first_engagement","clutch_att","creds_per_round"]
            def block(title, df):
                print(f"  {title}")
                print("   " + df[show].round(2).to_string(index=False).replace("\n","\n   ") + "\n")
            block("MOST AGGRESSIVE (highest first_engagement):", S.nlargest(5,"first_engagement"))
            block("LEAST AGGRESSIVE:", S.nsmallest(5,"first_engagement"))
            du = S[S.role=="duelist"]
            block("DUELISTS who take the FEWEST opening duels (off-role):",
                  du.nsmallest(5,"first_engagement"))
            sen = S[S.role=="sentinel"]
            block("SENTINELS who take the MOST opening duels (off-role):",
                  sen.nlargest(5,"first_engagement"))
