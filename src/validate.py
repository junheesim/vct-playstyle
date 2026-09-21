"""Phase 5 -- does the style space hold on a year the model never saw?

Everything so far was fit on all four years at once, so the model has seen every
player it describes. A model can find patterns that fit its own data and nothing else.

Here the ENTIRE pipeline is fitted on 2023-2025 only:
    - the quality-adjustment coefficients (feature ~ kd_ratio)
    - the standardisation (mean and sd of each residual)
    - the PCA rotation
2026 is then pushed through that fixed machinery cold. It contributes nothing to any
fitted quantity, so it is genuinely out of sample.

Caveat to state, not to correct for: 2026 has no Champions event (decision 02).
"""

import pandas as pd
import statsmodels.api as sm

import features as F, pca
from roles import with_role

TRAIN, TEST = [2023, 2024, 2025], 2026


def fit_and_project(cols):
    d  = F.build()
    ps = with_role(F.seasons(d), d)
    lab = (d[d.Side=="both"].groupby(["player_id","year"])
             .agg(handle=("handle","last"), team=("Team", lambda s: s.mode().iat[0])))
    ps = ps.merge(lab, on=["player_id","year"], how="left").dropna(subset=cols+[F.QUALITY])
    tr, te = ps[ps.year.isin(TRAIN)], ps[ps.year == TEST]

    # --- everything below is FITTED ON TRAIN ONLY ---
    coef, mu, sd = {}, {}, {}
    Rtr = {}
    for f in cols:
        m = sm.OLS(tr[f], sm.add_constant(tr[[F.QUALITY]])).fit()
        coef[f] = m.params
        r = tr[f] - (m.params["const"] + m.params[F.QUALITY]*tr[F.QUALITY])
        mu[f], sd[f] = r.mean(), r.std()
        Rtr[f] = (r - mu[f]) / sd[f]
    Ztr = pd.DataFrame(Rtr, index=tr.index)
    model, flip = pca.fit(Ztr.values, cols)

    # --- apply the FIXED machinery to 2026 ---
    Rte = {f: ((te[f] - (coef[f]["const"] + coef[f][F.QUALITY]*te[F.QUALITY])) - mu[f]) / sd[f]
           for f in cols}
    Zte = pd.DataFrame(Rte, index=te.index)

    def score(Z, idx):
        return pca.scores(model, flip, Z, idx)
    out = pd.concat([pd.concat([tr[["player_id","year","handle","team","role","main_agent"]],
                                score(Ztr, tr.index)], axis=1),
                     pd.concat([te[["player_id","year","handle","team","role","main_agent"]],
                                score(Zte, te.index)], axis=1)])
    return out, model, flip


def yoy(D, y0, y1, col):
    a = D[D.year == y0].set_index("player_id")[col]
    b = D[D.year == y1].set_index("player_id")[col]
    j = a.index.intersection(b.index)
    return (a.loc[j].corr(b.loc[j]), len(j))


if __name__ == "__main__":
    cols = F.STYLE
    H, model, flip = fit_and_project(cols)
    print(f"fitted on {TRAIN}, projected {TEST} cold\n")
    print(f"  train {len(H[H.year!=TEST])} player-seasons | test {len(H[H.year==TEST])}")
    print(f"  variance explained by the two components on TRAIN: "
          f"{model.explained_variance_ratio_[0]:.1%}, {model.explained_variance_ratio_[1]:.1%}\n")

    from components import loadings
    M, L, sc = loadings(cols)
    IN = pd.concat([M[["player_id","year"]], sc], axis=1)

    print("  Does a player's position hold from one year to the next?\n")
    print(f"  {'':<28}{'PC1':>10}{'PC2':>10}{'n pairs':>10}")
    for y0, y1, tag in [(2023,2024,"in-sample"), (2024,2025,"in-sample")]:
        r1,n = yoy(IN, y0, y1, "PC1"); r2,_ = yoy(IN, y0, y1, "PC2")
        print(f"  {tag+' '+str(y0)+'->'+str(y1):<28}{r1:>10.3f}{r2:>10.3f}{n:>10}")
    r1,n = yoy(H, 2025, 2026, "PC1"); r2,_ = yoy(H, 2025, 2026, "PC2")
    print(f"  {'HELD-OUT 2025->2026':<28}{r1:>10.3f}{r2:>10.3f}{n:>10}")
    ri1,_ = yoy(IN, 2025, 2026, "PC1"); ri2,_ = yoy(IN, 2025, 2026, "PC2")
    print(f"  {'(same pair, in-sample)':<28}{ri1:>10.3f}{ri2:>10.3f}{n:>10}")
