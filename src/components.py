"""Phase 4 -- how many style dimensions are real?

PCA always returns components, even from pure noise: on 8 columns of random numbers
the first "component" took 14.2% instead of 12.5%. So variance explained means nothing
on its own, and variance-explained THRESHOLDS are arbitrary -- on this data 70% keeps
3, 80% keeps 4, 90% keeps 6.

Instead, build a null and keep only what beats it.

    NULL   shuffle each feature column independently. Every column keeps its exact
           real distribution; which values share a row is randomised, so every
           relationship BETWEEN features is destroyed.

Run PCA on that many times to see how large each eigenvalue gets when there is
genuinely nothing to find. A component is retained only if it exceeds the 95th
percentile of the null. (Horn's parallel analysis; the permutation variant is used
because the features are skewed -- plants .97, first_engagement_gap 1.78 -- so a
normal null would assume something untrue. The normal null is reported alongside as
a check.)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import features as F, style as S, pca
import paths

N_DRAWS = 500


def eigenvalues(X: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))[::-1]


def null_cutoffs(X: np.ndarray, kind: str = "shuffle", n: int = N_DRAWS, pct: int = 95):
    """95th-percentile eigenvalues under 'no relationships between features'."""
    rng = np.random.default_rng(0)
    sim = []
    for _ in range(n):
        if kind == "shuffle":
            Z = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        else:                                    # classical Horn: normal random data
            Z = rng.normal(size=X.shape)
        sim.append(eigenvalues(Z))
    return np.percentile(np.array(sim), pct, axis=0)


def replication(X: np.ndarray, pid: np.ndarray, n: int = 200):
    """Fit PCA on two halves; how much do the components agree?
    A component that does not reproduce on half the data is not a finding.
    Halves are split by PLAYER, not by row -- 775 rows come from 400 players, and a
    row-wise split leaks the same player into both halves.

    Component k in one half is compared with component k in the other. If the two
    halves order two near-equal components differently, that pair reads as a failure
    to replicate, so this understates agreement for the weaker components."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        ps = rng.permutation(np.unique(pid)); h = set(ps[:len(ps)//2])
        m = np.array([p in h for p in pid])
        a = PCA().fit(X[m]).components_
        b = PCA().fit(X[~m]).components_
        out.append([abs(float(np.dot(a[k], b[k]))) for k in range(X.shape[1])])
    o = np.array(out)
    return np.median(o, axis=0), np.percentile(o, 5, axis=0)


def report(cols=None, label="STYLE"):
    cols = cols or F.STYLE
    M = S.build(cols)
    X = ((M[cols] - M[cols].mean()) / M[cols].std()).values
    ev = eigenvalues(X)
    shuf, norm = null_cutoffs(X, "shuffle"), null_cutoffs(X, "normal")
    med, lo = replication(X, M.player_id.values)

    print(f"\n{'='*78}\n{label}: {X.shape[0]} player-seasons x {X.shape[1]} features\n{'='*78}")
    print(f"  {'':<6}{'eigenvalue':>12}{'var':>8}{'cum':>8}"
          f"{'null(shuf)':>12}{'null(norm)':>12}{'replicates':>12}{'':>8}")
    for i in range(len(ev)):
        keep = ev[i] > shuf[i]
        rep  = "yes" if lo[i] > .8 else ("marginal" if lo[i] > .6 else "no")
        print(f"  PC{i+1:<4}{ev[i]:>12.2f}{ev[i]/len(ev):>8.1%}{np.cumsum(ev)[i]/len(ev):>8.1%}"
              f"{shuf[i]:>12.2f}{norm[i]:>12.2f}{rep:>12}{'   KEEP' if keep else '':>8}")
    k = int((ev > shuf).sum())
    print(f"\n  parallel analysis (shuffle null) -> keep {k}"
          f"   |  normal null -> keep {int((ev>norm).sum())}"
          f"   |  Kaiser (ev>1) -> keep {int((ev>1).sum())}")
    for t in (.70, .80, .90):
        print(f"  cumulative >= {t:.0%} would keep {int(np.argmax(np.cumsum(ev)/len(ev)>=t)+1)}"
              + ("   <- the threshold is doing the work" if t == .70 else ""))
    return M, cols, k


def loadings(cols, k=2):
    """The style matrix, what each retained component is MADE OF, and the scores.

    Loadings and scores come out of `pca.fit` under one sign convention, so they
    cannot disagree about which direction PC1 points."""
    M = S.build(cols)
    Z = pca.standardise(M, cols)
    model, flip = pca.fit(Z.values, cols, k)
    return M, pca.loadings(model, flip, cols), pca.scores(model, flip, Z, M.index)


if __name__ == "__main__":
    for label, cols in [("STRICT", F.STRICT), ("STYLE", F.STYLE)]:
        report(cols, label)

    print(f"\n{'='*78}\nWHAT THE TWO COMPONENTS ARE MADE OF\n{'='*78}")
    for label, cols in [("STRICT", F.STRICT), ("STYLE", F.STYLE)]:
        _, L, _ = loadings(cols)
        print(f"\n  {label}")
        print("   " + L.round(2).to_string().replace("\n", "\n   "))

    M, L, sc = loadings(F.STYLE)
    out = pd.concat([M[["handle","year","team","region","role","main_agent"]], sc], axis=1)
    paths.INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(paths.INTERIM / "style_scores.parquet")
    print(f"\n  wrote {paths.INTERIM/'style_scores.parquet'}  {out.shape}")
