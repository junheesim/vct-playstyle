"""Phase 4b -- are there discrete archetypes, or is playstyle continuous?

Clustering ALWAYS returns clusters. k-means asked for 5 groups gives 5 groups on any
data, including a featureless cloud. So the question is never "what are the clusters"
but "is there cluster structure at all".

Two checks, the same pair used throughout this project:

  REPLICATION  cluster two random halves separately; do they agree about who groups
               with whom? (adjusted Rand index: 1 = identical, 0 = chance.)
               Real gaps are found by both halves. In a continuous cloud the slicing
               lines are arbitrary and land somewhere different each time.

  NULL         run the identical procedure on data with NO group structure. An
               arbitrary cut of a cloud still replicates at .29-.45, so the real
               number is meaningless without this baseline.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture

import features as F, pca
from components import loadings

KS = range(2, 9)


def replication(X, k, pid, model="kmeans", n=40, seed=0):
    rng = np.random.default_rng(seed)
    fit = (lambda d: KMeans(k, n_init=4, random_state=0).fit(d)) if model == "kmeans" \
          else (lambda d: GaussianMixture(k, n_init=1, random_state=0).fit(d))
    out = []
    for _ in range(n):
        # split by PLAYER, not by row: 775 rows come from 400 players, and a row-wise
        # split leaks the same player into both halves.
        ps = rng.permutation(np.unique(pid)); h = set(ps[:len(ps)//2])
        m = np.array([p in h for p in pid]); a, b = np.where(m)[0], np.where(~m)[0]
        out.append(ari(fit(X[a]).predict(X), fit(X[b]).predict(X)))
    return float(np.median(out))


def null_like(X, seed=0):
    """A cloud with the same mean, spread and correlation as X, but no groups."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), len(X))


def report(X, pid, label):
    print(f"\n{'='*66}\n{label}  (n={len(X)})\n{'='*66}")
    print(f"  {'k':<4}{'kmeans real':>13}{'kmeans null':>13}{'gmm real':>11}{'gmm null':>11}{'':>6}")
    Z = null_like(X)
    for k in KS:
        kr, kn = replication(X, k, pid, "kmeans"), replication(Z, k, pid, "kmeans")
        gr, gn = replication(X, k, pid, "gmm"),    replication(Z, k, pid, "gmm")
        beats = "REAL" if (kr > kn + .10 and gr > gn + .10) else ""
        print(f"  {k:<4}{kr:>13.3f}{kn:>13.3f}{gr:>11.3f}{gn:>11.3f}{beats:>6}")
    print("\n  'REAL' = beats the no-structure baseline by >.10 on BOTH methods.")


if __name__ == "__main__":
    M, L, sc = loadings(F.STYLE)
    pid = M.player_id.values
    report(sc[["PC1","PC2"]].values, pid, "STYLE, 2 components")
    Zf = pca.standardise(M, F.STYLE).values
    report(Zf, pid, f"STYLE, all {len(F.STYLE)} raw features")
