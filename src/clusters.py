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
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

import features as F, pca
from components import loadings

KS = range(2, 9)
N_NULLS = 200        # draws behind every null band; one draw is not a baseline


def replication(X, k, pid, model="kmeans", n=40, seed=0, n_=None):
    n = n_ or n          # `n_` lets null_band pass a rep count past its own `n`
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
    """A cloud with the same mean, spread and correlation as X, but no groups.

    The null KEEPS the covariance on purpose. The shuffle null used for component
    retention would destroy it, turning the comparison cloud into a round blob while
    the real data stays an elongated, tilted one -- and the test would then "find
    structure" in the elongation rather than in any clumping. Each question needs a
    null that removes the thing being tested for and nothing else.
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), len(X))


def separation(X, k):
    """Silhouette: for each point, how much closer is its own group than the nearest
    other one, as a share of the larger distance. 1 = clean gaps, 0 = none."""
    return float(silhouette_score(X, KMeans(k, n_init=20, random_state=0).fit_predict(X)))


def null_band(X, k, stat, n=N_NULLS, **kw):
    """`stat` run on n structureless clouds -> (median, 5th, 95th, mean, sd).

    ONE null draw is not a baseline, it is a single guess. This was
    `null_like(X, seed=0)` until 2026-09-21, and the single draw it happened to
    produce made the replication result look decisive when the real null spans
    .31 to .78, and made the separation result look like nothing when the real null
    spans only .31 to .34. Both readings in the report were wrong, in opposite
    directions, from the same line of code.
    """
    v = np.array([stat(null_like(X, seed=s), k, **kw) for s in range(n)])
    # mean and sd, not a percentile range: a plus-or-minus is symmetric by
    # construction and reads far better than an asymmetric interval in brackets.
    # Two sd covers ~95% of the draws, which is the bar the verdict column uses.
    return (float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95)),
            float(v.mean()), float(v.std(ddof=1)))


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
    Zf = pca.standardize(M, F.STYLE).values
    report(Zf, pid, f"STYLE, all {len(F.STYLE)} raw features")
