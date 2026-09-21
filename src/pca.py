"""The one place the style space is fitted.

Standardise, rotate, and fix the sign. That block appeared in four files with three
different spellings, and the copy that read differently was the one carrying a bug:
it flipped the loadings without flipping the scores, so the axis could have read one
way in the loading table and the other way in every chart. Fitting through one
function makes that class of mistake impossible rather than merely absent.

SIGN CONVENTION. A component's sign is arbitrary -- PCA is free to return either
direction -- so it is pinned: PC1 always points toward MORE opening duels, which is
what lets the axis be called aggression. PC2 is left as returned; nothing in the
report depends on its direction.
"""

import pandas as pd
from sklearn.decomposition import PCA

ANCHOR = "first_engagement"


def standardise(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Mean 0, sd 1 per column, so PCA weights structure and not units."""
    return (df[cols] - df[cols].mean()) / df[cols].std()


def fit(Z, cols, k: int = 2):
    """Fit on Z and return (pca, flip). Apply `flip` to BOTH loadings and scores."""
    p = PCA(n_components=k).fit(Z)
    return p, (-1.0 if p.components_[0][list(cols).index(ANCHOR)] < 0 else 1.0)


def _names(model) -> list[str]:
    return [f"PC{i+1}" for i in range(model.n_components_)]


def scores(model, flip: float, Z, index=None) -> pd.DataFrame:
    Z = Z.values if hasattr(Z, "values") else Z
    s = pd.DataFrame(model.transform(Z), index=index, columns=_names(model))
    s["PC1"] *= flip
    return s


def loadings(model, flip: float, cols) -> pd.DataFrame:
    """What each component is MADE OF, on the same sign convention as the scores."""
    L = pd.DataFrame(model.components_.T, index=list(cols), columns=_names(model))
    L["PC1"] *= flip
    return L
