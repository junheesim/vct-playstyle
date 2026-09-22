"""The sign convention, and the bug that made it worth testing.

`loadings` used to flip the loading table without flipping the scores, so PC1 could
have pointed one way in the table and the other way in every chart. It never fired,
because PC1 came out positive on this data. These tests fire it deliberately.
"""
import numpy as np
import pytest

import pca


@pytest.fixture
def anchored():
    """Two frames whose PC1 sklearn returns pointing OPPOSITE ways at the anchor.

    sklearn pins each component so its largest-magnitude entry is positive. So when
    the anchor is the weaker, anti-correlated feature, PC1 comes back loading
    NEGATIVELY on it -- which is the case the convention exists to correct.
    """
    import pandas as pd
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    cols = [pca.ANCHOR, "b", "c"]
    down = pd.DataFrame({pca.ANCHOR: -0.5 * x + 0.5 * rng.normal(size=300),
                         "b": x, "c": x + 0.2 * rng.normal(size=300)})[cols]
    return cols, down.assign(**{pca.ANCHOR: -down[pca.ANCHOR]}), down


def test_pc1_always_points_at_the_anchor(anchored):
    cols, up, down = anchored
    for df in (up, down):
        Z = pca.standardize(df, cols)
        model, flip = pca.fit(Z.values, cols)
        L = pca.loadings(model, flip, cols)
        assert L.loc[pca.ANCHOR, "PC1"] > 0


def test_loadings_and_scores_share_one_sign(anchored):
    """The regression test. Reconstructing a feature from scores x loadings must
    recover its real direction -- which it cannot if only one of the two was flipped."""
    cols, up, down = anchored
    for df in (up, down):
        Z = pca.standardize(df, cols)
        model, flip = pca.fit(Z.values, cols)
        L, S = pca.loadings(model, flip, cols), pca.scores(model, flip, Z, df.index)
        recon = S["PC1"] * L.loc[pca.ANCHOR, "PC1"]
        assert np.corrcoef(recon, Z[pca.ANCHOR])[0, 1] > 0.5


def test_flip_is_exercised_by_the_fixture(anchored):
    """If neither frame triggers a flip the two tests above prove nothing."""
    cols, up, down = anchored
    flips = {pca.fit(pca.standardize(df, cols).values, cols)[1] for df in (up, down)}
    assert flips == {1.0, -1.0}


def test_scores_carry_the_index_they_are_given():
    """pandas concat aligns on LABELS. A frame whose index is gapped (after a dropna)
    concatenated with scores on a fresh RangeIndex unions the two, producing a longer
    scrambled frame rather than an error. That happened in robustness.run: 847 rows
    and R2 role .015 instead of 775 and .605."""
    import pandas as pd
    rng = np.random.default_rng(1)
    cols = [pca.ANCHOR, "b"]
    df = pd.DataFrame(rng.normal(size=(20, 2)), columns=cols).drop(index=[3, 7, 11])
    Z = pca.standardize(df, cols)
    model, flip = pca.fit(Z.values, cols)
    s = pca.scores(model, flip, Z.values, df.index)
    assert s.index.equals(df.index)
    joined = pd.concat([df, s], axis=1)
    assert len(joined) == len(df), "concat unioned two different indexes"
    assert joined.notna().all().all()
