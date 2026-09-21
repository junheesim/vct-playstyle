"""End-to-end checks on the real data.

These pin the claims the report makes. If the data source is refreshed and one of
them moves, that is information, not a nuisance -- the prose has to move with it.
"""
import pytest

import data as D
import features as F
from components import eigenvalues, null_cutoffs
from examples import guarded
import pca


@pytest.fixture(scope="module")
def maps():
    return F.build()


@pytest.fixture(scope="module")
def M():
    import style as S
    return S.build(F.STYLE)


def test_grain_is_unique():
    """One row per player x map x side. Everything downstream assumes it."""
    assert not D.overview().duplicated(D.GRAIN).any()


def test_overview_is_not_mutated_by_its_callers(maps):
    """`overview` and `build` are cached now, so a caller that adds a column in place
    would corrupt the copy every later script receives."""
    a = D.overview()
    a["scratch"] = 1
    assert "scratch" not in D.overview().columns


def test_headline_scope(M):
    assert len(M) == 775
    assert M.player_id.nunique() == 400
    assert set(M.year) == {2023, 2024, 2025, 2026}


def test_two_components_survive_the_null(M):
    X = pca.standardise(M, F.STYLE).values
    assert int((eigenvalues(X) > null_cutoffs(X, "shuffle", n=200)).sum()) == 2


def test_pc1_is_the_aggression_axis(M):
    """The report calls PC1 aggression. That is only legitimate while opening duels
    load positively on it and most strongly."""
    Z = pca.standardise(M, F.STYLE)
    model, flip = pca.fit(Z.values, F.STYLE)
    L = pca.loadings(model, flip, F.STYLE).PC1
    assert L["first_engagement"] > 0
    assert L.abs().idxmax() == "first_engagement"


def test_every_style_feature_is_present_and_finite(M):
    assert M[F.STYLE].notna().all().all()


def test_label_guard_is_stricter_than_the_raw_label():
    G = guarded()
    assert G.label_ok.sum() < len(G)
    assert (G.loc[G.label_ok, "role_share"] >= .70).all()


def test_seasons_respect_the_map_floor(maps):
    assert F.seasons(maps).maps.min() >= D.MIN_MAPS


def test_the_map_floor_is_fifteen():
    """Decision 13. Everything in decisions/ and on the site is quoted at this floor;
    the 20-map floor it replaced is retired, not carried alongside."""
    assert D.MIN_MAPS == 15


def test_role_shares_are_denominated_over_every_map():
    """A map on an agent outside the role taxonomy is not a map in the labelled role.
    Dropping it from the denominator overstated role_share by up to 28% of a season,
    enough to push four player-seasons over the 70% label guard they should fail."""
    from roles import shares, ROLES
    P = shares()
    assert (P[ROLES].sum(axis=1) <= 1.0 + 1e-9).all()
    assert (P[ROLES].max(axis=1) <= 1.0 + 1e-9).all()


def test_label_guard_is_exactly_the_role_share_rule():
    """One condition since decision 16: did the player actually spend most of the
    season in the role they are labelled with."""
    G = guarded()
    assert (G.label_ok == (G.role_share >= .70)).all()


def test_role_is_the_modal_role_not_the_modal_agents_role(maps):
    """Decision 16. `role` is the most common role ACROSS a player's maps, which
    covers a median 81% of them; the single most-played agent covers 44%."""
    from roles import ROLES, role_of, with_role
    w = with_role(F.seasons(maps), maps)
    b = maps[maps.Side == "both"]
    counts = (b.assign(_r=role_of(b.Agents)).groupby(["player_id", "year"])._r
                .value_counts().unstack().reindex(columns=ROLES).fillna(0.0))
    expected = counts.idxmax(axis=1).rename("role").reset_index()
    got = w[["player_id", "year", "role"]].reset_index(drop=True)
    assert got.merge(expected, on=["player_id", "year"]).eval("role_x == role_y").all()
    assert w.role_share.median() > 0.75
    # and it really is a different label from the one it replaced
    assert (w.role != role_of(w.main_agent)).sum() > 0
