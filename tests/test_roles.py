"""The agent -> role lookup, and the two ways it used to fail silently."""
import pandas as pd
import pytest

import roles


def test_multi_agent_cell_resolves_to_the_first_agent():
    """There were two spellings of this mapping, one that split the cell on a comma
    and one that did not, so a multi-agent cell became a role in one code path and
    NaN in the other. The data has no such cells today; the lookup handles them."""
    s = pd.Series(["jett", "jett, raze", " sage , omen "])
    assert list(roles.role_of(s)) == ["duelist", "duelist", "sentinel"]


def test_unknown_agent_raises():
    """An agent Riot adds after this was written used to map to NaN and drop out of
    every role-based result without a word."""
    with pytest.raises(KeyError, match="missing from ROLE"):
        roles.role_of(pd.Series(["jett", "some_new_agent"]))


def test_known_non_agents_stay_unmapped_without_raising():
    out = roles.role_of(pd.Series(sorted(roles.NOT_AN_AGENT)))
    assert out.isna().all()


def test_every_role_value_is_one_of_the_four():
    assert set(roles.ROLE.values()) == {"duelist", "initiator", "controller", "sentinel"}


def test_every_agent_in_the_data_has_a_role():
    """`veto` (sentinel) and `miks` (controller) were unmapped, so 64 and 2 maps
    counted toward no role at all. Riot adds agents; this fails loudly when it does."""
    import features as F
    b = F.build()
    b = b[b.Side == "both"]
    assert roles.role_of(b.Agents).notna().all()


def test_nothing_is_excluded_by_accident():
    """NOT_AN_AGENT is the deliberate escape hatch. It is empty, and should only grow
    with a stated reason -- silence is how veto went missing in the first place."""
    assert roles.NOT_AN_AGENT == set()
