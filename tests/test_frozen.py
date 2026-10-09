import copy
import json
import pickle

import pytest

from boiling_data.frozen import FrozenDict, freeze


def _frozen() -> FrozenDict:
    frozen = freeze({"physical": {"bulk_temp": 58.0}, "sites": [0.0, 1.0]})
    assert isinstance(frozen, FrozenDict)
    return frozen


@pytest.mark.parametrize(
    "change",
    [
        lambda frozen: frozen.__setitem__("x", 1),
        lambda frozen: frozen.__delitem__("sites"),
        lambda frozen: frozen.update({"x": 1}),
        lambda frozen: frozen.pop("sites"),
        lambda frozen: frozen.popitem(),
        lambda frozen: frozen.setdefault("x", 1),
        lambda frozen: frozen.clear(),
    ],
)
def test_frozen_dict_cannot_be_changed(change) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(TypeError, match="cannot be changed"):
        change(_frozen())


def test_freeze_reaches_every_level() -> None:
    frozen = _frozen()
    assert isinstance(frozen["physical"], FrozenDict)
    assert frozen["sites"] == (0.0, 1.0)


def test_frozen_dict_still_behaves_as_a_dict() -> None:
    frozen = _frozen()
    assert frozen["physical"] == {"bulk_temp": 58.0}
    assert json.loads(json.dumps(frozen)) == {
        "physical": {"bulk_temp": 58.0},
        "sites": [0.0, 1.0],
    }


def test_frozen_dict_pickles_and_copies() -> None:
    frozen = _frozen()
    for restored in (pickle.loads(pickle.dumps(frozen)), copy.deepcopy(frozen)):
        assert restored == frozen
        assert isinstance(restored["physical"], FrozenDict)
