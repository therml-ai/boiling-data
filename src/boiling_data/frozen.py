from typing import Any, NoReturn


class FrozenDict(dict[str, Any]):
    """A dict that cannot be changed after it is built. It is still a dict, so it
    compares equal to one, serializes to JSON, and pickles, which a DataLoader
    needs to send batches back from its worker processes."""

    def _immutable(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} cannot be changed")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __reduce__(self) -> tuple[type["FrozenDict"], tuple[dict[str, Any]]]:
        # the default rebuilds a dict subclass item by item through __setitem__
        return (type(self), (dict(self),))


def freeze(value: Any) -> Any:
    """value with every dict, at any depth, a FrozenDict and every list a tuple."""
    if isinstance(value, dict):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value
