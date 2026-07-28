# AGENTS.md

## Project

`boiling-data` is a utility library for converting between boiling data formats.

The core idea: read data in a specific format into a Python object, then write that
object out in another format. 
The conversions between different formats can always go through this single common
Python object that essentially defines various to_\*/from_\* methods to convert to/from
different file formats. We can also use this to convert numpy and pytorch tensor data 
into 

The implementations for different filetypes should basically be independent.

## Layout

- `src/boiling_data/` — package source (src-layout, `py.typed`)
- `tests/` — pytest suite

Dependencies are managed with `uv`. Use `uv add <pkg>` rather than editing
`pyproject.toml` by hand, and commit the resulting `uv.lock`.

## Code style

**Satisfy the linter.** Code must pass `uv run ruff check .` clean before it is
done. Run `uv run ruff format .` to handle formatting.

**Run the tests.** Every change runs `uv run pytest`. New behavior comes with a
test; a change that alters behavior updates the tests that covered it. Do not
report work as finished without having actually run the suite.

**Keep it clean.** Write comments only for what the code cannot say itself — a
non-obvious constraint, a format quirk, a reason for an odd workaround. Do not
narrate what the next line does, restate the function signature, or leave notes
addressed to a reviewer. Prefer a clearer name or a smaller function over a
comment explaining a confusing one.

Type hints are expected; `mypy` runs in strict mode (`uv run mypy`).

## Commands

```bash
uv sync              # install deps
uv run pytest        # tests
uv run ruff check .  # lint
uv run ruff format . # format
uv run mypy          # type check
```
