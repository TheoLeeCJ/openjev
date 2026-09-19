"""Create benchmark artifacts without replacing existing evidence."""

from contextlib import ExitStack
from pathlib import Path


def write_new_outputs(contents: dict[Path, str]) -> None:
    for path in contents:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output must be new: {path}")
    with ExitStack() as stack:
        destinations = {}
        for path in contents:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects against a writer racing the check.
            destinations[path] = stack.enter_context(path.open("x", encoding="utf-8"))
        for path, text in contents.items():
            destinations[path].write(text)
