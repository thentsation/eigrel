"""Code generators that turn Eigrel IR into runnable programs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    """A module the generated code imports, and the eigrel extra that installs it."""

    module: str
    extra: str


class UnsupportedError(Exception):
    """The program is valid, but this backend cannot express part of it."""
