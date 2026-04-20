"""
Ed4All CLI command subpackage.

Commands defined here are attached to the top-level ``ed4all`` Click group
in :mod:`cli.main`. Wave 7 adds the canonical ``ed4all run`` command; future
waves will migrate legacy subcommands into this package.
"""

from .run import register_run_command

__all__ = ["register_run_command"]
