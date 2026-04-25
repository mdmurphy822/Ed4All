"""
Ed4All CLI command subpackage.

Commands defined here are attached to the top-level ``ed4all`` Click group
in :mod:`cli.main`. Wave 7 adds the canonical ``ed4all run`` command;
Wave 34 adds the ``ed4all mailbox watch`` outer-session watcher.
"""

from .libv2_validate_packet import register_libv2_command
from .mailbox_watch import register_mailbox_command
from .run import register_run_command
from .state_prune import register_state_command

__all__ = [
    "register_run_command",
    "register_mailbox_command",
    "register_state_command",
    "register_libv2_command",
]
