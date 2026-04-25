"""
Ed4All CLI command subpackage.

Commands defined here are attached to the top-level ``ed4all`` Click group
in :mod:`cli.main``. Wave 7 adds the canonical ``ed4all run`` command;
Wave 34 adds the ``ed4all mailbox watch`` outer-session watcher.
Wave 77 adds ``ed4all libv2 query`` (faceted chunk explorer),
``ed4all libv2 generate-quiz`` (bloom-balanced assessment generator),
and ``ed4all libv2 generate-study-pack`` (study-pack / lesson-plan
generator).
"""

# Importing libv2_generate_quiz attaches the ``generate-quiz``
# subcommand to the shared ``libv2_group`` Click group at import time.
from . import libv2_generate_quiz  # noqa: F401
from .libv2_generate_study_pack import register_generate_study_pack_command
from .libv2_query import register_libv2_query_command
from .libv2_validate_packet import register_libv2_command as _register_libv2_validate_packet
from .libv2_validate_packet import libv2_group as _libv2_group
from .mailbox_watch import register_mailbox_command
from .run import register_run_command
from .state_prune import register_state_command
from .tutor import register_tutor_command


def register_libv2_command(cli_group):
    """Register the full ``ed4all libv2`` command group.

    Combines Wave 75's ``validate-packet``, Wave 77 Worker β's
    ``query``, Wave 77 Worker γ's ``generate-quiz``, and Wave 77
    Worker δ's ``generate-study-pack`` subcommands into the single
    ``libv2`` group so ``ed4all libv2 ...`` sees all four. Idempotent.
    """
    register_libv2_query_command(_libv2_group)
    register_generate_study_pack_command(_libv2_group)
    _register_libv2_validate_packet(cli_group)


__all__ = [
    "register_run_command",
    "register_mailbox_command",
    "register_state_command",
    "register_libv2_command",
    "register_libv2_query_command",
    "register_generate_study_pack_command",
    "register_tutor_command",
]
