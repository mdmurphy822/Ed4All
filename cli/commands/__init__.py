"""
Ed4All CLI command subpackage.

Commands defined here are attached to the top-level ``ed4all`` Click group
in :mod:`cli.main``. Wave 7 adds the canonical ``ed4all run`` command;
Wave 34 adds the ``ed4all mailbox watch`` outer-session watcher.
Wave 77 adds ``ed4all libv2 query`` (faceted chunk explorer),
``ed4all libv2 generate-quiz`` (bloom-balanced assessment generator),
and ``ed4all libv2 generate-study-pack`` (study-pack / lesson-plan
generator). Wave 78 adds ``ed4all libv2 ask`` (intent-routed
natural-language query).
"""

# Importing libv2_generate_quiz attaches the ``generate-quiz``
# subcommand to the shared ``libv2_group`` Click group at import time.
from . import libv2_generate_quiz  # noqa: F401
from .assistant import register_assistant_command
from .backup import register_backup_command
from .convert import register_convert_command
from .doctor import register_doctor_command
from .gui_cmd import register_gui_command
from .harvest_bloom_labels import register_harvest_bloom_labels_command
from .import_docs import register_import_docs_command
from .libv2_ask import register_libv2_ask_command
from .libv2_generate_study_pack import register_generate_study_pack_command
from .libv2_query import register_libv2_query_command
from .libv2_validate_packet import libv2_group as _libv2_group
from .libv2_validate_packet import register_libv2_command as _register_libv2_validate_packet
from .mailbox_watch import register_mailbox_command
from .objectives_cmd import register_objectives_command
from .run import register_run_command
from .state_prune import register_state_command
from .stop import register_stop_command
from .support_bundle import register_support_bundle_command
from .tutor import register_tutor_command


def register_libv2_command(cli_group):
    """Register the full ``ed4all libv2`` command group.

    Combines Wave 75's ``validate-packet``, Wave 77 Worker β's
    ``query``, Wave 77 Worker γ's ``generate-quiz``, Wave 77 Worker
    δ's ``generate-study-pack``, and Wave 78 Worker C's ``ask``
    (intent-routed query) subcommands into the single ``libv2`` group
    so ``ed4all libv2 ...`` sees all five. Idempotent.
    """
    register_libv2_query_command(_libv2_group)
    register_generate_study_pack_command(_libv2_group)
    register_libv2_ask_command(_libv2_group)
    _register_libv2_validate_packet(cli_group)


__all__ = [
    "register_run_command",
    "register_mailbox_command",
    "register_state_command",
    "register_stop_command",
    "register_objectives_command",
    "register_libv2_command",
    "register_libv2_query_command",
    "register_libv2_ask_command",
    "register_generate_study_pack_command",
    "register_tutor_command",
    "register_gui_command",
    "register_doctor_command",
    "register_import_docs_command",
    "register_harvest_bloom_labels_command",
    "register_convert_command",
    "register_support_bundle_command",
    "register_backup_command",
    "register_assistant_command",
]
