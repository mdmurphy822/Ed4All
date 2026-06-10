"""Offline-mode verification suite for the lexical retrieval pipeline.

Asserts the BM25 retrieval + citation-anchor path performs zero network I/O
by running it under the ``lib.testing.no_network`` socket guard. Selected by
path (``pytest tests/offline/``); no custom pytest marker is registered so the
repo's ``--strict-markers`` posture is untouched.
"""
