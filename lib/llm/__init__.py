"""Unified LLM/API endpoint registry package.

``lib.llm.endpoints`` is the single attach point for every LLM/API
endpoint the codebase talks to. Code attaches to endpoints BY NAME; the
transport + identity flow from ``config/endpoints.yaml``. See
``lib/llm/endpoints.py`` for the public surface.
"""
