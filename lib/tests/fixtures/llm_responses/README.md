# LLM Response Fixtures

Deterministic LLM responses used by the Wave 7 MockBackend when testing
the refactored call sites.

## Layout

Fixtures are named by the first 16 hex chars of
``sha256("system\nuser").hexdigest()``. Each file is a JSON object with a
single ``text`` field containing the canned response text.

Call sites using these fixtures:

- ``DART/pdf_converter/claude_processor.py`` — structure analysis
- ``DART/pdf_converter/alt_text_generator.py`` — image alt text
- ``Trainforge/align_chunks.py`` — teaching-role classification

Tests construct ``MockBackend(fixture_dir=...)`` and inject it via the new
``llm=`` kwarg on each class / function.
