"""BERT-Council package (Stage 3 of the v2 pipeline).

The council is a fan of small Role/Span/Region/Math/Table/Order/Semantic BERTs
sharing a single backbone encoder, each with a LoRA adapter and a typed
head. This package owns:

- types.py     : the data contracts (TypedSignal, BertOutput, CouncilState)
- base.py      : the shared backbone abstraction + LoRA loader stub
- registry.py  : (bert_name -> adapter_path, head_spec) lookup
- routing.py   : the static DAG describing which BERT runs when
- config.yaml  : per-BERT hyperparameters + base-encoder choice (DeBERTa-v3-base)

Phase 0 lands type definitions + empty registry + DAG config only. No model
weights are loaded; importing this package is free (stdlib only).
"""
