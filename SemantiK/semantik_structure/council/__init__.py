"""Five-specialist BERT council for the SemantiK compatibility cascade.

The default-off GLM-OCR lane bypasses this package, but the byte-compatible
omni cascade still routes through five registered specialists:
``merge_or_split``, ``structure``, ``semantic``, ``table_specialist``, and
``math_specialist``. They share a ModernBERT backbone and swap LoRA adapters
sequentially. This package owns:

- types.py     : the data contracts (TypedSignal, BertOutput, CouncilState)
- base.py      : the shared backbone abstraction + LoRA loader stub
- registry.py  : (bert_name -> adapter_path, head_spec) lookup
- routing.py   : the static DAG describing which BERT runs when
- config.yaml  : per-specialist hyperparameters + ModernBERT backbone choice

Imports remain lightweight: specialist modules register lazily when the runner
dispatches them, and model weights are never loaded merely by importing the
package.
"""
