"""Five-specialist BERT council for the SemantiK compatibility cascade.

The preferred GLM-OCR lane bypasses this package, but the reachable
compatibility cascade still routes through five registered specialists:
``merge_or_split``, ``structure``, ``semantic``, ``table_specialist``, and
``math_specialist``. They share a ModernBERT backbone and swap LoRA adapters
sequentially. This package owns:

The package's reachability does not qualify its weights. A trained checkpoint
must pass the council evaluation contract before this path can support a
production claim.

- types.py     : the data contracts (TypedSignal, BertOutput, CouncilState)
- base.py      : the shared backbone abstraction + LoRA loader stub
- registry.py  : (bert_name -> adapter_path, head_spec) lookup
- routing.py   : the static DAG describing which BERT runs when
- config.yaml  : per-specialist hyperparameters + ModernBERT backbone choice

Imports remain lightweight: specialist modules register lazily when the runner
dispatches them, and model weights are never loaded merely by importing the
package.
"""
