# Default qualitative-judge rubric

Score the model's answer on a 1-5 integer scale. The dimensions
weighted are clarity, faithfulness to the provided context, and
calibrated refusal on uncertainty.

## Score 5 - Excellent
- Concise, fully correct, and grounded in the supplied context.
- Cites the relevant chunk id(s) for every load-bearing fact.
- Refuses cleanly when the context lacks the needed evidence.

## Score 4 - Good
- Mostly correct and grounded; minor unsupported claim or missing
  citation that does not change the substance of the answer.

## Score 3 - Acceptable
- Partially correct. Some claims are supported, others are
  unsupported or vague. Citations may be missing or imprecise.

## Score 2 - Poor
- Mostly unsupported or off-topic. Confabulates facts not in the
  context, or misses the question's intent.

## Score 1 - Fail
- Wrong, incoherent, or hallucinated. No grounding in the context;
  no useful refusal.

Override per-course: edit `eval/rubric.md` in the course tree to
inject domain-specific scoring criteria (technical-element coverage,
refusal-rather-than-confabulate behaviour, etc.).
