"""Assessment-side helpers (honest IRT difficulty-calibration scaffold).

This package hosts the response-data-gated difficulty-provenance scaffold
(``TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD``). It deliberately ships NO fitted
1PL/2PL psychometric model — calibration is a deterministic
proportion-correct → difficulty-band map computed ONLY from real learner
responses, and the heuristic difficulty band is always kept when no
responses exist (never fabricated).
"""
