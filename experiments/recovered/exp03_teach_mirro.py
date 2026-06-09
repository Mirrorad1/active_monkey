# ============================================================
# Experiment 3 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01Qe1p2tQy1RU3xHnoBmgFWy
#   description  : Demonstrate: teach the model to 'say mirro' by repeated exposure, watch free energy drop
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np
from active_loop.lang_model import LangModel

target = 'mirro '
stream = target * 40   # it 'hears' mirro over and over
lm = LangModel(seed=1)
print(f'teaching it to say {target.strip()!r} by repeated exposure:')
print(f'  before any learning: surprise={lm.mean_surprise(stream):.2f} nats/char  sample={lm.generate(\"\", 6)!r}')
for ep in range(1, 13):
    lm.learn_stream(target * 8, epochs=1)
    if ep in (1, 3, 6, 9, 12):
        s = lm.mean_surprise(stream)
        sample = lm.generate('', 12)
        print(f'  after {ep:2d} exposures: surprise={s:.2f} nats/char  it says -> {sample!r}')