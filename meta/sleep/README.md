# Lab sleep cycles

This is advisory process memory, not scientific grading or hosted-model weight training.
Packets select up to eight committed experiments. Recorded findings/boundaries must be exact
excerpts from their experiment entry; every source includes an existing script and raw output.
The compiler always retains the recorded verdict and honest caveat. Generated hypotheses and
rehearsals require `origin: generated` and never become observed evidence.

```sh
uv run --python .venv python -m meta_monkey.sleep build --packet packet.json --out meta/sleep/run-id
uv run --python .venv python -m meta_monkey.sleep review --cycle meta/sleep/run-id --review review.json
uv run --python .venv python -m meta_monkey.sleep inspect --cycle meta/sleep/run-id
uv run --python .venv python loop/compose.py recursive-sleep --sleep-cycle meta/sleep/run-id
```

Packet shape:

```json
{"sources":[{"experiment":279,"script":"experiments/exp279_sleep_replay.py","output":"experiments/outputs/exp279.txt"}],"items":[{"kind":"boundary","source":279,"excerpt":"Exact text from that entry"},{"kind":"rehearsal","origin":"generated","source":279,"text":"A falsifiable next question"}]}
```

The example is a schema illustration; substitute real committed evidence. Review JSON contains
`content_hash` from the receipt, `reviewer`, boolean `accepted`, and fidelity `notes`. A fresh
reviewer checks source attribution, omitted negatives, caveats, and generated labels. The
file records that review; it does not authenticate reviewer identity. A stale, missing or
rejected review prevents prompt attachment. Source, packet or brief changes require a new
cycle and review. Cycles and review records cannot overwrite an existing cycle.

The complete brief has a conservative 1,500 UTF-8-byte ceiling; this is not an exact model
token count. Oversized packets fail without dropping evidence. Narrow the retrieval window
explicitly. The compiler extracts/carries knowledge; an independent prospective experiment
is needed before claiming it improves the lab's research.
