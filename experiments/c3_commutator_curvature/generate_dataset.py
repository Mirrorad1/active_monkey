"""Synthetic benchmark generator for the C3 experiment.

Each instance is a numbered-span prompt (12-40 spans) carrying scattered
relevant facts, distractors, redundant fragile commitments, answer-format
constraints, negations, numbers/units and cross-span dependencies.

Four task families (see README):

  A  Simple critical facts        -- each required commitment has ONE provider.
  B  Redundant fragile facts      -- commitments stated in TWO weak places.
  C  Scattered relation facts     -- chained 2-cover commitments + a shared
                                     bridge span (a node in two danger edges).
  D  Low-salience format consts   -- format/"do not" rules repeated weakly in
                                     short, low-density spans.

The hidden_* fields are ground truth for diagnostics ONLY; the selector never
reads them.

Usage:
    python experiments/c3_commutator_curvature/generate_dataset.py --n 1000 --seed 7
"""

from __future__ import annotations

import argparse
import os
import random

from common import render_prompt, write_dataset

# ---------------------------------------------------------------------------
# Content banks
# ---------------------------------------------------------------------------

# Redundant 2-cover commitments (Family B). Each has >=2 phrasings (providers)
# and optionally a trap distractor that flips the answer if the commitment is lost.
REDUNDANT_TOPICS = [
    dict(cid="unit_kg", value="KILOGRAMS",
         phrasings=["The final answer must be reported in kilograms, not pounds.",
                    "All pound-based intermediate results were rejected by the reviewer.",
                    "Per the spec, mass is always expressed in kilograms."],
         trap=("An older worksheet expressed the total in pounds.", "POUNDS")),
    dict(cid="currency_eur", value="EUR",
         phrasings=["Report all monetary totals in euros.",
                    "Dollar figures from the legacy sheet were discarded.",
                    "The accounting standard here mandates euro denomination."],
         trap=("The first quote was given in US dollars.", "USD")),
    dict(cid="round_2dp", value="TWO_DP",
         phrasings=["Round every number to exactly two decimal places.",
                    "Three-decimal outputs were flagged as non-compliant.",
                    "Precision is fixed at two decimals throughout."],
         trap=("A draft kept four decimals for safety.", "FOUR_DP")),
    dict(cid="tz_utc", value="UTC",
         phrasings=["All timestamps must be in UTC.",
                    "Local-time entries were explicitly rejected.",
                    "The log convention here is strictly UTC."],
         trap=("Some early notes used local time.", "LOCAL")),
    dict(cid="incl_tax", value="INCL_TAX",
         phrasings=["Prices must include tax.",
                    "Pre-tax figures were marked invalid.",
                    "The quoted total is always tax-inclusive."],
         trap=("One column listed pre-tax prices.", "EXCL_TAX")),
    dict(cid="sign_neg", value="SIGNED",
         phrasings=["Losses must be written as negative numbers.",
                    "Absolute-value entries were sent back for correction.",
                    "Keep the sign: debits are negative."],
         trap=("A summary showed losses as positive magnitudes.", "UNSIGNED")),
]

# Single-provider critical facts (Family A, and filler criticals elsewhere).
CRITICAL_TOPICS = [
    dict(cid="rate_7", value="R7", phrasing="The base interest rate is 7 percent."),
    dict(cid="deadline", value="D630", phrasing="The submission deadline is 2026-06-30."),
    dict(cid="cap_500", value="C500", phrasing="The hard budget cap is 500 units."),
    dict(cid="region_eu", value="REU", phrasing="This contract applies only in the EU region."),
    dict(cid="tier_gold", value="TG", phrasing="The customer is on the gold tier."),
    dict(cid="qty_42", value="Q42", phrasing="Exactly 42 pallets were shipped."),
    dict(cid="speed_88", value="S88", phrasing="The conveyor runs at 88 items per minute."),
    dict(cid="ph_5", value="PH5", phrasing="The required pH is 5.0, no higher."),
]

# Low-salience, low-density format constraints (Family D). Short = easy to delete.
FORMAT_TOPICS = [
    dict(cid="fmt_json", value="JSON",
         phrasings=["Answer in JSON.", "Output must be valid JSON.", "Return JSON only."]),
    dict(cid="fmt_noprose", value="NOPROSE",
         phrasings=["Do not include prose.", "No explanations.", "Skip commentary."]),
    dict(cid="fmt_key", value="KEYED",
         phrasings=['Use the key "result".', 'Wrap it under "result".', 'Top-level key: result.']),
    dict(cid="fmt_lower", value="LOWER",
         phrasings=["Keys must be lowercase.", "No capital letters in keys.", "Lowercase keys only."]),
    dict(cid="fmt_nonull", value="NONULL",
         phrasings=["Never emit null.", "Nulls are forbidden.", "Omit empty fields, no null."]),
]

DISTRACTOR_TEMPLATES = [
    "Shipment {a} departed from warehouse {b} on day {c}.",
    "Operator {a} logged {b} idle minutes during shift {c}.",
    "The cafeteria served {a} meals on floor {b} last {c}.",
    "Truck {a} carried {b} crates along route {c}.",
    "Sensor {a} reported {b} pings near gate {c}.",
    "Team {a} closed {b} tickets in sprint {c}.",
    "Building {a} used {b} kWh during week {c}.",
    "Vendor {a} delivered {b} pallets to dock {c}.",
    "Batch {a} cured for {b} hours in oven {c}.",
    "Drone {a} flew {b} laps over field {c}.",
]


def _distractor_text(rng: random.Random) -> str:
    t = rng.choice(DISTRACTOR_TEMPLATES)
    return t.format(a=rng.randint(1, 99), b=rng.randint(1, 99), c=rng.randint(1, 31))


# ---------------------------------------------------------------------------
# Instance assembly
# ---------------------------------------------------------------------------

class _Builder:
    """Accumulates span objects (pre-id) plus commitment specs referencing them."""

    def __init__(self):
        self.spans: list[dict] = []           # each dict gets an "id" after shuffle
        self.commitments: list[dict] = []      # {cid, value, providers:[span_obj,...]}
        self.fragile_groups: list[list] = []   # lists of span_obj that redundantly cover
        self.dangerous_pairs: list[tuple] = [] # (span_obj, span_obj)

    def span(self, text, role) -> dict:
        s = {"text": text, "role": role}
        self.spans.append(s)
        return s

    def commitment(self, cid, value, provider_spans, trap=None):
        self.commitments.append(dict(cid=cid, value=value, providers=list(provider_spans)))
        if trap:
            text, wrong = trap
            self.span(text, "distractor")  # plain trap; wired below
            self.spans[-1]["contradicts"] = cid
            self.spans[-1]["wrong_value"] = wrong

    def k_cover(self, cid, value, phrasings, role, k=2, trap=None):
        """A redundant commitment with k providers.

        For k=2 this is one dangerous PAIR (its 2 providers). For k>=3 the
        minimal failing set is the whole k-group; NO pair is dangerous on its
        own (deleting any 2 leaves a cover), so pairwise sigma is blind to it.
        We still record the k-group as the fragile group; dangerous_pairs is
        only populated for k==2 (it is the 2nd-order ground truth).
        """
        providers = [self.span(phrasings[i], role) for i in range(k)]
        self.commitments.append(dict(cid=cid, value=value, providers=list(providers)))
        self.fragile_groups.append(list(providers))
        if k == 2:
            self.dangerous_pairs.append((providers[0], providers[1]))
        if trap:
            text, wrong = trap
            t = self.span(text, "distractor")
            t["contradicts"] = cid
            t["wrong_value"] = wrong
        return providers

    def two_cover(self, cid, value, phrasings, role, trap=None):
        return self.k_cover(cid, value, phrasings, role, k=2, trap=trap)


def _finalize(b: _Builder, rng: random.Random, family: str, idx: int, n_target: int) -> dict:
    # pad with distractors up to n_target spans
    while len(b.spans) < n_target:
        b.span(_distractor_text(rng), "distractor")
    rng.shuffle(b.spans)
    for i, s in enumerate(b.spans):
        s["id"] = i
    sid = lambda obj: obj["id"]

    required = [dict(cid=c["cid"], value=c["value"],
                     provider_spans=sorted(sid(p) for p in c["providers"]))
                for c in b.commitments]
    fragile = [sorted(sid(p) for p in grp) for grp in b.fragile_groups]
    danger = [sorted((sid(a), sid(b_))) for a, b_ in b.dangerous_pairs]

    inst = {
        "id": f"{family}_{idx:05d}",
        "family": family,
        "question": _QUESTION[family],
        "gold_answer": "TASK_OK",
        "spans": [{"id": s["id"], "text": s["text"], "role": s["role"],
                   **({"contradicts": s["contradicts"], "wrong_value": s["wrong_value"]}
                      if "contradicts" in s else {})}
                  for s in sorted(b.spans, key=sid)],
        "original_prompt": render_prompt(sorted(b.spans, key=sid)),
        "hidden_required_commitments": required,
        "hidden_fragile_groups": fragile,
        "hidden_dangerous_pairs": danger,
    }
    return inst


_QUESTION = {
    "A": "Given the constraints above, produce the compliant result.",
    "B": "Report the result in the units/format the prompt commits to.",
    "C": "Decide whether container K can be exported, with full justification.",
    "D": "Return the result obeying every stated output-format constraint.",
}


def build_A(rng, idx, kcover=2):
    b = _Builder()
    k = rng.randint(3, 5)
    for topic in rng.sample(CRITICAL_TOPICS, k):
        p = b.span(topic["phrasing"], "fact")
        b.commitments.append(dict(cid=topic["cid"], value=topic["value"], providers=[p]))
    n = rng.randint(14, 26)
    return _finalize(b, rng, "A", idx, n)


def build_B(rng, idx, kcover=2):
    b = _Builder()
    k = rng.randint(3, 4)
    for topic in rng.sample(REDUNDANT_TOPICS, k):
        ph = list(topic["phrasings"]); rng.shuffle(ph)
        b.k_cover(topic["cid"], topic["value"], ph[:kcover], "redundant",
                  k=kcover, trap=topic.get("trap"))
    # a couple of single criticals for substance
    for topic in rng.sample(CRITICAL_TOPICS, rng.randint(1, 2)):
        p = b.span(topic["phrasing"], "fact")
        b.commitments.append(dict(cid=topic["cid"], value=topic["value"], providers=[p]))
    n = rng.randint(20, 34)
    return _finalize(b, rng, "B", idx, n)


def build_C(rng, idx, kcover=2):
    b = _Builder()
    # export chain: X (rule) AND Y (K blue) AND Z (K not archived)
    # (kcover affects only B/D; C's structure is intrinsically 2-cover + bridge)
    x1 = b.span("Only blue containers are cleared for export.", "bridge")
    x2 = b.span("Export clearance applies exclusively to blue-colored containers.", "bridge")
    b.commitments.append(dict(cid="rule_blue", value="BLUE_RULE", providers=[x1, x2]))
    b.fragile_groups.append([x1, x2]); b.dangerous_pairs.append((x1, x2))

    # shared bridge span: covers BOTH "K is blue" and "K not archived"
    bridge = b.span("Container K is blue and is not archived.", "bridge")
    y1 = b.span("The color log lists container K as blue.", "bridge")
    z1 = b.span("The archive registry has no entry for container K.", "bridge")
    b.commitments.append(dict(cid="k_blue", value="K_BLUE", providers=[bridge, y1]))
    b.fragile_groups.append([bridge, y1]); b.dangerous_pairs.append((bridge, y1))
    b.commitments.append(dict(cid="k_unarchived", value="K_UNARCH", providers=[bridge, z1]))
    b.fragile_groups.append([bridge, z1]); b.dangerous_pairs.append((bridge, z1))

    # negation trap distractor
    t = b.span("A stale flag once marked container K as archived.", "distractor")
    t["contradicts"] = "k_unarchived"; t["wrong_value"] = "NOT_EXPORTABLE"
    n = rng.randint(18, 32)
    return _finalize(b, rng, "C", idx, n)


def build_D(rng, idx, kcover=2):
    b = _Builder()
    k = rng.randint(3, 5)
    for topic in rng.sample(FORMAT_TOPICS, k):
        ph = list(topic["phrasings"]); rng.shuffle(ph)
        b.k_cover(topic["cid"], topic["value"], ph[:kcover], "format", k=kcover)
    # content criticals so the answer has substance beyond format
    for topic in rng.sample(CRITICAL_TOPICS, rng.randint(1, 2)):
        p = b.span(topic["phrasing"], "fact")
        b.commitments.append(dict(cid=topic["cid"], value=topic["value"], providers=[p]))
    n = rng.randint(18, 30)
    return _finalize(b, rng, "D", idx, n)


BUILDERS = {"A": build_A, "B": build_B, "C": build_C, "D": build_D}


def generate(n: int, seed: int, kcover: int = 2) -> list[dict]:
    rng = random.Random(seed)
    families = ["A", "B", "C", "D"]
    per = n // len(families)
    instances = []
    for fam in families:
        for i in range(per):
            instances.append(BUILDERS[fam](rng, i, kcover=kcover))
    rng.shuffle(instances)
    return instances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kcover", type=int, default=2,
                    help="redundancy depth for families B/D (2 = pairwise-detectable; "
                         ">=3 exposes the pairwise blind spot)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "data", "synthetic_c3.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    insts = generate(args.n, args.seed, kcover=args.kcover)
    write_dataset(out, insts)
    from collections import Counter
    fam = Counter(i["family"] for i in insts)
    print(f"wrote {len(insts)} instances -> {out}")
    print("by family:", dict(fam))


if __name__ == "__main__":
    main()
