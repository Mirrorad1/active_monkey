# Recovered Experiment Scripts — Exp 1–40

These scripts were recovered on 2026-06-09 from the original session transcripts after the
correction entry in `EXPERIMENTS.md` (dated 2026-06-09) noted that the scripts for Exp 1–40
were never committed (they were run as inline heredocs or temp files in Claude Code sessions).

## Source transcripts

Primary source (Exp 1–40, all confirmed):
- `session_id`: `72317201-ec87-49eb-88d2-beffa86bd7ec`
- Transcript file: `/Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl`

Exp 35 (`converse_demo.py`) was also committed to the repo root during the original session
and still runs — it is the only script that was committed at the time. The recovered copy here
is from the transcript `Write` tool call for completeness/provenance.

## File naming

- `expNN_<slug>.py` — Python script body exactly as found in the transcript (no code changes),
  with ONLY a comment header block added at the top (clearly marked "added at recovery time").
- `outputs/expNN.txt` — raw recorded output from the matching `tool_result` in the transcript.
- `outputs/expNN_rerun_2026-06-09.txt` — re-run output (all 40, produced 2026-06-09).

## Extraction notes

- **Exp 1–12, 14–15, 17–31, 33–34, 37–40**: scripts ran as inline `python -c "..."` heredocs in
  a Bash tool call; extracted from `input.command`.
- **Exp 13**: written to `/tmp/exp13.py` via heredoc, then patched (batch-axis fix applied by
  sed in a later tool call), then re-run. The recovered script reflects the **fixed** version
  (the sed substitution `np.eye(T)[tk])[0]` → `np.eye(T)[tk])` applied, matching the final
  run). Two earlier failed attempts are noted: first run failed due to missing `PYTHONPATH`;
  second failed due to a batch-axis shape bug in the teacher-forced prior.
- **Exp 16**: written to `exp16.py` at repo root via a `Write` tool call
  (`tool_use_id: toolu_01XtynLHq3srAe1K2XumM7MV`), then run, then `rm -f exp16.py` at commit.
- **Exp 32, 35–36**: Exp 35 = `Write` tool call for `converse_demo.py`; Exp 36 = heredoc
  `cat > exp36.py <<'PY' ... PY` in a Bash call, then run in background, then removed.

## Full Re-verification (all 40 scripts) — 2026-06-09

Re-run command: `cd /Users/mirro/Projects/active-loop && PYTHONPATH=. .venv/bin/python experiments/recovered/expNN_<slug>.py`
Re-run date: 2026-06-09. Output files: `outputs/expNN_rerun_2026-06-09.txt` (all 40).

### Transcription artifact (fixed 2026-06-09)

Three scripts (Exp 3, 7, 34) had a recovery transcription artifact: the originals ran via
`python -c "..."` heredoc where the shell consumed backslash-escapes, so `\"` reached Python as
`"`. The recovered `.py` files preserved the literal `\"`, which Python 3.12 rejects at parse
time with `SyntaxError: unexpected character after line continuation character`. Fix applied
2026-06-09: replaced every literal `\"` with `"` in the f-string expressions. No other code
was altered. After the fix, all 3 re-run outputs match the originals exactly. A provenance note
(`#   transcription fix: 2026-06-09 — restored shell-consumed \" escapes to " (python -c artifact)`)
was added to each file's header comment block.

### MISMATCH note (EXPERIMENTS.md logged claim vs original output)

Exp 01: EXPERIMENTS.md claims "held-out 4.81 → 4.00 bits/char". The original recorded output
(`outputs/exp01.txt`) and the re-run both show "4.007 → 3.424 bits/char". The logged narrative
numbers (likely from a mental summary or a different iteration count) do not match the actual
output. The re-run reproduces the original output exactly.

### Verification table

| Exp | Script | Orig output? | Re-run verdict | Note |
|-----|--------|-------------|----------------|------|
| 1 | exp01_aif_hmm_baseline.py | yes | MATCH | Re-run = original output exactly; EXPERIMENTS.md narrative claims "4.81→4.00" but both outputs show "4.007→3.424" (pre-existing logged-claim discrepancy, not a re-run issue) |
| 2 | exp02_bandit_positive_feedback.py | yes | MATCH | |
| 3 | exp03_teach_mirro.py | yes | MATCH | Transcription artifact fixed (2026-06-09): `\"` → `"` in f-string expression; re-run output matches original exactly (surprise 3.38→1.61, all samples identical); JAX UserWarning is incidental stderr, not script output |
| 4 | exp04_memory_order_qa.py | yes | MATCH | |
| 5 | exp05_context_state_order.py | yes | MATCH | |
| 6 | exp06_bigram_greedy.py | yes | MATCH | |
| 7 | exp07_ngram_context_depth.py | yes | MATCH | Transcription artifact fixed (2026-06-09): `\"` → `"` in f-string expressions; re-run output matches original exactly (n=2 'iro miro mi', n=3 'irro mirro ', Q->A n=3 'mirro. ') |
| 8 | exp08_aif_pair_state.py | yes | MATCH | |
| 9 | exp09_longrange_binding_flat.py | yes | MATCH | |
| 10 | exp10_topic_conditioned_binding.py | yes | MATCH | |
| 11 | exp11_twofactor_scaffold.py | yes | MATCH | |
| 12 | exp12_unsupervised_topic_fails.py | yes | MATCH | |
| 13 | exp13_semisupervised_topic.py | yes | MATCH | |
| 14 | exp14_valence_grounding.py | yes | MATCH | |
| 15 | exp15_affective_loop_efe.py | yes | MATCH | |
| 16 | exp16_warmstart_meanfield.py | yes | MATCH | |
| 17 | exp17_embodied_gridworld.py | yes | MATCH | |
| 18 | exp18_place_path_integration.py | yes | MATCH | |
| 19 | exp19_learn_sensory_map_aliased.py | yes | MATCH | |
| 20 | exp20_place_fields_continuous.py | yes | MATCH | |
| 21 | exp21_place_fields_2d.py | yes | MATCH | |
| 22 | exp22_fuse_place_valence.py | yes | MATCH | |
| 23 | exp23_navigation_horizon.py | yes | MATCH | |
| 24 | exp24_object_place_binding.py | yes | MATCH | |
| 25 | exp25_recall_navigate.py | yes | MATCH | |
| 26 | exp26_proto_opinion.py | yes | MATCH | |
| 27 | exp27_opinion_drives_behavior.py | yes | MATCH | |
| 28 | exp28_ask_what_it_thinks.py | yes | MATCH | |
| 29 | exp29_compositional_query.py | yes | MATCH | |
| 30 | exp30_scalable_planning_vi.py | yes | MATCH | |
| 31 | exp31_learn_a_and_b_fails.py | yes | MATCH | |
| 32 | exp32_hierarchy_room_concept.py | yes | MATCH | |
| 33 | exp33_hierarchical_planning.py | yes | MATCH | |
| 34 | exp34_language_bridge.py | yes | MATCH | Transcription artifact fixed (2026-06-09): `\"` → `"` in f-string expressions; re-run output matches original exactly (word map correct, A likes red / B likes green, surprise values 1.6/0.0 bits identical) |
| 35 | exp35_converse_demo.py | yes | MATCH | |
| 36 | exp36_scale_6x6.py | yes | MATCH | |
| 37 | exp37_scale_6concepts.py | yes | MATCH | |
| 38 | exp38_integrated_stack.py | yes | MATCH | |
| 39 | exp39_noise_robustness.py | yes | MATCH | |
| 40 | exp40_opinion_revisable.py | yes | MATCH | |

**Summary: 40/40 scripts recovered; 40/40 original outputs recovered; 40 MATCH, 0 QUALITATIVE-MATCH, 0 MISMATCH, 0 FAIL.**

All 40 scripts now reproduce. Exp 3, 7, and 34 had a transcription artifact (see below) fixed
on 2026-06-09; after the fix their re-run outputs match the originals exactly.

Additionally, EXPERIMENTS.md's narrative for Exp 1 cites "4.81 → 4.00 bits/char" but both the
original transcript output and the 2026-06-09 re-run show "4.007 → 3.424 bits/char". This is a
pre-existing logged-claim inaccuracy in the text narrative (not introduced by re-verification).
