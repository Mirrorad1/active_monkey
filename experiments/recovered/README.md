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
- `outputs/expNN_rerun_2026-06-09.txt` — re-run output (for Exp 17, 20, 21 only).

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

## Re-verification (Exp 17, 20, 21)

Re-run command: `cd /Users/mirro/Projects/active-loop && PYTHONPATH=. uv run --python .venv python experiments/recovered/expNN_<slug>.py`
Run date: 2026-06-09.

| Exp | Key logged number | Re-run result | Match? |
|-----|------------------|---------------|--------|
| 17 | transition error 0.003 | error left=0.003 right=0.003; match=True | **MATCH** |
| 20 | tuning == true cmap, 0.00 bits | structural-match=True, 0.00 bits, ok=True | **MATCH** |
| 21 | exact colormap, 0.00 bits | exact-match=True, 0.00 bits, ok=True | **MATCH** |

All three re-runs reproduced the logged key numbers exactly.

## Recovery table

| Exp | Slug | Script recovered? | Original output recovered? | Re-verified? |
|-----|------|------------------|---------------------------|--------------|
| 1 | aif_hmm_baseline | yes | yes | no |
| 2 | bandit_positive_feedback | yes | yes | no |
| 3 | teach_mirro | yes | yes | no |
| 4 | memory_order_qa | yes | yes | no |
| 5 | context_state_order | yes | yes | no |
| 6 | bigram_greedy | yes | yes | no |
| 7 | ngram_context_depth | yes | yes | no |
| 8 | aif_pair_state | yes | yes | no |
| 9 | longrange_binding_flat | yes | yes | no |
| 10 | topic_conditioned_binding | yes | yes | no |
| 11 | twofactor_scaffold | yes | yes | no |
| 12 | unsupervised_topic_fails | yes | yes | no |
| 13 | semisupervised_topic | yes (fixed version) | yes | no |
| 14 | valence_grounding | yes (refined run) | yes | no |
| 15 | affective_loop_efe | yes | yes | no |
| 16 | warmstart_meanfield | yes | yes | no |
| 17 | embodied_gridworld | yes | yes | yes: MATCH |
| 18 | place_path_integration | yes | yes | no |
| 19 | learn_sensory_map_aliased | yes | yes | no |
| 20 | place_fields_continuous | yes | yes | yes: MATCH |
| 21 | place_fields_2d | yes | yes | yes: MATCH |
| 22 | fuse_place_valence | yes (re-run) | yes | no |
| 23 | navigation_horizon | yes | yes | no |
| 24 | object_place_binding | yes | yes | no |
| 25 | recall_navigate | yes | yes | no |
| 26 | proto_opinion | yes | yes | no |
| 27 | opinion_drives_behavior | yes | yes | no |
| 28 | ask_what_it_thinks | yes | yes | no |
| 29 | compositional_query | yes | yes | no |
| 30 | scalable_planning_vi | yes | yes | no |
| 31 | learn_a_and_b_fails | yes | yes | no |
| 32 | hierarchy_room_concept | yes | yes | no |
| 33 | hierarchical_planning | yes | yes | no |
| 34 | language_bridge | yes | yes | no |
| 35 | converse_demo | yes (Write tool call) | yes | no (converse_demo.py in repo root still runs) |
| 36 | scale_6x6 | yes (heredoc) | yes | no |
| 37 | scale_6concepts | yes | yes | no |
| 38 | integrated_stack | yes | yes | no |
| 39 | noise_robustness | yes | yes | no |
| 40 | opinion_revisable | yes | yes | no |

**Summary: 40 of 40 scripts recovered; 40 of 40 original outputs recovered; 3 of 40 re-verified (all MATCH).**

Scripts not re-verified (37/40) are unverified recovered artifacts — they represent the script bodies
found in the transcript, and the logged outputs are the original recorded results. They have not been
re-run; API drift or other changes may affect reproducibility.
