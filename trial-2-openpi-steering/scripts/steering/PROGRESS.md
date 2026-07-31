# ATHENA steering on π0.5 / LIBERO — progress tracker

**Purpose:** survive lost SSH connections / SLURM allocation changes. Read this first to see
what is done, what is running, and what is next. Update it whenever a batch completes.

Last updated: 2026-07-20 (after paired bowl control completed)

---

## 0. Restart checklist (do this first after a dropped session)

1. `cd /scratch2/nalagand/openpi`
2. Check GPU + server: `nvidia-smi` ; `ss -ltn | grep 8001`
   - **The policy server does NOT survive a lost allocation.** It must be relaunched.
3. Relaunch server (takes ~60–90 s to load; idle footprint ~24.7 GB / 32 GB):
   ```bash
   nohup bash scripts/steering/serve_steered_policy.sh > /tmp/SLURM_$SLURM_JOB_ID/serve.log 2>&1 &
   ```
   Wait until `ss -ltn | grep 8001` shows LISTENING.
4. Verify the whole stack before spending GPU time (~2 min, 10 stages):
   ```bash
   PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero ATHENA_TASK=middle_bowl \
     examples/libero/.venv/bin/python scripts/steering/pipeline_healthcheck.py
   ```
5. **Write batch logs to `scripts/steering/results/`, not the scratchpad.** Scratchpad logs are
   destroyed when the SLURM allocation ends — that is how `bowl_paired.log` was lost. CSVs and
   `results/athena_feedback_logs/RESULTS.md` did survive, because they are written into the repo.

---

## 1. Where results live (these persist)

| Path | What |
|---|---|
| `results/athena_feedback_logs/RESULTS.md` | Human-readable summary block per batch — **start here** |
| `results/athena_feedback_batch_*.csv` | Raw per-rollout, red-mug away batch |
| `results/athena_feedback_middle_bowl_batch_*.csv` | Raw per-rollout, bowl away batch |
| `results/paired_athena_feedback_*.csv` | Paired same-noise-seed, red-mug |
| `results/paired_athena_middle_bowl_*.csv` | Paired same-noise-seed, bowl |

---

## 2. DONE

### Understanding / correction
- Read the ATHENA paper (`/scratch2/nalagand/2603.19676v1.pdf`, extracted via pure-Python zlib —
  no poppler/pypdf on this node, network sandboxed).
- **Key correction:** ATHENA steers **AWAY** from a control prompt.
  Eq. 5: `ε̃ = ε + γ(ε − ε̂)`, then norm-match to `‖ε‖` (Alg. 2 line 5). γ=4 in the paper.
  ATHENA-Feedback: observe wrong count `c`, build control prompt by replacing target `k` with `c`,
  steer away from it. The pre-existing `dual_instruction_denoise.run_dual_instruction_denoise`
  (`v_orig + γ(v_corrected − v_orig)`) is the **toward/attract mirror image**, NOT an ATHENA port.

### Implementation (all additive; existing regression-tested paths untouched)
- `correction_instruction.build_control_instruction()` — ATHENA control prompt (target phrase
  swapped for the observed-wrong object's display name).
- `dual_instruction_denoise.run_athena_feedback_denoise()` + `_match_norm()` — the away update.
- `serve_steered_policy.py` — `steer_mode` param: `"toward"` (default, backward compatible) /
  `"away"`.
- `run_athena_feedback_rollout.py` — red-mug-hardcoded away driver.
- `athena_tasks.py` — task registry (`ATHENA_TASK` env var).
- `run_athena_feedback_task.py` — generalized away driver (any registered task).
- `run_paired_athena_feedback_rollout.py` — paired control, red-mug.
- `run_paired_athena_task.py` — generalized paired control.
- `pipeline_healthcheck.py` — 10-stage end-to-end check.
- `test_athena_math.py` — CPU unit check of the away math.

### Validation passed
- `test_athena_math.py`: γ=0 exact no-op; norm matched; sign provably away (cos<0, distance to
  control grows).
- Live smoke on real weights: γ=0 away ≈ unsteered (1.5e-3, under the documented 3.9e-3
  jit-vs-eager noise floor); γ=4 gives a real bounded effect (0.039), actions stay in-scale.
- `noise_seed` determinism: same seed → bit-identical (0.0); different seed → differs (1.995).
- Control prompts confirmed grammatical live: `"put the white mug on the left plate"`,
  `"put the back black bowl on the plate"`.
- `pipeline_healthcheck.py` on `middle_bowl`: **10/10 PASS**.

### Experiments completed

| # | Case | Task | Run | Result |
|---|---|---|---|---|
| 1 | color/identity | red mug | away batch, 12 rollouts, γ=4 | 0/12 success, target grasped 0/12, moved >1e-4 only 1/12 |
| 1 | color/identity | red mug | **paired same-noise-seed**, 12 | unsteered 0/12, away 0/12, flip-to-success 0/12, changed 2/12 |
| 2 | spatial | middle bowl | away batch, 12 rollouts, γ=4 | 4/12 (33%) raw — **but** all 4 successes were rollouts where steering barely/never fired; 0/8 heavily-steered rollouts rescued |
| 2 | spatial | middle bowl | **paired same-noise-seed**, 12 | unsteered 5/12, away 5/12, **net +0**; 1 flip-to-success, 1 flip-to-failure, changed 8/12 |
| 2 | spatial | middle bowl | **noise floor** (both arms unsteered), 12 | outcome disagree **2/12** at states 5 & 10 — the same two states, same directions, as the "flips" above ⇒ those flips were noise |
| 3 | identity fixation | orange juice | **unsteered baseline**, 12 | **0/12 success**, target moved >1e-4 only 1/12; grasped **milk_1 11/12 (92%)**, tomato_sauce_1 1/12 |
| 3 | identity fixation | orange juice | **ungated away-steering** (γ=4, `STEER_DIVERGE_THRESH=999` ⇒ steered 80/80 replans), 12 | **0/12 success**; grasped **milk_1 11/12, tomato_sauce_1 1/12 — distribution IDENTICAL to baseline** |

### Case 3 is the strongest null in the project

Design note: this task has a 0% baseline (like red-mug), so binary success has no resolution. Instead
the metric is the **grasped-object distribution**, which had a very tight baseline (milk 11/12) and is
exactly the quantity the mechanism claims to move — the control prompt is literally "pick up the milk
and put it in the basket" and away-steering is supposed to repel from it. Gating was removed
(threshold 999) because the fidelity signal starts at −0.10 here for pure scene-geometry reasons
(the orange juice simply starts farther from the gripper than several distractors), making
feedback gating vacuous. That makes this run ATHENA-**Static**-like rather than Feedback — which
also discharges the "try Static later" TODO.

**Result: maximal repulsion from the milk prompt on every replan left the milk-grab rate exactly
unchanged (11/12 → 11/12).** This is stronger evidence than "0/12 success", which only shows the
checkpoint can't do the task. It shows the steering does not perturb the object choice *at all* —
the specific thing it exists to do.

**Verdict so far:** ATHENA-faithful away-steering at the *language/prompt* level produces no causal
improvement on either failure case tested. Case 1 is a flat null (target literally never touched).
Case 2 is a wash (one win, one loss).

---

## 3. ROOT-CAUSED (2026-07-20, after reconnect): the renderer is nondeterministic

The determinism leak below is **diagnosed**. Scripts: `diagnose_paired_determinism.py`,
`diagnose_render_repeat.py`. Logs: `logs/determinism_diag*.log`, `logs/render_repeat.log`.

**Finding.** Two identical unsteered arms (same init state, same per-replan `noise_seed`) have
**bit-identical physics** at the first replan — all 7 joint angles `0.0`, all three bowl positions
`0.0`, eef state `0.0` — but the **rendered images differ by up to 137 grey levels across ~6% of
pixels**. That image difference feeds the policy, so actions differ immediately (1.3e-2 at replan 0)
and the trajectories drift apart from there (joint diff 0.0 → 1.2e-3 by replan 1).

**Isolated to the renderer alone.** `diagnose_render_repeat.py` does pure sim + render with *no
policy calls at all*: four identical `reset() → set_init_state() → 10 dummy steps → render` cycles
produce four different images (max 99–181 grey levels, 5.7–8.4% of pixels each time). Not a
cold-start artifact — every reset renders differently, reproducibly (the run-0/run-1 hashes match
the ones seen in the paired diagnostic exactly).

**What this invalidates.**
- The `noise_seed` mechanism is fine — it controls the *denoising* noise and was verified
  bit-identical. It never controlled the *observation*, so "same-noise-seed" never actually held
  the policy's input fixed. The pairing was weaker than its name claims.
- **Bowl paired control:** the 1 flip-to-success and 1 flip-to-failure are attributable to render
  nondeterminism, not steering. The net-zero headline (5 vs 5) is unaffected.
- **Red-mug paired control:** unaffected in substance — 0/12 both arms with the target literally
  never touched leaves no variance for the leak to explain.
- **Raw batches (red-mug 0/12, bowl 4/12):** unaffected — they never claimed paired observations.

**Options for a real control (undecided — needs a call):**
1. Fix the renderer (investigate MuJoCo/robosuite shadows, multisampling, GL context reuse). Best
   if cheap, unknown effort.
2. Drop pixel-level pairing; run larger n and use a proper paired statistical test (McNemar) or
   bootstrap over init states. Pragmatic, definitely works, costs GPU hours.
3. Keep pairing only for tasks where the effect is large enough to clear the render noise floor —
   quantify that floor first (how often do two identical unsteered arms disagree on outcome?).

**Caveat found while reviewing the paired script (2026-07-20):** `run_paired_athena_task.py` defines
`changed` as *different grasped object OR |target_disp delta| > 1e-3*. That second clause trips on
millimetre wobble, so the bowl control's `changed=8/12` is almost certainly inflated and should not
be quoted as "behaviour changed". The interpretable numbers are the two **outcome flips** (states 5
and 10). Consider tightening or dropping that clause.

### NOISE FLOOR MEASURED — the bowl "flips" were noise (`run_render_noise_floor.py`)

Both arms UNSTEERED, identical init state + per-replan `noise_seed`, bowl task, 12 trials
(`logs/noise_floor.log`, `results/render_noise_floor_middle_bowl_*.csv`).

**Outcome disagreement: 2/12 (17%) — at init states 5 and 10.**

Those are *exactly* the two states that "flipped" in the bowl paired control, in the same
directions:

| state | paired control (with steering) | noise floor (NO steering) |
|---|---|---|
| 5 | SUCCESS -> FAILURE ("steering hurt") | SUCCESS -> FAILURE |
| 10 | FAILURE -> SUCCESS ("steering helped") | FAILURE -> SUCCESS |

**Conclusion:** both flips are reproduced with the intervention entirely removed. States 5 and 10
are bistable (near a decision boundary). Steering explained nothing; the bowl net-zero is a genuine
characterised zero, not a coincidental tie. The other 10 states are perfectly stable on both
outcome and grasped object, so the pipeline is not broadly noisy -- it is two specific states.

**Detection threshold this establishes:** on 12 trials an effect must exceed ~2 flips to mean
anything. Design future batches accordingly: either more init states, or exclude/flag the known
bistable states, or use McNemar with the discordant-pair count.

---

## 3b. Original issue writeup (now explained by section 3)

**The paired control has a determinism leak.** In `paired_athena_middle_bowl_20260720_134145.csv`,
init states 3, 8, 9, 10 have `away_n_steered = 0` — steering never fired — yet all four are marked
`changed=True`, and **state 10 flipped FAILURE→SUCCESS with zero steering applied**.

With identical `base_seed` and identical per-replan `noise_seed`, and no steering interventions,
both arms should be identical. They are not. Therefore:
- the single "steering helped" datapoint is **not attributable to steering**;
- the `changed=8/12` figure is inflated by the same leak.

This does not change the headline (5 vs 5 is a null either way, and the raw-batch anti-correlation
points the same direction), but the control is weaker than designed.

**Suspects (untested):** `env.reset()` not restoring identical sim state between the two arms; or a
server-side RNG path that `noise_seed` does not cover in the `steer=False` branch.

**Proposed diagnostic:** run one init state through both arms with steering force-disabled and
identical seeds, diff the action chunks / eef trajectory step-by-step, find where they part.

---

## 3c. LANGUAGE SENSITIVITY PROBE — my spatial hypothesis was WRONG (2026-07-20)

`probe_language_sensitivity.py`: same observation, same `noise_seed`, only the prompt changes;
rel_L2 = ||A1 - A2|| / ||A1||. Identical-prompt control = **0.0000 exactly** on all three tasks, so
the probe is valid.

| task | contrast kind | rel_L2 |
|---|---|---|
| red_mug | ATTRIBUTE (red vs white/yellow mug) | 0.0088 – 0.0094 |
| red_mug | SPATIAL (left vs right plate) | 0.0164 |
| middle_bowl | SPATIAL (front/back, middle/front, left/right) | 0.0142 – 0.0236 |
| all | MOTION (up/down, reach left/right) | 0.0088 – 0.0216 |
| **orange_juice** | **IDENTITY, noun-level (orange juice vs milk / ketchup)** | **0.2464 – 0.2474** |

**Two conclusions, both against my prior reasoning:**

1. **Spatial language is NOT a special lever.** Spatial contrasts (~0.018) sit barely above attribute
   and motion contrasts (~0.010) — all in the same small range. The plan to re-phrase corrections
   spatially is not supported; do not spend rollouts on it on this evidence.
   Also: my claim that the old `test_real_weights.py` cosine of 0.815 proved motion language is a
   "strong lever" was an over-read. Cosine measured the *direction* coherence of the steering, not
   its *magnitude*. Measured by magnitude here, motion language is as weak as colour language.

2. **Noun-level object identity IS read strongly — ~25x more than colour adjectives.** Swapping
   "orange juice" -> "milk" moves the actions by 25% rel_L2. Swapping "red mug" -> "white mug" moves
   them by 0.9%. So the policy attends to *which object noun* is named, and almost ignores an
   *adjective* distinguishing two instances of the same noun.

**The sharpest finding in the project, from combining this with the case-3 result:** on the
orange-juice task the language demonstrably modulates the action distribution by 25% — and yet
away-steering left the grasped object completely unchanged (milk 11/12 -> 11/12). So it is *not*
true that "the model ignores the prompt". The model reads the prompt, produces materially different
actions, and still converges on the same visually salient object. That localises the bottleneck
precisely: **language modulates the trajectory but does not control grasp-target selection.**

---

## 4. TODO — in priority order

1. ~~Determinism diff~~ **DONE — root-caused, see section 3.** Follow-up: measure the
   render-induced noise floor (N identical unsteered pairs → outcome disagreement rate), then pick
   one of the three control options in section 3.
2. **Case 3 — identity fixation.** ✅ **REGISTERED** as `ATHENA_TASK=orange_juice`
   (`LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket`, target `orange_juice_1`,
   6 distractors, `basket_1` deliberately excluded as it is the destination). Control prompts
   verified grammatical ("pick up the milk and put it in the basket"). All existing drivers pick it
   up via the env var; no code changes needed.
   **Still to do, in order:** (a) pipeline healthcheck on it; (b) **unsteered baseline first** --
   we do not know the success rate, and case 1 taught us a ~0% baseline is the wrong hill for a
   method that amplifies latent competence; (c) only then the away batch + paired control.
   ~~Not set up yet.~~ Needs a task where the arm fixates on one object
   regardless of instruction (candidate: `LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket`,
   target `orange_juice_1`, distractors `alphabet_soup_1`, `cream_cheese_1`, `tomato_sauce_1`,
   `ketchup_1`, `milk_1`, `butter_1`). Work: add one entry to `athena_tasks.py`, then run
   `run_athena_feedback_task.py` + `run_paired_athena_task.py` — no new code needed.
3. **ATHENA-Static** — *partially discharged*: the case-3 ungated run (steering on every replan, no
   feedback gate) is Static's regime and produced an exact null. Still untried in its literal form
   (control prompt = attribute stripped) on red-mug/bowl: control prompt = attribute stripped
   ("pick up the mug" / "pick up the black bowl"), steer away from step 0, no feedback gating.
4. **γ sweep** (γ=8, 10; wider window) on red-mug — expected null, but closes the "you didn't tune
   it" objection cheaply (~20 min each).
5. **Spatial-vocabulary steering** — the most promising untried idea, and it stays inside the
   user's "language, in action space" constraint. Rationale: `test_real_weights.py` measured
   cosine **0.815** for "move up" vs "move down" (motion language moves the action velocity a lot),
   versus ~0 for the color contrast. So phrase corrections spatially/directionally instead of by
   object attribute:
   - cheap diagnostic first: cosine of `v("...on the left plate")` vs `v("...on the right plate")`
     on the same image. Large → the lever exists; ~0 → don't spend rollouts.
   - then `build_spatial_control_instruction()` from sim positions, and paired batches.

---

## 5. Constraints to remember

- User wants to stay in **language / action space** — activation-level steering (COAST-style) was
  discussed and explicitly deprioritized in favor of language methods.
- Two venvs, never merged: `openpi/.venv` (server, py3.11) and `examples/libero/.venv` (sim, py3.8).
  LIBERO scripts need `PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero`.
- CARC account `biyik_1165`. Server port 8001 (`STEERED_POLICY_PORT`).
- Checkpoint from `/scratch1`, never `/home1` (NFS restore is ~16 min vs ~30 s).
