# ATHENA → Robotics: Implementation Progress

**Read this file first when picking the project back up.**
Source spec: `/scratch1/nalagand/Executive Summary (4).pdf` (12 pp).
Goal: adapt ATHENA's test-time steering (text-to-image diffusion) into an
inference-time **predict → verify → re-steer** loop for π0.5 on LIBERO-90.

Last updated: **2026-07-30**

---

## How to resume in one command

```bash
cd /scratch1/nalagand/athena_robotics
source /scratch1/nalagand/openpi/examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:/scratch1/nalagand/openpi/third_party/libero:/scratch1/nalagand/athena_robotics"
export MUJOCO_GL=egl

python tests/test_logic.py                      # 34 offline checks, no GPU needed
python tests/test_perception_live.py            # 12-check live-sim smoke test
python scripts/run_experiment.py --preset adaptive --task-subset pilot --dry-run
python scripts/analyze.py                       # whatever results exist so far

# Characterisation (what the policy actually DOES — needs a live server)
python scripts/run_characterization.py --tasks 67 68 --num-trials 5 \
    --run-id my_run --port 8000 --save-video
python scripts/analyze_characterization.py --path results/my_run/episodes.jsonl

# Gibberish screen: does this scene use the instruction at all?
bash scripts/run_gibberish_probe.sh
```

**Screen any new scene with the gibberish probe before spending GPU on
steering.** If a nonsense prompt scores as well as the real instruction, the
scene cannot measure steering — see "THE DISSOCIATION". This is a 5-episode
check that would have saved the two pilot nulls.

**Run `tests/test_logic.py` after any edit to steering/verifier/metrics.** It is
fast, needs no GPU, and carries regression tests for two prompt-construction
bugs that are easy to reintroduce (see Key design decisions 6 and 7).

All runs are **resumable**: `results/<run_id>/episodes.jsonl` is append-only and
already-recorded `(task_id, episode_idx)` pairs are skipped, so re-submitting a
preempted job continues where it stopped. Nothing needs manual bookkeeping.

---

## Status board

| # | Stage | Status | Evidence / next action |
|---|-------|--------|------------------------|
| 1 | Read + decompose exec summary | **DONE** | All 12 pp parsed; §3 mapping table drives module layout |
| 2 | Survey existing infra | **DONE** | `openpi` + LIBERO already present; prior libero_90 job **failed** (see Known issues) |
| 3 | BDDL task parser (`taskspec.py`) | **DONE** | Parses 90/90; found **32/90 confusable tasks** |
| 4 | Perception layer (`perception.py`) | **DONE** | Oracle + noisy detector; validated live (12/12 checks) |
| 5 | State verifier (`verifier.py`) | **DONE** | Fires `WRONG_OBJECT` on distractor reach; no false alarm in free space |
| 6 | Steering policies (`steering.py`) | **DONE** | none / static / adaptive / dual-plan |
| 7 | Metrics + resume (`metrics.py`) | **DONE** | All §5 metrics; Wilson CIs |
| 8 | Closed-loop runner (`runner.py`) | **DONE** | Imports clean; **not yet run against a live policy server** |
| 9 | SLURM launcher | **DONE** | `scripts/run_athena.sbatch`; `set -u` bug fixed |
| 10 | **Pilot run (8 tasks × 5 eps)** | **DONE — both arms 40/40** | baseline 0.375, adaptive 0.325. Ceiling safety check passed; predicted null observed. See "FINAL pilot result" |
| 10b | **Retry-budget fix (option 3)** | **APPLIED, tests green (47/47)** | `steering.py` `should_intervene` + ladder clamp. **First live run was 10d** (confounded with dual-plan): steer events 2.40 → 12.28 as predicted, and it surfaced the verifier re-fire bug in Known issues. Needs an `adaptive_persist` arm to isolate |
| 10c | **Prompt-stickiness correction** | **DONE** | `runner.py:210/263` — corrective prompt never reverts. Invalidates the "~95% unsteered" reading; adaptive null is closer to genuine |
| 10d | **`dual_select` pilot** | **DONE — 40/40** | 0.325 success, identical to adaptive on every rate metric at 5× the interventions. Ceiling 9/10 (first ever ceiling loss); `P(s\|ok)` broke 1.000 → 0.889. See "dual_select pilot result" |
| 16 | **Attractor characterisation** (`attractor_baseline`) | **DIED at 30/320** — resumable | Allocation killed (SIGKILL, shell exited) 2026-07-21 01:59 after tasks 0-2 only. Results intact; resume-by-skip continues from ep 31. Superseded in priority by 17 |
| 17 | **LIVING_5 routine characterisation** (`living5_char`) | **DONE — 20/20, 2026-07-30** | Full-episode event logging, 4 tasks × 5 eps, no steering. **Fixed instruction-independent two-phase routine confirmed.** See "CONFIRMED: a fixed motor routine, not object selection" |
| 18 | **Gibberish-prompt probe** (`gibberish_mugs`, `language_oj`, `gibberish_oj`) | **DONE — 20/20, 2026-07-30** | Nonsense instruction. Mugs **10/10 success on gibberish** (no effect); orange juice **5/5 → 1/5** (large effect). **Dissociation: the routine is scene-specific, not model-wide.** See "THE DISSOCIATION" |
| 19 | **Steering on a scene that IS language-driven** | **NOT STARTED — now the main line** | libero_object is the first scene where a selection process demonstrably exists to steer. See "What to do next" |
| 11 | Baseline vs adaptive on confusable subset | **DEPRIORITISED** | Superseded by 16 — a third steering arm measures the same wall. Revisit only if 16 shows the instruction does move grasp choice |
| 12 | Full LIBERO-90 evaluation | NOT STARTED | after 11 |
| 13 | Ablations (vision / early-late / retries / blending) | NOT STARTED | presets already defined |
| 14 | Learned RGB detector (replace oracle) | NOT STARTED | interface ready; see Open questions |
| 15 | Analysis writeup + figures | NOT STARTED | |

---

## What exists

```
athena_robotics/
├── PROGRESS.md              <- you are here
├── athena/
│   ├── taskspec.py          BDDL parsing + confusion analysis
│   ├── perception.py        Detector interface, OracleDetector, NoisyDetector
│   ├── verifier.py          StateVerifier, Verdict, VerdictKind
│   ├── steering.py          none / static / adaptive / dual-plan re-steering
│   ├── metrics.py           EpisodeRecord, MetricsWriter (resume), summarize
│   ├── config.py            ExperimentConfig + 12 named PRESETS
│   ├── eventlog.py          InteractionTracker: FULL per-episode grasp/
│   │                        release/placement sequence for every object
│   └── runner.py            closed-loop predict→verify→steer→execute
├── scripts/
│   ├── run_experiment.py    CLI (supports --dry-run without a server)
│   ├── analyze.py           aggregate metric tables + Wilson CIs
│   ├── run_characterization.py  baseline-only, does NOT stop at success,
│   │                        --prompt-override, --suite, --save-video
│   ├── analyze_characterization.py  sections A/B/C + stop-condition
│   ├── run_gibberish_probe.sh   the 3 nonsense-prompt runs
│   └── run_athena.sbatch    GPU launcher, multi-preset, resumable
├── tests/test_perception_live.py   live-sim smoke test
└── results/<run_id>/episodes.jsonl  (+ videos/ when --save-video)
```

**Use `run_characterization.py`, not `run_experiment.py`, for any question about
*what the policy does*.** The main runner records only `first_grasped` and
`success` and stops the episode at success — two scalars that cannot see a
multi-step routine. Three separate misreadings of task 68 trace to that.

---

## Key design decisions (and why they deviate from the doc)

These are deliberate. Re-read before changing anything.

1. **π0.5 emits no subtask text in this stack.** The doc's pseudocode assumes
   `subtask, action = π0.5_step(...)` then `parse_target_from(subtask)`. Verified
   in `src/openpi/policies/libero_policy.py:100` — the server returns
   `{"actions": ...}` only. **So we infer intent from behaviour instead**: which
   object the gripper is closing on / has grasped. This is the faithful
   adaptation, and is arguably more robust than trusting self-reported text.

2. **Oracle detector first, learned detector later.** The doc lists "oracle
   detector" as an explicit condition (§5, baseline 10) to isolate algorithm
   quality from perception quality. Starting there means the steering logic can
   be evaluated *now*, and `NoisyDetector` sweeps degradation parametrically —
   which answers ablation §5.19 more cleanly than one particular detector would.
   A learned detector plugs in behind the same `Detector` interface.

3. **Ground truth for "which object is the target" comes from BDDL
   `:obj_of_interest`**, not from parsing the instruction. This is exact and
   avoids a whole class of NLP bugs. It *does* mean the verifier has privileged
   information — noted as a limitation for the writeup.

4. **Dual-plan defaults to `select`, not `blend`.** ATHENA mixes ε in a
   continuous latent space. Convex-combining two action chunks can produce a
   trajectory that is neither (e.g. averaging "reach left" and "reach right"
   drives the gripper straight into the table). Both are implemented;
   `blend` is available as an ablation.

5. **Verification runs every `verify_every=5` control steps**, aligned to the
   action-chunk boundary (`replan_steps=5`), so a re-steer discards exactly one
   un-executed chunk rather than interrupting mid-chunk.

6. **Negation is only emitted when the distractor reads differently from the
   target.** `akita_black_bowl_1` and `_2` both render as "black bowl", so
   "do not pick up the black bowl, pick up the black bowl" is incoherent — and
   that is precisely the hardest case. For identical instances we assert the
   spatial qualifier instead. Locked by a regression test.

7. **The spatial qualifier is only used when position is the *only*
   discriminator.** In "put the red mug on the left plate", "left" modifies the
   *plate*; attaching it to the mug ("the red coffee mug at the left") is wrong.
   `_needs_spatial_qualifier` gates this. Locked by a regression test.

---

## Empirical findings so far

- **32/90 LIBERO-90 tasks have same-token distractors** (`confusion_score > 0`);
  28 have literal duplicate object categories. The method can only help here —
  aggregate LIBERO-90 numbers will dilute the effect, so **always report the
  confusable subset separately** (`analyze.py --confusable-only`).
- Hardest cases (score 7): the 4 mug tasks — red / white / yellow-and-white mugs
  plus two plates. Next (score 5): three identical `akita_black_bowl` instances
  disambiguated only by "at the front / in the middle / at the back", and the
  three-book shelf tasks.
- Live-sim validation on task 13 ("put the black bowl at the front on the
  plate"): verifier flagged `WRONG_OBJECT` at step 35 while reaching for
  `akita_black_bowl_2`, and stayed silent in free space. No false alarms.

---

## Interpretation caveat: π0.5 sometimes self-recovers

Observed in the pilot (baseline, task 13 ep1): the policy grasped the **wrong**
bowl and the episode still **succeeded**, in 206 steps versus 87 for a clean
success. It appears to have put the wrong bowl down and gone back for the right
one, unprompted.

Consequences for analysis:

- `grasped_correct` (first grasp) measures **initial intent**; `success`
  measures **final outcome**. They are different quantities and will not move
  together 1:1.
- Steering fires on the first wrong grasp, so some interventions will "fix"
  episodes the policy would have recovered from anyway. **Object-accuracy gains
  therefore overstate the expected success-rate gain.**
- Report both, and treat success rate as the honest headline number.
- Worth adding later: a `self_recovered` flag (wrong first grasp + success) to
  size this effect directly, and per-episode grasp *sequences* rather than only
  the first.

Verified separately that LIBERO's goal predicate names the specific instance
(`(on akita_black_bowl_3 plate_1)`), so these are genuine wrong-object failures,
not an artifact of how we resolve the target.

`self_recovery_rate` is **derived**, not stored — a pure function of
`grasped_correct` and `success` (`metrics.is_self_recovered`). This was
deliberate: it applies retroactively to records written before the metric
existed, avoiding a schema split between the baseline run (already in flight
when the metric was added) and later runs.

### First measurement — the premise holds (baseline, n=23, preliminary)

| metric | value |
|--------|-------|
| success rate | 0.217 |
| object accuracy | 0.174 |
| self-recovery | 0.053 (1/19) |
| **P(success \| correct first grasp)** | **1.000 (4/4)** |

**Object selection is the entire failure mode on these tasks.** Every correct
grasp led to success; wrong grasps recovered only 5% of the time. Nothing else
is failing — not the grasp, not the placement. So steering that fixes object
choice should convert to success nearly 1:1, and the self-recovery correction
term is small rather than the 30-40% that would have muddied the story.

Caveat: n=23, concentrated in the three-`akita_black_bowl` family (tasks 12-13);
`P(s|ok)=1.000` rests on 4 successes and will fall with more data. Re-check this
table when the pilot completes.

### Re-measured at the full baseline (n=40, COMPLETE — supersedes the n=23 table)

| metric | n=23 (prelim) | **n=40 (final)** |
|--------|---------------|------------------|
| success rate | 0.217 | **0.375** [0.242, 0.530] |
| object accuracy | 0.174 | **0.231** (9/39) |
| self-recovery | 0.053 (1/19) | **0.167 (5/30)** |
| **P(success \| correct first grasp)** | 1.000 (4/4) | **1.000 (9/9)** |

**The premise holds and got stronger: `P(s|ok)` survived the n=23 → n=40
jump at exactly 1.000, now on 9 correct grasps rather than 4.** Every single
correct first grasp led to success. Object selection really is the whole
failure mode on these tasks.

**But the self-recovery correction term tripled (5.3% → 16.7%)**, against the
n=23 claim above that it was "small rather than the 30-40% that would have
muddied the story." It is no longer negligible: roughly 1 in 6 wrong grasps
succeeds anyway. Object-accuracy gains overstate expected success gains by
about that margin — keep success rate as the headline, as already decided.

### The pilot's 8 tasks are strongly bimodal — effective N is 20, not 40

| task | success | obj acc | self-rec | band |
|------|---------|---------|----------|------|
| 12 KITCHEN_SCENE2 bowl at back | 2/5 | 2/5 | 0 | measurable |
| 13 KITCHEN_SCENE2 bowl at front | 1/5 | 0/5 | 1 | measurable |
| 14 KITCHEN_SCENE2 middle bowl → plate | 2/5 | 2/5 | 0 | measurable |
| 15 KITCHEN_SCENE2 middle bowl → top of | 0/5 | 0/5 | 0 | **floor** |
| 65 red mug → left plate | 0/5 | 0/5 | 0 | **floor** |
| 66 red mug → right plate | 0/5 | 0/5 | 0 | **floor** |
| 67 white mug → left plate | 5/5 | 4/5 | 1 | **ceiling** |
| 68 yellow-and-white mug → **right** plate | 5/5 | 1/5 | 4 | **ceiling** |

Three regimes, and only one of them can show an effect:

- **Ceiling (67, 68 — 10/10 success).** No headroom; steering can only hurt.
  These are worth keeping precisely as a **safety check** (does intervening
  break something that already worked?), but they cannot show a gain.
- **Floor (15, 65, 66 — 0/15 success, 0/15 correct grasp).** The "wrong hill"
  already documented for red-mug in the openpi steering work: a method that
  amplifies latent competence has nothing to amplify at 0%.
- **Measurable (12, 13, 14 — 5/15 success).** The only band with resolution.

**Consequence for design: the pilot's effective sample for detecting a steering
effect is 15 episodes, not 40.** Any future confusable-subset run should
pre-screen tasks by baseline success rate and drop floor/ceiling tasks, or the
aggregate will be dominated by tasks that structurally cannot move.

**Task 68 is the sharpest single case:** 5/5 success on 1/5 correct first
grasps — four of the six self-recoveries in the entire baseline come from this
one task. It is the concrete existence proof that first-grasp accuracy and
success are genuinely different quantities.

---

## Design issue found in the pilot: the retry budget exhausts immediately

> ⚠️ **CORRECTED 2026-07-20 — the original version of this section was WRONG in
> its central claim, and the error propagated into two sessions of
> interpretation. It said the episode "runs unsteered" after the budget is
> spent. It does not. See the correction immediately below; the framing here is
> kept only so the mistake is legible.**

Observed live (adaptive, task 12, first episodes):

```
verifications=80  failed=21  steers=3  max_esc=3
```

A 400-step episode with `verify_every=5` yields 80 checks. `max_retries=3` is
consumed by the **first three** failed checks — within roughly the first 15-20
steps. The remaining **18 failed verifications get no intervention at all**, so
~95% of the episode runs unsteered. "Adaptive" is currently a burst of three
corrections at the start, not a closed loop.

Options (untested, pick after the pilot completes):

1. **Refresh the budget periodically** — allow `max_retries` per N steps rather
   than per episode. Closest to a true closed loop.
2. **Raise `max_retries`** — `adaptive_r5` preset exists; blunt but trivial.
3. **Only count *escalations* against the budget**, not re-assertions — let the
   top rung keep firing once reached, instead of going silent.
4. **Gate on state change** — re-steer whenever the observed wrong object
   changes, regardless of budget.

Option 3 is probably the cheapest fix that preserves the ATHENA analogy: ATHENA
does not stop steering after k corrections, it keeps applying the (adapted)
weight for the rest of the trajectory.

### CORRECTION: the prompt is sticky — the episode is steered throughout

**The claim above that "~95% of the episode runs unsteered" is false.** The
corrective prompt persists for the rest of the episode once set.

`runner.py`: `current_prompt` is initialised at line 210 and reassigned **only**
at line 263, to a corrected prompt. There is no path that restores the base
instruction. Every subsequent `client.infer` at line 284 sends `current_prompt`.

Verified in the data (adaptive, task 12, ep0 — `steers=3`, `policy_calls=80`):

```
0: put the black bowl at the back on the plate
1: ... . pick up the black bowl
2: ... . pick up the black bowl at the back
3: ... . pick up only the back black bowl, not the other one
```

The ladder is exhausted within the first ~15-20 steps, and prompt 3 — the most
explicit one we ever construct — is then sent on **all ~76 remaining policy
calls**. The correct statement is the near-opposite of the original:

> ~95% of the episode runs under the **strongest** corrective prompt, with no
> further adaptation.

**What the budget actually gates** (narrower than the section above implies):

- further **escalation** up the ladder — but it is already at the top rung, so
  there is nothing left to climb;
- the **chunk discard** (`action_plan.clear()`), which forces an immediate
  replan rather than executing a chunk computed under the old prompt;
- the `n_steer_events` counter.

It does **not** gate the prompt.

**Consequence for interpretation — this is the important part.** The pilot's
adaptive numbers are *not* a crippled test to be discounted pending a fix.
π0.5 received an explicit, unambiguous corrective instruction on 76 consecutive
inference calls and still grasped the wrong bowl. That is much closer to a
genuine null, and it converges with the independent finding from the openpi
steering work: **language modulates the trajectory but does not control
grasp-target selection.**

**Consequence for priority.** Option 3 remains correct — continued adaptation
and forced replanning are both real — but it is **no longer the blocker**, and
should not be described as one. Duration was never the limitation. The
informative experiments are the ones that change the *injection mechanism*:
dual-plan `select` (acts in action space, does not require the policy to honour
text), retraction on commitment, and the 67/68 ceiling safety check.

**Process note for future sessions:** this error survived because the symptom
(`steers=3` against `failed=21`) was read as evidence for a mechanism that was
never checked in the runner. Verify the claim in the code path before building
interpretation on top of a counter.

### CONFIRMED across every steered episode (not just the first few)

Across the first 16 adaptive episodes, the pattern is universal: **every episode
that steered at all reached `max_escalation=3` and stopped escalating**, with
`steers` of 3-5 against `failed` verifications of 8-25. Worst case (t15 ep0):
4 interventions against 25 failed checks.

*(Wording corrected: an earlier revision of this line said these episodes "went
silent." They did not — the top-rung prompt keeps being sent. What stops is
escalation and chunk-discard. See the CORRECTION section above.)*

The mechanism is `steering.py:132`:

```python
return verdict.actionable and self.escalation < self.max_retries
```

`escalation` is doing double duty — it is both the **rung on the prompt ladder**
and the **spent-budget counter**. Reaching rung 3 (the strongest prompt, the one
that names the distractor and rules it out) is exactly what stops further
*intervention events*. The prompt itself stays in force; what ends is
adaptation and the forced replan.

**Fix (option 3) — still correct, but no longer the blocker.** Decouple the two
roles: always intervene when the verdict is actionable, and clamp the ladder
rather than the budget:

```python
def should_intervene(self, verdict):
    return verdict.actionable            # no budget gate
# and in steer(), replace  self.escalation += 1  with:
    self.escalation = min(self.escalation + 1, self.max_retries)
```

This keeps the ladder climbing 1→2→3, then keeps firing at the top rung for the
rest of the episode — matching ATHENA, which keeps applying the adapted weight
rather than stopping after k corrections. The `helped` → hold-level logic is
unaffected.

**Be precise about what this buys**, given the stickiness correction above: the
prompt text on rung 3 is *already* being sent on every call, so the fix does not
add prompt coverage. It adds (a) a chunk discard on each subsequent failed
check, so the policy replans immediately instead of executing a stale chunk, and
(b) live re-evaluation of `helped`, so the level can still be held or adjusted.
Expect a **small** effect, not a rescue of the null.

**Inheritance note:** `DualPlanSteering(AdaptiveSteering)` inherits both methods,
so `dual_select` / `dual_blend` get the same semantics. Those presets have not
been run, so this changes unrun conditions only — it does not revise any
recorded result.

**Test impact:** `tests/test_logic.py:91` asserts
`len(prompts) == 3  # "adaptive stops at max_retries=3"` — this encodes the old
behaviour and must be rewritten to assert the intended semantics: interventions
continue past `max_retries`, distinct rungs still cap at 3
(`len(set(prompts)) == 3`), and `escalation` never exceeds `max_retries`. The
two regression checks either side of it (no self-contradictory negation for
identical instances; no destination-qualifier leakage onto the target) are
unaffected and must keep passing.

**Two consequences to expect, neither a bug:** `n_steer_events` will jump from
3-5 to potentially 25 per episode, so `intervention_rate` changes meaning and is
not comparable across the fix; and `max_escalation` still tops out at 3, so it
remains a valid ladder-depth measure.

**Write the fixed run to a new `run_id` (e.g. `adaptive_persist`), not
`adaptive`.** `episodes.jsonl` is append-only with resume-by-skip, so reusing
the id would silently interleave two different algorithms in one file.

### FINAL pilot result (both arms 40/40 — supersedes the interim section below)

| metric | baseline | adaptive |
|--------|----------|----------|
| success rate | **0.375** [0.242, 0.530] | **0.325** [0.201, 0.480] |
| object accuracy | 0.231 | 0.243 |
| self-recovery | 0.167 (5/30) | 0.107 (3/28) |
| P(success \| correct grasp) | 1.000 | 1.000 |
| intervention rate | 0.000 | **0.750** |
| mean steer events | 0.00 | 2.40 |

By band:

| band | tasks | baseline | adaptive |
|------|-------|----------|----------|
| measurable | 12, 13, 14 | 5/15 | **3/15** |
| floor | 15, 65, 66 | 0/15 | **0/15** (48 steer events, zero movement) |
| **ceiling** | 67, 68 | 10/10 | **10/10** |

**1. The ceiling safety check PASSES.** Steering did not break episodes that
already worked: 10/10 → 10/10, and task 68's object accuracy improved 1/5 → 2/5
under 7 interventions. A sticky corrective prompt is not destabilising. This was
the real open risk and it is now closed.

**2. Everything else is a null, and it is the *predicted* null.** The open
question in this file said: *"if `adaptive` shows a high `intervention_rate` but
no lift in `object_accuracy`, the policy is ignoring our re-steered prompts."*
That is exactly what happened — **`intervention_rate` 0.750 with `object_accuracy`
0.231 → 0.243.** Interventions fired in 30 of 40 episodes and object choice did
not move. Success is nominally lower (0.375 → 0.325) but the CIs overlap heavily;
read this as "no effect," not "harmful."

**3. The floor band is the cleanest single statement.** 48 steer events across
15 episodes on tasks 15/65/66 produced **zero** successes and **zero** correct
first grasps — identical to baseline in both. Maximum intervention, no movement
whatsoever.

**4. Weak harm signal, unresolved.** Episodes grasping nothing at all: 1/40
baseline → 3/40 adaptive. Still too small to call, but the "re-prompting induces
dithering" hypothesis is not ruled out.

Combined with the stickiness correction above — each steered episode ran ~76
consecutive calls under an explicit corrective instruction — **prompt-level
re-steering of π0.5 does not control grasp-target selection on these tasks.**
The next line of work should change the injection mechanism, not its dosage.

### `dual_select` pilot result (40/40, COMPLETE — 2026-07-20 20:42→22:12)

Ran in the existing interactive allocation (job `10436516`, node d13-05),
**not** via sbatch — same in-allocation pattern as the baseline/adaptive pilot.
Server: `serve_policy.py --env LIBERO` on port 8000. Command:

```bash
python -u scripts/run_experiment.py --preset dual_select --port 8000 \
    --task-subset pilot --pilot-size 8 --num-trials-per-task 5
```

This run is doing **two** first-times at once, which is worth separating when
reading the result:

1. **First run of `DualPlanSteering`** (`dual_mode="select"`) — two forward
   passes per steered call, keeping the chunk whose first positional delta
   points more directly at the intended object (`steering.py:260-285`). This is
   the "change the injection mechanism, not the dosage" follow-up the pilot
   conclusion called for — though note `select` still *chooses between* two
   prompt-conditioned chunks, so it is only a partial escape from prompt-level
   control.
2. **First live exercise of the 10b retry-budget fix.** `DualPlanSteering`
   inherits the un-gated `should_intervene`, so this run carries it. The
   predicted signature is visible: `n_steer_events` per episode now reaches
   24-33 where the adaptive pilot averaged 2.40, and `max_escalation` still caps
   at 3. That is the expected consequence, not a bug — but it does mean
   `intervention_rate` and `mean_steer_events` are **not comparable** to the
   adaptive pilot's numbers.

All three arms, 40 episodes each:

| metric | baseline | adaptive | dual_select |
|--------|----------|----------|-------------|
| success rate | **0.375** [0.242, 0.530] | 0.325 [0.201, 0.480] | **0.325** [0.201, 0.480] |
| object accuracy | 0.231 | 0.243 | **0.243** |
| wrong-object rate | 0.769 | 0.757 | 0.757 |
| self-recovery | 0.167 (5/30) | 0.107 (3/28) | **0.107 (3/28)** |
| P(success \| correct grasp) | 1.000 | 1.000 | **0.889** |
| intervention rate | 0.000 | 0.750 | 0.725 |
| mean steer events | 0.00 | 2.40 | **12.28** |
| policy calls / episode | 61.5 | 64.2 | **109.1** |
| wall s / episode | 79.9 | 82.4 | 133.0 |

By band (success / n):

| band | tasks | baseline | adaptive | dual_select |
|------|-------|----------|----------|-------------|
| measurable | 12, 13, 14 | 5/15 | 3/15 | 4/15 |
| floor | 15, 65, 66 | 0/15 | 0/15 | **0/15** |
| ceiling | 67, 68 | 10/10 | 10/10 | **9/10** |

**1. dual_select is metric-for-metric identical to adaptive** on success, object
accuracy, wrong-object rate and self-recovery — at **5× the interventions, 1.7×
the policy calls and 1.7× the wall time**. Correct first grasps: **9/40 in all
three arms.** Changing the injection mechanism from prompt-level re-steering to
chunk-level re-selection moved nothing. This is the second consecutive null and
a stronger one: `select` chooses between two *already-computed* chunks using
ground-truth geometry, so it does not depend on the policy honouring
instruction text at all — and it still fails to move object choice.

**2. The floor band is the cleanest statement in the project.** 0/15 in every
arm, with dual_select spending 17-30 steer events per episode there. Prompt-level
and chunk-level intervention fail *identically* where the policy has no latent
competence to amplify. ATHENA's premise — amplify a capability the model already
has — has no purchase on this band.

**3. Ceiling: 9/10, the first ceiling loss across 30 ceiling episodes.** The loss
is t68 ep3 (9 steer events). dual_select steers 4 of 5 episodes on task 68 at
9-11 events each, where adaptive steered 3 of 5 at 1-3 — so the hardest-
intervening arm is the one that broke an episode. **This is n=1 and not
distinguishable from noise.** Treat as a watch item, not a finding; the ceiling
safety check should be re-read as "no longer clean," not "failed."

**CORRECTION (2026-07-21) — an earlier draft of this section dismissed task 68 as
a broken probe. That was wrong and is retracted.** The claim was that t68
"succeeds while grasping the wrong object, so its success predicate does not
require a correct grasp." It does not hold up: `grasped_object()` uses
robosuite's contact-based `_check_grasp` (`perception.py:123-140`), and
`first_grasped` records only the **first** commitment (`runner.py:251`). So t68's
pattern — wrong first grasp, success anyway — is the policy grabbing the white
mug, releasing, and recovering onto the correct one. That is exactly what the
**self-recovery** metric measures, and the arithmetic confirms it: baseline's 5
self-recoveries are 4 from t68 plus 1 elsewhere. Nothing is mis-scored. **t68 is
a legitimate probe, and is in fact the one task where π0.5 reliably
self-corrects** — which makes the dual_select loss there slightly more
interesting, not less. Do not re-dismiss it without new evidence.

**4. `P(s|ok)` broke its 1.000 invariant — and not on the ceiling episode.**
The violation is **t12 ep2**: correct object grasped, 9 steer events, escalated
to rung 3 and then `reassert@3:wrong_object` six more times, ran to the 400-step
wall. Steering kept firing `wrong_object` *after* the correct grasp. See Known
issues — this is a real interaction bug between the verifier and the un-gated
`should_intervene`, not a property of dual-plan selection.

**Caveat that limits all of the above: this run confounds two changes.** It is
the first `DualPlanSteering` run *and* the first live run of the 10b
retry-budget fix (inherited via `AdaptiveSteering`). `intervention_rate` and
`mean_steer_events` are therefore **not comparable** to the adaptive pilot
(2.40 → 12.28 mean steer events is the fix, not the mechanism). To attribute the
null to dual-plan specifically, an `adaptive_persist` arm — the fix without the
dual-plan mechanism — is needed. Cheap and worth doing before the confusable
subset.

### Did steering help at all? No — and the episode-level view is the proof

Seeds are identical per `(task, episode)` slot across all three arms, so these
are **matched** comparisons, not independent samples. Correct first grasps:

| | count | shared with baseline | gained | lost |
|---|---|---|---|---|
| baseline | 9/40 | — | — | — |
| adaptive | 9/40 | 8 of 9 | +1 (t68 ep1) | −1 (t12 ep1) |
| dual_select | 9/40 | **6 of 9** | +3 (t12 ep2, t14 ep2, t68 ep0) | −3 (t12 ep1, t67 ep0, t68 ep3) |

Successes, dual_select vs baseline: **+1** (t14 ep2), **−3** (t12 ep1, t13 ep1,
t68 ep3).

**The identical 9/40 is not steering holding the line — it is gains and losses
cancelling.** dual_select churns twice as much as adaptive (6/9 shared vs 8/9)
and steers 5× as often. Churn scaling with intervention count, at zero net
effect, is the signature of **adding noise to the system, not exerting control
over it**. Only one episode in 120 was converted to a success, against three
converted the other way.

Limit on the claim: "no help" is measured where measurement is possible, i.e.
15 episodes. A few-percentage-point effect is not excluded. But the
grasp-*distribution* evidence does not depend on N the same way — on t15, t65
and t66 the distribution under 17-30 steer events/episode is **bit-identical to
the zero-intervention baseline**. That is not a small effect hidden in noise; it
is no influence at all on the quantity the mechanism exists to control.

### ROOT CAUSE: π0.5 has a fixed per-scene grasp attractor (2026-07-21)

Pooling all three arms (the instruction varies across tasks within a scene; the
policy's behaviour does not):

| scene | episodes | most-grasped object | share | grasp == named target |
|---|---|---|---|---|
| KITCHEN_SCENE2 | 60 | `akita_black_bowl_3` | 58% | **20%** |
| LIVING_ROOM_SCENE5 | 60 | `porcelain_mug_1` | **82%** | **25%** |

Both are **at or below the 33% expected from picking randomly among three
candidate objects.** In LIVING_ROOM_SCENE5, `red_coffee_mug_1` was grasped
**0/60** despite being the named target in 30 of those episodes. In
KITCHEN_SCENE2, `akita_black_bowl_1` was grasped 5% despite being named in 15.

**This single fact explains every result in the pilot:**

- **t67 succeeds 5/5 in all arms because the attractor *is* the target.**
  Steering fires zero times — the verifier never sees a wrong object. The
  "ceiling band" is not competence being preserved; it is the attractor
  coinciding with the instruction.
- **t65/t66/t15 are 0/15 because the attractor is a fixed wrong object.** Under
  17-30 steer events per episode the distribution is unchanged. Steering had no
  purchase, at any dosage, by either mechanism.
- **t12/t13/t14 look noisy because the attractor is softer there** (b3 58%, b2
  32%). Successes are draws that happened to land on the target, which is why
  they move between arms with no relation to intervention count.

**The sharpest single probe is t14 vs t15.** Same scene, same target object
(`akita_black_bowl_2`), same distractors. Only the *destination* phrase differs
("on the plate" vs "on top of the cabinet"). t14 gets 2-3/5; t15 gets **0/15
with a rigid `bowl_3` grasp in every arm.** A phrase carrying no information
about *which bowl* flips grasp behaviour completely — so the instruction is
being consumed as a whole-string mode selector, not parsed compositionally into
a target.

**Why this reframes the whole project.** Both interventions were competing
against a scene-level prior that the instruction itself barely moves. A third
steering arm measures the same wall. The open question is no longer "how do we
steer better" but "**is grasp choice a function of the scene rather than the
instruction?**" — which is answerable with **no steering at all**, and is a
positive characterisation rather than a third null. That is stage 16.

**Not read as harm:** within dual_select, episodes with `steers=0` succeed and
steered episodes largely fail. That is the *same selection-bias artifact*
documented in the interim adaptive section below — steering fires precisely when
the policy is already heading for the wrong object. Only across-arm comparison on
matched tasks is valid.

### Interim adaptive result (n=16, pilot still in flight — do not quote as final)

On the three matched measurable tasks (12, 13, 14): baseline 5/15 success,
adaptive 3/15. Worse, but far inside noise at n=15.

The eye-catching pattern — **every episode where steering fired failed, and all
three successes had `steers=0`** — is **selection bias, not evidence of harm**.
Steering fires precisely when the policy is already heading for the wrong
object; the unsteered successes were never in trouble. Within-condition
"steered vs unsteered" is not a valid comparison. Only adaptive-vs-baseline on
matched tasks is, and that is the 3/15 vs 5/15 above.

One thing genuinely worth watching: **2/16 adaptive episodes grasped nothing at
all** (t14 ep0, ep2 — `first_grasped=None`, 3-4 steers each) versus **1/40** in
baseline. Too small to conclude anything, but "re-prompting induces dithering
until the clock runs out" is a plausible harm mechanism and cheap to check once
n grows.

**How much weight this carries, post-correction.** Because the prompt is sticky,
these episodes were not under-treated: each ran ~76 consecutive policy calls
under an explicit, unambiguous corrective instruction naming exactly which bowl
to take. So this is a *fair* test of prompt-based correction, not a preview of
one — and it read as no effect. It should be treated as a real (if small-n)
null, and the burden now sits on mechanisms that do not rely on the policy
honouring instruction text.

---

## CONFIRMED: a fixed motor routine, not object selection (2026-07-30)

`living5_char`, 20 episodes, all 4 LIVING_ROOM_SCENE5 tasks × 5, **no steering**,
full per-step event logging from ground-truth sim state. This settles what the
attractor finding only suggested.

**Two methodological changes made the behaviour visible for the first time:**

1. **Log every grasp/release/placement for every object, not just
   `first_grasped`.** A two-step routine is *invisible* in `(first_grasped,
   success)` — which is why the task-68 story was misread twice.
2. **Do not stop the episode at success.** LIBERO's `done` is just
   `_check_success()` recomputed each step, so stepping past it is safe. On
   task 67 the goal holds at t≈92 while the unprompted second grasp happens at
   t≈223 — every prior run truncated ~130 steps before the interesting part.
   `success_t` is recorded separately so the success metric is unchanged.
   (`scripts/run_characterization.py`, `athena/eventlog.py`.)

**The routine.** Regardless of which mug is named:

```
phase 1   t≈52-95    grasp WHITE mug (porcelain_mug_1)  -> place on LEFT plate  (plate_1)
phase 2   t≈187-260  grasp YELLOW-WHITE (white_yellow_mug_1) -> place on RIGHT plate (plate_2)
red_coffee_mug_1: never touched, 0/20, mean gripper distance 0.173 m vs 0.089-0.092 for the two it handles
```

**The terminal scene state is identical in 20/20 episodes across all four
instructions**: white on plate_1, red on table, yellow-white on plate_2.

**One fixed terminal state predicts every task's success rate with no reference
to language:**

| task | goal | satisfied by terminal state? | predicted | pilot | `living5_char` |
|---|---|---|---|---|---|
| 65 | red on LEFT | no | 0% | 0/5 | 0/5 |
| 66 | red on RIGHT | no | 0% | 0/5 | 0/5 |
| 67 | white on LEFT | **yes — phase 1** | 100% | 5/5 | 5/5 |
| 68 | yellow-white on RIGHT | **yes — phase 2** | 100% | 5/5 | 5/5 |

40/40 episodes predicted exactly. **Naming a mug does not change when it is
grasped:** yellow-white is first grasped at t≈187 when named (t68) and t≈187-201
when not named (t65/66/67). White at t≈51-66 whether named or not.

First grasp is the white mug in **18/20 episodes (90%)**, including 3 of the 4
tasks where it is *not* the named target.

**Consequences that revise entries above in this file:**

- **The "ceiling band" is an artifact.** t67 and t68 are 10/10 not because
  competence is preserved but because the routine's phase 1 and phase 2
  respectively satisfy their goals. t67 records **zero steer events in all three
  arms** — the verifier never fires. These two tasks cannot measure steering and
  must be dropped from future evaluation.
- **t68's "self-recovery" is not recovery.** 5/5 of its successes are
  routine-coincidental: target not grasped first, picked up second as phase 2.
  There is no notice-and-correct. Since 4 of baseline's 5 self-recoveries are
  t68, the reported self-recovery rate (0.167) is largely this artifact. **This
  is the third revision of the t68 story** (mis-scored -> self-recovery ->
  routine); the earlier retraction defending self-recovery is itself now
  retracted. The cause of all three passes was logging only two scalars.
- **Do not run steering arms on this scene.** There is no selection process to
  redirect. STOP condition met.

**Detector caveat found here:** t67 ep1 and t65 ep3 succeeded/moved the white mug
with **no `_check_grasp` event for it at all** (min gripper distance 0.101 m, in
the same range as confirmed grasps). `first_grasped` under-reports grasps, so
some historical "grasped nothing" episodes (1/40 baseline, 3/40 steered) are
likely detection misses rather than dithering — this weakens the
"re-prompting induces dithering" harm hypothesis.

**Mechanistic hypothesis (untested).** All four tasks share near-identical
images; only the sentence differs. A policy underweighting language relative to
vision cannot separate them and appears to have collapsed them into one
scene-conditioned motor program. Checkable prediction: **the routine should still
run under an empty or nonsense instruction.** That is the cheap next experiment
and would convert "steering failed" into a positive finding about how pi0.5
binds language to behaviour.

---

## THE DISSOCIATION: gibberish probe (2026-07-30)

The decisive experiment. If the mug-scene routine ignores the instruction, a
**meaningless** instruction should change nothing. It doesn't — and in a
different scene the same test breaks the policy completely.

Nonsense string, identical in every run, chosen to be pronounceable, free of
real content words and object nouns, and close in token count to a genuine
LIBERO instruction (an *empty* prompt would confound "no meaning" with "no
tokens"):

```
"blicket dax fep wug zorp tulver nace"
```

Success is still scored by LIBERO's own unmodified `_check_success()`; only the
string handed to pi0.5 changes. Each record carries `prompt_sent` and
`prompt_is_override`.

### Mug scene (libero_90 LIVING_ROOM_SCENE5) — language does NOTHING

| task | condition | success | 1st grasp correct | success_t |
|---|---|---|---|---|
| 67 | real instruction | 5/5 | 4/5 | t≈94 |
| 67 | **gibberish** | **5/5** | **5/5** | t≈103 |
| 68 | real instruction | 5/5 | 0/5 | t≈225 |
| 68 | **gibberish** | **5/5** | **1/5** | t≈224 |

**10/10 success on nonsense.** Same routine, same order, same clock. Task 68's
success time is unchanged to three steps (225 -> 224) — it still sits through
phase 1 before the routine reaches the step matching its goal. The gibberish run
was *marginally more* routine-consistent than the language run.

### Orange juice scene (libero_object task 9) — language does a LOT

| condition | success | 1st grasp correct | first object grasped |
|---|---|---|---|
| real instruction | **5/5** | **5/5** | orange_juice_1 ×5 |
| **gibberish** | **1/5** | **1/5** | **butter_1 ×3**, none ×1, orange_juice_1 ×1 |

Six pickable objects in this scene. With a real instruction pi0.5 picks the
right one 5/5; with nonsense it goes for the butter.

### What this establishes

**pi0.5 is not a policy that ignores language.** It grounds language reliably in
the orange-juice scene and not at all in the mug scene. The correct claim is:

> pi0.5 grounds language in some scenes and falls back on a fixed,
> scene-triggered motor routine in others. Where the routine dominates, the
> instruction has no influence on object selection, and benchmark success
> measures whether the instruction happened to describe what the routine was
> going to do anyway.

**This reframes both nulls rather than adding a third.** Every steering episode
ever run in this project was on scenes where the fallback dominates — 65/66/67/68
are the mug scene; 12/13/14/15 are KITCHEN_SCENE2, which the attractor analysis
showed has the same problem in weaker form (`akita_black_bowl_3` 58%). The
steering arms were not failing because the steering was bad. **There was no
selection process to redirect.**

Three results now stand where there was one failure:
1. a benchmark-scoring artifact — success is earnable without listening;
2. a characterised failure mode — scene-triggered motor routines overriding
   instructions;
3. a **positive control** proving the model *can* ground language, so (2) is a
   scene property, not a model-wide limitation.

### Caveat

5 episodes per cell. The mug result is safe (effect size zero, behaviour rigidly
identical). **The orange-juice 5/5 -> 1/5 contrast should be extended to ~20
episodes/cell before it goes in a writeup** — ~25 min of GPU. Also untested:
whether the effect is specific to this nonsense string (a second string, or an
empty prompt, would settle it).

### What to do next

- **Extend the orange-juice cells** to n=20 each. Cheapest, highest value.
- **Run the steering arms on libero_object**, not on LIBERO-90's mug/bowl
  scenes. This is the first scene where a selection process demonstrably exists,
  so `adaptive` / `dual_select` finally have something to act on, and
  `first_grasp_object` (not success) is the metric.
- **Screen every candidate scene with the gibberish probe first.** A scene where
  gibberish scores as well as language cannot measure steering. This is a cheap
  5-episode pre-filter and should gate stage 11/12 task selection.

---

## Known issues / gotchas

- **The verifier keeps firing `wrong_object` after a correct grasp, and with the
  10b fix that now costs episodes.** Found in `dual_select` t12 ep2 (the only
  `P(s|ok)` violation ever recorded, 1.000 → 0.889): the episode grasped the
  correct object, yet steering escalated to rung 3 and then issued six
  `reassert@3:wrong_object` events, running out the 400-step clock instead of
  completing.
  **Why it appears now.** The old budget capped interventions at 3, so a stale or
  mistaken verdict self-limited and was invisible. The 10b fix removed the budget
  gate deliberately (`should_intervene` returns `verdict.actionable`), so the same
  bad verdict now re-fires for the rest of the episode — and each re-fire discards
  the action chunk, forcing a replan. A policy that has already committed
  correctly gets repeatedly interrupted.
  **This is a verifier-correctness question, not a steering-dosage one.** Either
  the verifier is wrong about which object is grasped once the gripper closes, or
  it is right and the metric's `grasped_correct` disagrees with it — those two
  cases have opposite fixes and have **not** been distinguished yet. Do that
  first, by dumping per-step verdicts for t12 ep2 against the recorded
  `first_grasped`.
  **Likely fix once diagnosed:** latch the verifier off (or downgrade the verdict
  to non-actionable) once a correct grasp is committed, so `wrong_object` cannot
  re-fire against a correct commitment. Do not simply restore the budget — that
  re-hides the bug rather than fixing it.
  **Scope:** affects every arm that inherits the un-gated `should_intervene`
  (`adaptive` post-10b, `dual_select`, `dual_blend`). The recorded pre-fix
  adaptive pilot is unaffected.
- **Prior job 9358798 died instantly**: `set -euo pipefail` + bare `$PYTHONPATH`
  → `unbound variable`. Fixed in our sbatch via `${PYTHONPATH:-}`. The old
  `openpi/run_libero90.sbatch` still has this bug if you reuse it.
- `MUJOCO_GL=egl` is **required** or env creation fails headless.
- LIBERO must be on `PYTHONPATH` (it is not pip-installed).
- The LIBERO venv is Python 3.8 — no `match`, no `X | Y` at runtime in
  annotations (we use `from __future__ import annotations` everywhere; keep it).
- `robosuite` prints macro warnings on every import; harmless.
- Never run the policy server on a login node — GPU + long-running, must be in
  a SLURM allocation.

---

## Open questions to resolve

- ~~**Does π0.5 actually respond to mid-episode prompt changes?**~~
  **ANSWERED 2026-07-30, and the answer is scene-dependent — see "THE
  DISSOCIATION".** In LIVING_ROOM_SCENE5 the policy scores 10/10 on a *nonsense*
  prompt, so language has no influence on object choice there and no amount of
  re-prompting could have worked. In libero_object the same nonsense drops it
  5/5 -> 1/5, so language *does* drive selection there. The old provisional
  answer below ("apparently not") was right about the scenes we tested and wrong
  as a general claim about the policy. Retained for the record:
  **PROVISIONAL ANSWER: apparently not — at n=15, on the measurable band.**
  This was flagged as the single biggest risk to the whole approach (doc §7,
  "Policy Non-cooperation"), and the pilot's test condition turned out to be
  stronger than designed: because the prompt is sticky, each steered episode ran
  ~76 consecutive inference calls under an explicit corrective instruction. The
  predicted failure signature — interventions fire, `object_accuracy` does not
  move — is what we observe (baseline 4/15 → adaptive 3/15 object accuracy,
  5/15 → 3/15 success).
  Not yet conclusive: n=15, one task family (`akita_black_bowl`), oracle
  detector. But the burden has shifted, and **a different injection mechanism is
  now the main line, not a contingency.** Corroborated independently in the
  openpi steering work, which measured that noun-level object identity moves
  π0.5's actions by ~25% rel_L2 while leaving the grasped object unchanged —
  i.e. the policy reads the prompt and acts differently, yet still converges on
  the same visually salient object.
- Is `grasp_radius=0.055 m` the right commitment threshold? Tuned by eye from
  one episode; `adaptive_strict` / `adaptive_loose` presets sweep it.
- Should a re-steer also *retract* the gripper, rather than only changing the
  prompt? Currently we only re-prompt; if the gripper has already closed on the
  wrong object, a prompt change may be too late.

---

## Experiment presets

`baseline`, `static`, `adaptive`, `dual_select`, `dual_blend`,
`adaptive_noisy_light`, `adaptive_noisy_heavy`, `adaptive_late`,
`adaptive_r1`, `adaptive_r5`, `adaptive_strict`, `adaptive_loose`

```bash
# pilot (fast; ~40 episodes/preset)
PRESET_ARGS="--task-subset pilot --pilot-size 8 --num-trials-per-task 5" \
  sbatch scripts/run_athena.sbatch baseline adaptive

# confusable subset (32 tasks)
PRESET_ARGS="--task-subset confusable --num-trials-per-task 10" \
  sbatch scripts/run_athena.sbatch baseline static adaptive dual_select

# full suite
PRESET_ARGS="--task-subset all --num-trials-per-task 5" \
  sbatch scripts/run_athena.sbatch baseline adaptive
```

---

## Live run log

Append one row per submitted job so a later session can tell what was actually
executed vs merely configured.

| Job ID | Date | Presets | Scope | Outcome |
|--------|------|---------|-------|---------|
| 10434741 | 2026-07-20 | baseline, adaptive | pilot, 8 tasks × 5 eps | **cancelled** — queue had 437 pending GPU jobs; ran in an existing allocation instead |
| 10428083 (interactive) | 2026-07-20 | baseline, adaptive | pilot, 8 tasks × 5 eps (80 ep) | **complete 80/80** — see "FINAL pilot result" |
| 10436516 (interactive) | 2026-07-20 | dual_select | pilot, 8 tasks × 5 eps (40 ep) | **complete 40/40** (20:42→22:12, d13-05) — 0.325 success; see "dual_select pilot result". Allocation later died 23:23 (exit 137, SIGKILL — **not** OOM: MaxRSS 826 MB / 48 G; the interactive shell holding the `salloc` ended). All results had already been written to `/scratch1` and are intact |
| 10438099 (interactive) | 2026-07-21 | attractor_baseline | confusable, 32 tasks × 10 eps (320 ep) | **DIED 30/320** — exit 137, allocation shell ended 01:59 (MaxRSS 986 MB, not OOM). Tasks 0-2 recorded and intact; resumable |
| 10709603 (interactive) | 2026-07-30 | living5_char | LIVING_5, 4 tasks × 5 eps (20 ep) | **complete 20/20** (06:56→07:29, d13-06) — fixed routine confirmed; see "CONFIRMED: a fixed motor routine" |
| 10709603 (interactive) | 2026-07-30 | gibberish_mugs / language_oj / gibberish_oj | tasks 67,68 + libero_object 9, 5 eps each (20 ep) | **complete 20/20** (08:07→08:36, d13-06) — mugs 10/10 on nonsense; OJ 5/5→1/5. **Videos saved** under `results/<run_id>/videos/`. See "THE DISSOCIATION" |

### Check for an idle allocation before queueing

`sbatch` is not always the fastest path. On 2026-07-20 the GPU queue was 437
jobs deep while two interactive allocations were already held and **idle**
(V100, 0% utilisation). Running in the existing allocation skipped the wait
entirely. Before submitting, always run:

```bash
squeue -u $USER -o "%.10i %.14j %.10T %.10M %.12l %R"   # already hold a GPU?
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv  # is it idle?
```

To run in-allocation rather than via sbatch:

```bash
cd /scratch1/nalagand/openpi
export PYTHONPATH="${PYTHONPATH:-}:/scratch1/nalagand/openpi/third_party/libero:/scratch1/nalagand/athena_robotics"
nohup uv run scripts/serve_policy.py --env LIBERO \
  > /scratch1/nalagand/athena_robotics/logs/server_$(hostname -s).log 2>&1 &
# wait for port 8000, then:
source /scratch1/nalagand/openpi/examples/libero/.venv/bin/activate
export MUJOCO_GL=egl
cd /scratch1/nalagand/athena_robotics
python scripts/run_experiment.py --preset adaptive --task-subset pilot \
    --pilot-size 8 --num-trials-per-task 5
```

Run the server under `nohup` so it outlives a dropped session. Kill it with
`kill $(cat logs/server.pid)` when done — an orphaned server holds the GPU.

To check an old job after the fact:

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -50 logs/athena_<jobid>.out
tail -30 logs/server_<jobid>.log      # policy-server side
```

---

## Changelog

- **2026-07-30 (latest)** — **Gibberish-prompt probe: a clean dissociation.**
  Feeding pi0.5 the nonsense string `"blicket dax fep wug zorp tulver nace"`
  instead of the instruction leaves the mug scene **completely unchanged**
  (10/10 success, same routine, same timing to within 3 steps) while breaking
  the orange-juice scene (**5/5 -> 1/5**, grabbing butter 3/5). So pi0.5 *can*
  ground language — it just doesn't in LIVING_ROOM_SCENE5, where a fixed motor
  routine overrides it. **This reframes both earlier nulls: the steering arms
  were tested almost entirely on scenes with no selection process to redirect.**
  New main line (stage 19): run steering on libero_object, and gibberish-screen
  every scene before spending GPU on it. Videos saved for all 20 episodes.
  Also fixed a real bug in `eventlog.py`: it hardcoded the `On` predicate and
  would have silently reported "never placed" for every basket task, whose goal
  uses `In` against a site contain-region.
- **2026-07-30 (latest)** — **The attractor is a fixed two-phase motor routine,
  and it is instruction-independent.** New full-episode event logging
  (`athena/eventlog.py`, `scripts/run_characterization.py`) plus not
  terminating at success revealed that in LIVING_ROOM_SCENE5 pi0.5 always runs
  white->left plate, then yellow-white->right plate, never touching the red mug
  (0/20). The terminal scene state is identical in **20/20 episodes across all
  four instructions**, and that single state predicts all four tasks' success
  rates exactly (40/40 episodes incl. the pilot). Naming a mug does not change
  when it is grasped. **The 'ceiling band' is an artifact** (t67 = phase 1,
  t68 = phase 2) and **t68's 'self-recovery' is phase 2, not recovery** —
  third and final revision of that story. STOP condition met: **do not run
  steering arms on this scene.** Also fixed a load-bearing data error in this
  file: task 68 is the **right** plate, not the left. Next: nonsense-instruction
  probe to test whether the routine is language-gated at all.
- **2026-07-21 (latest)** — **Root cause identified: a fixed per-scene grasp
  attractor.** Pooling all 120 pilot episodes, the policy grasps
  `porcelain_mug_1` 82% of the time in LIVING_ROOM_SCENE5 and
  `akita_black_bowl_3` 58% in KITCHEN_SCENE2 **regardless of which object the
  instruction names**; grasp matches the named target only 20-25%, at or below
  the 33% random-choice rate. `red_coffee_mug_1` was grasped 0/60 while named in
  half of those episodes. This explains every pilot result — including the
  "ceiling" band, which is just the attractor coinciding with the target — and
  reframes both nulls as interventions competing against a prior the instruction
  itself barely moves. **Stage 16 (`attractor_baseline`) launched** to test it
  properly on 32 tasks with no steering at all. Also added the episode-level
  matched-seed analysis (steering churns which episodes land, +1/−3 net, at
  identical 9/40 correct grasps) and **retracted an incorrect claim** that task
  68 was mis-scored — its wrong-first-grasp successes are genuine self-recovery.
- **2026-07-20** — **`dual_select` pilot complete, 40/40** (job
  10436516, d13-05, 20:42→22:12). **Second null, and a stronger one:**
  identical to adaptive on success (0.325), object accuracy (0.243),
  wrong-object rate and self-recovery, at 5× the interventions and 1.7× the
  compute. Correct first grasps 9/40 in all three arms. Floor band 0/15 in every
  arm. Chunk-level re-selection — which uses ground-truth geometry and does not
  require the policy to honour instruction text — moves object choice no more
  than prompt-level re-steering did. Two new items: ceiling **9/10**, the first
  ceiling loss in 30 ceiling episodes (n=1, not separable from noise); and
  **`P(s|ok)` 1.000 → 0.889**, traced to a verifier bug now logged under Known
  issues. **Confounded run** — first dual-plan *and* first live 10b fix; an
  `adaptive_persist` arm would be needed to separate them, but see the attractor
  finding below before spending GPU hours on it.
- **2026-07-20** — **Pilot complete, both arms 40/40.** baseline 0.375
  vs adaptive 0.325 (overlapping CIs); `intervention_rate` 0.750 with
  `object_accuracy` 0.231 → 0.243 — the exact "policy ignores re-steered prompts"
  signature this file predicted. Ceiling safety check **passed** (10/10 → 10/10).
  Floor band: 48 steer events, zero movement. **Retry-budget fix applied** and
  `test_logic.py` updated; 47/47 checks green — but note the fix is untested
  against a live server, and given stickiness it is expected to be a small
  effect, not a rescue. Next line should change the *injection mechanism*
  (dual-plan `select`, retraction), not the dosage.
- **2026-07-20 (later)** — Baseline pilot complete (40/40): success 0.375,
  `P(s|ok)` held at 1.000 on 9 correct grasps, self-recovery revised up
  0.053 → 0.167. Found the 8 pilot tasks are **bimodal** (2 ceiling, 3 floor,
  3 measurable) — effective N is 15, not 40. **Corrected a load-bearing error in
  this file:** the retry budget does not leave the episode unsteered; the
  corrective prompt is sticky (`runner.py:210/263`) and runs for ~95% of the
  episode at the strongest rung. This reframes the adaptive result from
  "crippled test" to "closer to a genuine null" and demotes the retry-budget fix
  from blocker to nice-to-have.
- **2026-07-20** — Project created. Stages 1–9 complete: full pipeline
  implemented and validated against the live simulator (12/12 checks). Found
  32/90 confusable tasks. Fixed the `set -u` sbatch bug inherited from the
  prior job. Next: stage 10, the pilot run.
