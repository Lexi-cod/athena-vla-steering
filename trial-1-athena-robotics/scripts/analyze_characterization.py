"""Analysis for the LIVING_ROOM_SCENE5 multi-object routine characterisation.

Answers sections A, B and C of the characterisation brief, and evaluates the
STOP / PROCEED condition.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

LABEL = {
    "porcelain_mug_1": "white",
    "red_coffee_mug_1": "red",
    "white_yellow_mug_1": "yellow-white",
    "plate_1": "LEFT plate",
    "plate_2": "RIGHT plate",
    "table": "table",
}
MUGS = ["porcelain_mug_1", "red_coffee_mug_1", "white_yellow_mug_1"]


def lab(x):
    return LABEL.get(x, x)


def seq_str(order):
    return " -> ".join(lab(o) for o in order) if order else "(nothing grasped)"


def load(path):
    recs = [json.loads(l) for l in pathlib.Path(path).open() if l.strip()]
    by_task = collections.defaultdict(list)
    for r in recs:
        by_task[r["task_id"]].append(r)
    for v in by_task.values():
        v.sort(key=lambda r: r["episode_idx"])
    return by_task


def section_a(by_task):
    print("=" * 78)
    print("A. PER-RUN OBJECT INTERACTION SUMMARY")
    print("=" * 78)
    for tid in sorted(by_task):
        recs = by_task[tid]
        r0 = recs[0]
        print(f"\n--- task {tid}: {r0['language']!r}")
        print(f"    goal: {lab(r0['target'])} mug -> {lab(r0['target_receptacle'])}")
        print(f"    {'ep':<3} {'grasp order':<34} {'success':<9} {'final: white / red / yellow-white'}")
        for r in recs:
            fl = r["final_locations"]
            locs = " / ".join(lab(fl[m]) for m in MUGS)
            s = f"{r['success']}"
            if r["success"]:
                s += f"@{r['success_t']}"
            print(f"    {r['episode_idx']:<3} {seq_str(r['grasp_order']):<34} {s:<9} {locs}")

        # aggregate touch/grasp
        touched = {m: sum(bool(r["touched"][m]) for r in recs) for m in MUGS}
        grasped = {m: sum(m in r["grasp_order"] for r in recs) for m in MUGS}
        n = len(recs)
        print(f"    touched : " + ", ".join(f"{lab(m)} {touched[m]}/{n}" for m in MUGS))
        print(f"    grasped : " + ", ".join(f"{lab(m)} {grasped[m]}/{n}" for m in MUGS))
        mis = collections.Counter()
        for r in recs:
            for m in r["misplaced_on_plate"]:
                mis[(m, r["final_locations"][m])] += 1
        if mis:
            print("    UNPROMPTED placements (object not named by the instruction):")
            for (m, plate), c in sorted(mis.items()):
                print(f"        {lab(m)} mug -> {lab(plate)}   in {c}/{n} episodes")
        else:
            print("    unprompted placements: none")

        orders = collections.Counter(tuple(r["grasp_order"]) for r in recs)
        print("    grasp-order distribution:")
        for o, c in orders.most_common():
            print(f"        {c}/{n}  {seq_str(list(o))}")


def section_b(by_task, white_task, yellow_task):
    print("\n" + "=" * 78)
    print("B. FIRST-GRASP CORRECTNESS: white-direct vs yellow-direct")
    print("=" * 78)
    for tid, name in ((white_task, "white-direct"), (yellow_task, "yellow-direct")):
        if tid not in by_task:
            continue
        recs = by_task[tid]
        n = len(recs)
        correct = sum(bool(r["first_grasp_correct"]) for r in recs)
        succ = sum(bool(r["success"]) for r in recs)
        print(f"\n  task {tid} ({name}, target={lab(recs[0]['target'])}):")
        print(f"    first grasp correct : {correct}/{n}")
        print(f"    success             : {succ}/{n}")

        # The caveat: success delivered by step 2 of a fixed routine.
        coincidental = []
        for r in recs:
            order = r["grasp_order"]
            tgt = r["target"]
            if (
                r["success"]
                and order
                and order[0] != tgt
                and tgt in order
                and order.index(tgt) > 0
            ):
                coincidental.append(r)
        if coincidental:
            print(f"    *** {len(coincidental)}/{n} successes are ROUTINE-COINCIDENTAL:")
            print(f"        the target was NOT grasped first; an unnamed mug was handled")
            print(f"        first, and the target was picked up later in a fixed sequence")
            print(f"        whose step happens to satisfy the instruction.")
            for r in coincidental:
                order = r["grasp_order"]
                print(f"          ep{r['episode_idx']}: {seq_str(order)}"
                      f"  (target {lab(r['target'])} grasped at position "
                      f"{order.index(r['target']) + 1})")
            print(f"        => these are NOT evidence of language grounding.")
        else:
            print(f"    routine-coincidental successes: 0")


def section_c(by_task, yellow_task, other_tasks):
    print("\n" + "=" * 78)
    print("C. LANGUAGE-EFFECT TEST: is the named mug ever grasped FIRST?")
    print("=" * 78)
    print("  Genuine language evidence requires: yellow-white grasped FIRST when")
    print("  yellow-white is named, AND not grasped first when red/white is named.\n")

    def first_is(recs, obj):
        return sum(1 for r in recs if r["grasp_order"] and r["grasp_order"][0] == obj)

    Y = "white_yellow_mug_1"
    out = {}
    if yellow_task in by_task:
        recs = by_task[yellow_task]
        c = first_is(recs, Y)
        out["named"] = (c, len(recs))
        print(f"  yellow-white named   (task {yellow_task}): "
              f"grasped FIRST in {c}/{len(recs)} episodes")
    for tid in other_tasks:
        if tid not in by_task:
            continue
        recs = by_task[tid]
        c = first_is(recs, Y)
        out[tid] = (c, len(recs))
        print(f"  yellow-white NOT named (task {tid}): "
              f"grasped FIRST in {c}/{len(recs)} episodes")
    return out


def stop_condition(by_task):
    print("\n" + "=" * 78)
    print("STOP / PROCEED")
    print("=" * 78)
    # Is the first-grasp identity independent of which mug is named?
    firsts = {}
    for tid, recs in sorted(by_task.items()):
        c = collections.Counter(
            r["grasp_order"][0] if r["grasp_order"] else None for r in recs
        )
        firsts[tid] = c
        tgt = recs[0]["target"]
        print(f"  task {tid} (names {lab(tgt):<12}): first grasp = "
              + ", ".join(f"{lab(o)} {n}/{len(recs)}" for o, n in c.most_common()))

    all_first = collections.Counter()
    for c in firsts.values():
        all_first.update(c)
    dominant, dom_n = all_first.most_common(1)[0]
    total = sum(all_first.values())
    print(f"\n  Pooled across all tasks: first grasp is {lab(dominant)} in "
          f"{dom_n}/{total} episodes ({100*dom_n/total:.0f}%), "
          f"regardless of which mug the instruction names.")

    # Does the dominant first grasp hold even when another mug is named?
    violating = [
        tid for tid, recs in by_task.items()
        if recs[0]["target"] != dominant
        and firsts[tid].most_common(1)[0][0] == dominant
    ]
    print(f"  Tasks where {lab(dominant)} is grasped first despite NOT being named: "
          f"{sorted(violating)}")
    return dominant, violating


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/scratch1/nalagand/athena_robotics/"
                                      "results/living5_char/episodes.jsonl")
    ap.add_argument("--white-task", type=int, default=67)
    ap.add_argument("--yellow-task", type=int, default=68)
    ap.add_argument("--red-tasks", type=int, nargs="*", default=[65, 66])
    args = ap.parse_args()

    by_task = load(args.path)
    section_a(by_task)
    section_b(by_task, args.white_task, args.yellow_task)
    section_c(by_task, args.yellow_task, [args.white_task] + args.red_tasks)
    stop_condition(by_task)


if __name__ == "__main__":
    main()
