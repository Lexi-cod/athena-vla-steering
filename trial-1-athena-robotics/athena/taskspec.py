"""Standalone BDDL task parsing for LIBERO.

We reuse LIBERO's own parser (`robosuite_parse_problem`) rather than
reimplementing it, so object/goal naming always matches what the simulator
actually instantiates. This module adds the layer we care about on top:
which objects are *distractors* for a given instruction, and which tasks are
therefore candidates for wrong-object confusion.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import re

from libero.libero import get_libero_path
from libero.libero.envs import bddl_utils

# Tokens that carry no discriminative meaning when matching an instruction
# phrase against a BDDL object name.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "on", "in", "into", "onto", "to", "at", "and",
        "it", "its", "up", "down", "put", "pick", "place", "move", "push",
        "pull", "open", "close", "turn", "off", "that", "is", "with", "from",
        "then", "please", "top", "bottom", "side",
    }
)


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with trailing object indices stripped."""
    text = re.sub(r"_(\d+)$", "", text)
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOPWORDS}


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """Everything we need to know about a LIBERO task, parsed from its BDDL."""

    bddl_path: str
    language: str
    objects: tuple[str, ...]          # instantiated object names, e.g. butter_1
    fixtures: tuple[str, ...]
    obj_of_interest: tuple[str, ...]  # ground-truth targets named by the BDDL
    goal_predicates: tuple[tuple[str, ...], ...]
    # category -> instances, taken straight from the BDDL (:objects) block.
    categories: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def name(self) -> str:
        return pathlib.Path(self.bddl_path).stem

    @property
    def scene(self) -> str:
        """e.g. KITCHEN_SCENE10 from the filename prefix."""
        m = re.match(r"([A-Z_]+SCENE\d+)", self.name)
        return m.group(1) if m else "UNKNOWN"

    # -- confusion analysis -------------------------------------------------

    @functools.cached_property
    def category_groups(self) -> dict[str, tuple[str, ...]]:
        """Instantiated objects grouped by BDDL category.

        `butter_1`, `butter_2` -> {"butter": ("butter_1", "butter_2")}
        """
        return dict(self.categories)

    @functools.cached_property
    def duplicate_categories(self) -> tuple[str, ...]:
        """Categories with >1 instance in the scene — same-category distractors."""
        return tuple(k for k, v in self.category_groups.items() if len(v) > 1)

    @functools.cached_property
    def distractors(self) -> tuple[str, ...]:
        """Manipulable objects that are not the target of this task."""
        return tuple(o for o in self.objects if o not in self.obj_of_interest)

    @functools.cached_property
    def confusable_distractors(self) -> tuple[str, ...]:
        """Distractors sharing >=1 content token with a target object.

        These are the objects a policy is most likely to grasp by mistake:
        a second butter, a second bowl, a bowl of a different colour, etc.
        """
        target_toks: set[str] = set()
        for t in self.obj_of_interest:
            target_toks |= content_tokens(t)
        if not target_toks:
            return ()
        return tuple(
            d for d in self.distractors if content_tokens(d) & target_toks
        )

    @functools.cached_property
    def confusion_score(self) -> int:
        """Crude ranking of wrong-object risk. Higher = more confusable."""
        return 2 * len(self.confusable_distractors) + len(self.duplicate_categories)


def parse_bddl(bddl_path: str | pathlib.Path) -> TaskSpec:
    """Parse a single BDDL file into a TaskSpec."""
    bddl_path = str(bddl_path)
    problem = bddl_utils.robosuite_parse_problem(bddl_path)

    # LIBERO returns objects as {category: [instance, ...]} in `fixtures`/`objects`.
    def _flatten(d) -> tuple[str, ...]:
        if isinstance(d, dict):
            return tuple(name for names in d.values() for name in names)
        return tuple(d)

    objects = problem.get("objects", {}) or {}
    goal_state = problem.get("goal_state", []) or []

    # `language_instruction` comes back as a token list, not a string.
    language = problem.get("language_instruction", "")
    if isinstance(language, (list, tuple)):
        language = " ".join(language)

    return TaskSpec(
        bddl_path=bddl_path,
        language=language.strip(),
        objects=_flatten(objects),
        fixtures=_flatten(problem.get("fixtures", {})),
        obj_of_interest=tuple(problem.get("obj_of_interest", [])),
        goal_predicates=tuple(tuple(p) for p in goal_state),
        categories=tuple(
            (cat, tuple(insts)) for cat, insts in sorted(objects.items())
        ),
    )


def suite_bddl_dir(task_suite_name: str = "libero_90") -> pathlib.Path:
    return pathlib.Path(get_libero_path("bddl_files")) / task_suite_name


def parse_suite(task_suite_name: str = "libero_90") -> list[TaskSpec]:
    """Parse every BDDL in a suite, sorted by filename for stable ordering."""
    return [
        parse_bddl(p) for p in sorted(suite_bddl_dir(task_suite_name).glob("*.bddl"))
    ]
