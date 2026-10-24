#!/usr/bin/env python3
"""Compute SR and stage-aggregated PSR from EgoVLA result_log.txt files.

Usage: python eval/egovla/stat_success_rate.py output/egovla/skel_wam
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TASK_PSR_FIELDS = {
    "Stack-Can": ("place_success",),
    "Push-Box": ("reach_success",),
    "Open-Drawer": ("reach_success", "move_door_success"),
    "Close-Drawer": ("reach_success", "move_door_success"),
    "Flip-Mug": ("reach_success",),
    "Pour-Balls": ("reach_bottle_success", "lift_success"),
    "Open-Laptop": ("move_lid_success",),
}

HEADER_RE = re.compile(
    r"^Task:\s*(?P<task>[^,]+).*?"
    r"Room Idx:\s*(?P<room>\d+).*?"
    r"Table Idx:\s*(?P<table>\d+)",
    re.MULTILINE,
)
VALUE_RE = re.compile(r"\b(?P<name>[A-Za-z_]*success):\s*(?P<value>[-+]?\d+(?:\.\d+)?)")


@dataclass
class Counts:
    episodes: int = 0
    sr_successes: int = 0
    psr_successes: int = 0
    psr_total: int = 0

    def add(self, success: bool, psr_values: list[bool]) -> None:
        self.episodes += 1
        self.sr_successes += int(success)
        self.psr_successes += sum(psr_values)
        self.psr_total += len(psr_values)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def parse_log(path: Path, expected_task: str) -> list[tuple[int, int, bool, list[bool]]]:
    text = path.read_text(encoding="utf-8")
    headers = list(HEADER_RE.finditer(text))
    records: list[tuple[int, int, bool, list[bool]]] = []

    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[header.start():end]
        logged_task = header.group("task").strip()
        if logged_task != expected_task:
            warn(f"{path}: record {index + 1} says task {logged_task!r}, expected {expected_task!r}")

        values = {match.group("name"): float(match.group("value")) for match in VALUE_RE.finditer(block)}
        if "success" not in values:
            warn(f"{path}: record {index + 1} has no success field; skipped")
            continue

        missing = [field for field in TASK_PSR_FIELDS[expected_task] if field not in values]
        if missing:
            warn(f"{path}: record {index + 1} missing {', '.join(missing)}; skipped")
            continue

        success = values["success"] > 0.5
        # A fully successful episode implies that every partial-success stage
        # succeeded, even when a stale/inconsistent detail flag says otherwise.
        psr_values = [
            True if success else values[field] > 0.5
            for field in TASK_PSR_FIELDS[expected_task]
        ]
        records.append((int(header.group("room")), int(header.group("table")), success, psr_values))

    return records


def ratio(successes: int, total: int) -> str:
    if total == 0:
        return "0/0 (N/A)"
    return f"{successes}/{total} ({100.0 * successes / total:.2f}%)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report seen, unseen, and overall SR/PSR for seven EgoVLA tasks."
    )
    parser.add_argument("input_dir", type=Path, help="directory containing task result_log.txt files")
    args = parser.parse_args()

    root = args.input_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"input directory does not exist: {root}")

    log_paths = sorted(root.rglob("result_log.txt"))
    if not log_paths:
        parser.error(f"no result_log.txt found under: {root}")

    counts: dict[str, dict[str, Counts]] = defaultdict(
        lambda: {split: Counts() for split in ("seen", "unseen", "overall")}
    )
    found_tasks: set[str] = set()

    for path in log_paths:
        tasks = {m.group("task").strip() for m in HEADER_RE.finditer(path.read_text(encoding="utf-8"))}
        if len(tasks) != 1 or not tasks.issubset(TASK_PSR_FIELDS):
            warn(f"{path}: expected one supported task per log, got {sorted(tasks)}; skipped")
            continue
        task = next(iter(tasks))
        found_tasks.add(task)
        records = parse_log(path, task)
        if len(records) != 93:
            warn(f"{path}: parsed {len(records)} records, expected 93")

        for room, table, success, psr_values in records:
            split = "seen" if room in (1, 2, 3) and table == 1 else "unseen"
            counts[task][split].add(success, psr_values)
            counts[task]["overall"].add(success, psr_values)

    missing_tasks = [task for task in TASK_PSR_FIELDS if task not in found_tasks]
    if missing_tasks:
        warn(f"no log found for: {', '.join(missing_tasks)}")

    print(f"Input: {root}")
    print(f"{'Task':<14} {'Split':<8} {'SR':>20} {'PSR':>20}")
    print("-" * 66)
    for task in TASK_PSR_FIELDS:
        if task not in found_tasks:
            continue
        for split in ("seen", "unseen", "overall"):
            item = counts[task][split]
            print(
                f"{task:<14} {split:<8} "
                f"{ratio(item.sr_successes, item.episodes):>20} "
                f"{ratio(item.psr_successes, item.psr_total):>20}"
            )

        overall = counts[task]["overall"]
        if overall.episodes != 93:
            warn(f"{task}: overall has {overall.episodes} records, expected 93")
        if counts[task]["seen"].episodes != 27 or counts[task]["unseen"].episodes != 66:
            warn(
                f"{task}: seen/unseen counts are "
                f"{counts[task]['seen'].episodes}/{counts[task]['unseen'].episodes}, expected 27/66"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
