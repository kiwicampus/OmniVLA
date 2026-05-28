"""Utilities for compact episode subset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def collapse_episode_ranges(values: Iterable[int]) -> list[list[int]]:
    episodes = sorted({int(value) for value in values})
    if not episodes:
        return []

    ranges: list[list[int]] = []
    start = previous = episodes[0]
    for episode in episodes[1:]:
        if episode == previous + 1:
            previous = episode
            continue
        ranges.append([start, previous])
        start = previous = episode
    ranges.append([start, previous])
    return ranges


def expand_episode_ranges(ranges: Iterable[Iterable[int]]) -> list[int]:
    episodes: list[int] = []
    for raw_range in ranges:
        values = list(raw_range)
        if len(values) != 2:
            raise ValueError(f"Episode ranges must have exactly two values, got {values!r}")
        start, end = int(values[0]), int(values[1])
        if end < start:
            raise ValueError(f"Invalid episode range [{start}, {end}]")
        episodes.extend(range(start, end + 1))
    return sorted(set(episodes))


def load_structured_file(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def dump_structured_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_episode_subset(path: Path) -> list[int]:
    payload = load_structured_file(path)

    if isinstance(payload, list):
        return sorted({int(ep) for ep in payload})

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported episode subset format in {path}")

    if "clean_episodes" in payload:
        return sorted({int(ep) for ep in payload["clean_episodes"]})
    if "episodes" in payload:
        return sorted({int(ep) for ep in payload["episodes"]})
    if "clean_episode_ranges" in payload:
        return expand_episode_ranges(payload["clean_episode_ranges"])
    if "episode_ranges" in payload:
        return expand_episode_ranges(payload["episode_ranges"])

    raise ValueError(
        f"Unsupported episode subset format in {path}. Expected `clean_episodes`, "
        "`episodes`, `clean_episode_ranges`, or `episode_ranges`."
    )
