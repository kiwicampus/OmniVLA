#!/usr/bin/env python3
"""Create compact clean-episode manifests for LeRobot-style datasets."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnivla_training.episode_manifest import collapse_episode_ranges, dump_structured_file, expand_episode_ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one compact episode manifest by combining manual exclusions "
            "with optional video timestamp validation."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo-id", help="Hugging Face dataset repo id, e.g. my_org/my_dataset")
    source.add_argument("--root", type=Path, help="Local dataset snapshot root")
    parser.add_argument("--revision", default="main", help="Dataset revision when using --repo-id")
    parser.add_argument("--local-files-only", action="store_true", help="Use only local Hugging Face cache files")
    parser.add_argument("--video-key", default="observation.image.main", help="Video key in LeRobot metadata")
    parser.add_argument("--output", type=Path, required=True, help="Output .yaml/.yml or .json manifest")

    parser.add_argument(
        "--check-video-duration",
        action="store_true",
        help="Read MP4 durations and exclude episodes whose metadata points past the real video duration",
    )
    parser.add_argument(
        "--overflow-threshold-s",
        type=float,
        default=0.5,
        help="Allowed timestamp overflow in seconds before excluding an episode",
    )
    parser.add_argument(
        "--max-video-files",
        type=int,
        default=None,
        help="Debug option: validate only the first N grouped video files",
    )
    parser.add_argument(
        "--keep-unreadable-videos",
        action="store_true",
        help="Report unreadable/missing videos without excluding their episodes",
    )

    parser.add_argument("--exclude-episode", type=int, nargs="*", default=[], help="Episode ids to exclude")
    parser.add_argument(
        "--exclude-episode-range",
        type=int,
        nargs=2,
        action="append",
        default=[],
        metavar=("START", "END"),
        help="Inclusive episode range to exclude; can be repeated",
    )
    parser.add_argument(
        "--exclude-file-index",
        type=int,
        nargs="*",
        default=[],
        help="Exclude every episode mapped to these file_index values, across chunks",
    )
    parser.add_argument(
        "--exclude-video",
        action="append",
        default=[],
        metavar="CHUNK:FILE_INDEX",
        help="Exclude one concrete video file, for example chunk-000:33; can be repeated",
    )
    return parser.parse_args()


def parse_video_ref(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise ValueError(f"Invalid --exclude-video `{value}`. Expected CHUNK:FILE_INDEX, e.g. chunk-000:33")
    chunk, file_index = value.split(":", 1)
    chunk = chunk.strip()
    if not chunk:
        raise ValueError(f"Invalid --exclude-video `{value}`: empty chunk")
    return chunk, int(file_index)


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.root is not None:
        return args.root.resolve()

    from huggingface_hub import snapshot_download

    allow_patterns = ["meta/episodes/**/*.parquet"]
    if args.check_video_duration:
        allow_patterns.append(f"videos/{args.video_key}/**/*.mp4")

    return Path(
        snapshot_download(
            args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_files_only=args.local_files_only,
            allow_patterns=allow_patterns,
        )
    )


def load_episode_tables(root: Path) -> Any:
    import pandas as pd

    frames = []
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["video_chunk"] = path.parent.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No parquet files found under {root / 'meta' / 'episodes'}")

    return pd.concat(frames, ignore_index=True)


def require_columns(frame: Any, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset metadata is missing required column(s): {', '.join(missing)}")


def video_duration_seconds(video_path: Path) -> tuple[float | None, int | None, float | None, str | None]:
    try:
        import av

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif stream.frames and stream.average_rate:
                duration = float(stream.frames / stream.average_rate)
            else:
                duration = None
            frames = stream.frames if stream.frames else None
            fps = float(stream.average_rate) if stream.average_rate else None
        return duration, frames, fps, None
    except Exception as exc:  # pragma: no cover - depends on local codecs/video files.
        return None, None, None, repr(exc)


def build_video_path(root: Path, video_key: str, chunk: str, file_index: int) -> Path:
    return root / "videos" / video_key / chunk / f"file-{file_index:03d}.mp4"


def find_timestamp_overflow_episodes(
    root: Path,
    episodes: Any,
    video_key: str,
    overflow_threshold_s: float,
    max_video_files: int | None,
    exclude_unreadable_videos: bool,
) -> tuple[set[int], list[dict[str, Any]]]:
    file_col = f"videos/{video_key}/file_index"
    from_col = f"videos/{video_key}/from_timestamp"
    to_col = f"videos/{video_key}/to_timestamp"
    require_columns(episodes, [file_col, from_col, to_col, "video_chunk", "episode_index"])

    excluded: set[int] = set()
    bad_videos: list[dict[str, Any]] = []

    grouped = episodes.groupby(["video_chunk", file_col], sort=True)
    for count, ((chunk, raw_file_index), subset) in enumerate(grouped, start=1):
        if max_video_files is not None and count > max_video_files:
            break

        file_index = int(raw_file_index)
        video_path = build_video_path(root, video_key, chunk, file_index)
        duration, frames, fps, error = video_duration_seconds(video_path)

        if duration is None:
            bad_episode_ids = sorted(subset["episode_index"].astype(int).tolist()) if exclude_unreadable_videos else []
            excluded.update(bad_episode_ids)
            bad_videos.append(
                {
                    "chunk": str(chunk),
                    "file_index": file_index,
                    "path": str(video_path.relative_to(root)) if video_path.exists() else str(video_path),
                    "status": "unreadable",
                    "error": error,
                    "episode_count": int(len(subset)),
                    "excluded_episode_count": len(bad_episode_ids),
                    "excluded_episode_ranges": collapse_episode_ranges(bad_episode_ids),
                }
            )
            continue

        bad_subset = subset.loc[subset[to_col] > duration + overflow_threshold_s]
        if bad_subset.empty:
            continue

        bad_episode_ids = sorted(bad_subset["episode_index"].astype(int).tolist())
        excluded.update(bad_episode_ids)
        max_to = float(subset[to_col].max())
        bad_videos.append(
            {
                "chunk": str(chunk),
                "file_index": file_index,
                "path": str(video_path.relative_to(root)),
                "status": "timestamp_overflow",
                "episode_count": int(len(subset)),
                "duration_s": round(duration, 4),
                "frames": frames,
                "fps": fps,
                "max_to_timestamp_s": round(max_to, 4),
                "max_overflow_s": round(max_to - duration, 4),
                "excluded_episode_count": len(bad_episode_ids),
                "excluded_episode_ranges": collapse_episode_ranges(bad_episode_ids),
            }
        )

    return excluded, bad_videos


def main() -> int:
    args = parse_args()
    root = resolve_dataset_root(args)
    episodes = load_episode_tables(root)
    require_columns(episodes, ["episode_index"])

    all_episodes = sorted({int(ep) for ep in episodes["episode_index"].astype(int).tolist()})
    excluded_by_reason: dict[str, set[int]] = {}

    manual_episode_ids = set(int(ep) for ep in args.exclude_episode)
    manual_episode_ids.update(expand_episode_ranges(args.exclude_episode_range))
    excluded_by_reason["manual_episode"] = manual_episode_ids

    file_col = f"videos/{args.video_key}/file_index"
    manual_file_ids: set[int] = set()
    if args.exclude_file_index:
        require_columns(episodes, [file_col])
        manual_file_ids.update(
            episodes.loc[episodes[file_col].isin(args.exclude_file_index), "episode_index"].astype(int).tolist()
        )
    excluded_by_reason["manual_file_index"] = manual_file_ids

    manual_video_ids: set[int] = set()
    manual_video_refs = [parse_video_ref(value) for value in args.exclude_video]
    if manual_video_refs:
        require_columns(episodes, [file_col, "video_chunk"])
        for chunk, file_index in manual_video_refs:
            mask = (episodes["video_chunk"] == chunk) & (episodes[file_col] == file_index)
            manual_video_ids.update(episodes.loc[mask, "episode_index"].astype(int).tolist())
    excluded_by_reason["manual_video"] = manual_video_ids

    bad_videos: list[dict[str, Any]] = []
    if args.check_video_duration:
        timestamp_ids, bad_videos = find_timestamp_overflow_episodes(
            root=root,
            episodes=episodes,
            video_key=args.video_key,
            overflow_threshold_s=float(args.overflow_threshold_s),
            max_video_files=args.max_video_files,
            exclude_unreadable_videos=not args.keep_unreadable_videos,
        )
        excluded_by_reason["timestamp_overflow"] = timestamp_ids

    excluded_episodes = sorted(set().union(*excluded_by_reason.values()))
    clean_episodes = sorted(set(all_episodes) - set(excluded_episodes))

    payload = {
        "kind": "omnivla_episode_manifest",
        "schema_version": 1,
        "repo_id": args.repo_id,
        "dataset_root": None if args.repo_id else str(root),
        "revision": args.revision if args.repo_id else None,
        "video_key": args.video_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "command": " ".join(sys.argv),
            "check_video_duration": bool(args.check_video_duration),
            "overflow_threshold_s": float(args.overflow_threshold_s),
            "max_video_files": args.max_video_files,
        },
        "manual_rules": {
            "exclude_episode_ranges": collapse_episode_ranges(manual_episode_ids),
            "exclude_file_indices": sorted({int(value) for value in args.exclude_file_index}),
            "exclude_videos": [
                {"chunk": chunk, "file_index": file_index}
                for chunk, file_index in manual_video_refs
            ],
        },
        "summary": {
            "total_episode_count": len(all_episodes),
            "clean_episode_count": len(clean_episodes),
            "excluded_episode_count": len(excluded_episodes),
        },
        "clean_episode_ranges": collapse_episode_ranges(clean_episodes),
        "excluded_episode_ranges": collapse_episode_ranges(excluded_episodes),
        "excluded_by_reason": {
            reason: collapse_episode_ranges(values)
            for reason, values in excluded_by_reason.items()
            if values
        },
        "bad_videos": bad_videos,
    }

    dump_structured_file(args.output, payload)
    print(
        f"Wrote {args.output}: {len(clean_episodes)} clean / "
        f"{len(excluded_episodes)} excluded / {len(all_episodes)} total episodes"
    )
    if bad_videos:
        print(f"Detected {len(bad_videos)} bad video file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
