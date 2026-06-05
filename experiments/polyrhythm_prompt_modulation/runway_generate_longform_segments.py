#!/usr/bin/env python3
"""Generate segmented Runway videos for the 180 second longform version."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_SEGMENTS = EXPERIMENT / "data" / "longform_runway_segments.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "polyrhythm_prompt_modulation" / "longform" / "runway_segments"
API_BASE = "https://api.dev.runwayml.com/v1"
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED"}


def api_key() -> str:
    key = os.environ.get("RUNWAYML_API_SECRET") or os.environ.get("RUNWAY_API_KEY")
    if not key:
        raise SystemExit("RUNWAYML_API_SECRET or RUNWAY_API_KEY is not set. Run: source ${HOME}/.config/secrets.zsh")
    return key


def request_json(method: str, path: str, key: str, api_version: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {key}",
        "X-Runway-Version": api_version,
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{API_BASE}/{path.lstrip('/')}", data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Runway API HTTP {exc.code}: {detail}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_id(response: dict[str, Any]) -> str:
    value = response.get("id") or response.get("taskId") or response.get("task_id")
    if not value:
        raise RuntimeError(f"Runway response did not include a task id: {response}")
    return str(value)


def output_url(task: dict[str, Any]) -> str | None:
    output = task.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("url") or first.get("uri")
    return task.get("url")


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as response:
        path.write_bytes(response.read())


def build_payload(segment: dict[str, Any], model: str, ratio: str, duration: int) -> dict[str, Any]:
    return {
        "model": model,
        "promptText": segment["promptText"],
        "ratio": ratio,
        "duration": duration,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="Submit billable Runway jobs.")
    parser.add_argument("--prepare", action="store_true", help="Write payloads only.")
    parser.add_argument("--model", default=os.environ.get("RUNWAY_MODEL", "gen4.5"))
    parser.add_argument("--ratio", default=os.environ.get("RUNWAY_RATIO", "1280:720"))
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--api-version", default=os.environ.get("RUNWAY_API_VERSION", "2024-11-06"))
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--poll-timeout", type=float, default=60 * 45)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    segments = json.loads(args.segments.read_text(encoding="utf-8"))["segments"]
    manifest: dict[str, Any] = {
        "schema": "mrt-longform-runway-segments-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segments_path": str(args.segments),
        "model": args.model,
        "ratio": args.ratio,
        "duration": args.duration,
        "segments": [],
    }

    for segment in segments:
        payload = build_payload(segment, args.model, args.ratio, args.duration)
        payload_path = args.out_dir / f"segment_{segment['index']:02d}.payload.json"
        write_json(payload_path, payload)

    if args.prepare or not args.execute:
        write_json(args.out_dir / "longform_runway_manifest.json", manifest)
        print(f"Prepared {len(segments)} payloads in {args.out_dir}")
        return 0

    key = api_key()
    active: list[dict[str, Any]] = []
    for segment in segments:
        payload_path = args.out_dir / f"segment_{segment['index']:02d}.payload.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        create_path = args.out_dir / f"segment_{segment['index']:02d}.create.json"
        if create_path.exists():
            create = json.loads(create_path.read_text(encoding="utf-8"))
        else:
            create = request_json("POST", "/text_to_video", key, args.api_version, payload)
            write_json(create_path, create)
        active.append({"segment": segment, "task_id": task_id(create), "payload_path": payload_path, "create_path": create_path})
        print(f"submitted segment {segment['index']:02d}: {active[-1]['task_id']}")

    pending = {item["task_id"]: item for item in active}
    deadline = time.time() + args.poll_timeout
    while pending and time.time() < deadline:
        for task, item in list(pending.items()):
            status_path = args.out_dir / f"segment_{item['segment']['index']:02d}.task.json"
            detail = request_json("GET", f"/tasks/{task}", key, args.api_version)
            write_json(status_path, detail)
            status = str(detail.get("status") or "").upper()
            print(f"segment {item['segment']['index']:02d}: {status or 'UNKNOWN'}")
            if status in TERMINAL_STATUSES:
                if status != "SUCCEEDED":
                    raise RuntimeError(f"Runway task failed for segment {item['segment']['index']:02d}: {detail}")
                url = output_url(detail)
                if not url:
                    raise RuntimeError(f"No output URL for segment {item['segment']['index']:02d}: {detail}")
                clip_path = args.out_dir / f"segment_{item['segment']['index']:02d}.mp4"
                if not clip_path.exists():
                    download(url, clip_path)
                manifest["segments"].append(
                    {
                        **item["segment"],
                        "task_id": task,
                        "payload_path": str(item["payload_path"]),
                        "task_path": str(status_path),
                        "clip_path": str(clip_path),
                    }
                )
                pending.pop(task)
        if pending:
            time.sleep(args.poll_interval)

    if pending:
        raise TimeoutError(f"Timed out with {len(pending)} Runway tasks still pending")

    manifest["segments"] = sorted(manifest["segments"], key=lambda item: item["index"])
    write_json(args.out_dir / "longform_runway_manifest.json", manifest)
    print(args.out_dir / "longform_runway_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
