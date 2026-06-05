#!/usr/bin/env python3
"""Generate a base Runway landscape and four Aleph variants."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_MODULATION = EXPERIMENT / "data" / "modulation_10s_104bpm.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "polyrhythm_prompt_modulation" / "runway"
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


def poll(task_id: str, key: str, api_version: str, status_path: Path, interval: float, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = request_json("GET", f"/tasks/{task_id}", key, api_version)
        write_json(status_path, last)
        status = str(last.get("status") or "").upper()
        print(f"{task_id} {status or 'UNKNOWN'}")
        if status in TERMINAL_STATUSES:
            return last
        time.sleep(interval)
    raise TimeoutError(f"Timed out polling Runway task {task_id}: {last}")


def task_id(response: dict[str, Any]) -> str:
    value = response.get("id") or response.get("taskId") or response.get("task_id")
    if not value:
        raise RuntimeError(f"Runway response did not include a task id: {response}")
    return str(value)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as response:
        path.write_bytes(response.read())


def build_jobs(modulation: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_payload = {
        "model": args.base_model,
        "promptText": modulation["base_runway_prompt"],
        "ratio": args.ratio,
        "duration": args.duration,
    }
    aleph_jobs = []
    for index, slot in enumerate(modulation["slots"], start=1):
        aleph_jobs.append(
            {
                "slot": slot["id"],
                "index": index,
                "label": slot["label"],
                "payload": {
                    "model": args.aleph_model,
                    "videoUri": "__BASE_VIDEO_URI__",
                    "promptText": slot["video_prompt"],
                    "ratio": args.ratio,
                },
            }
        )
    return base_payload, aleph_jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulation", type=Path, default=DEFAULT_MODULATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="Submit billable Runway tasks.")
    parser.add_argument("--prepare", action="store_true", help="Write payloads only.")
    parser.add_argument("--base-model", default=os.environ.get("RUNWAY_BASE_MODEL", "gen4.5"))
    parser.add_argument("--aleph-model", default=os.environ.get("RUNWAY_ALEPH_MODEL", "gen4_aleph"))
    parser.add_argument("--ratio", default=os.environ.get("RUNWAY_RATIO", "1280:720"))
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--poll-timeout", type=float, default=60 * 30)
    parser.add_argument("--api-version", default=os.environ.get("RUNWAY_API_VERSION", "2024-11-06"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    modulation = json.loads(args.modulation.read_text(encoding="utf-8"))
    base_payload, aleph_jobs = build_jobs(modulation, args)
    write_json(args.out_dir / "base_landscape.payload.json", base_payload)
    for job in aleph_jobs:
        write_json(args.out_dir / f"slot_{job['index']:02d}_{job['slot']}.payload.template.json", job["payload"])

    if args.prepare or not args.execute:
        print(f"Prepared payloads in {args.out_dir}")
        return 0

    key = api_key()
    manifest: dict[str, Any] = {
        "schema": "mrt-polyrhythm-runway-videos-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "modulation": str(args.modulation),
        "base": {},
        "slots": [],
    }

    base_create = request_json("POST", "/text_to_video", key, args.api_version, base_payload)
    write_json(args.out_dir / "base_landscape.create.json", base_create)
    base_task_id = task_id(base_create)
    base_task = poll(
        base_task_id,
        key,
        args.api_version,
        args.out_dir / "base_landscape.task.json",
        args.poll_interval,
        args.poll_timeout,
    )
    if str(base_task.get("status") or "").upper() != "SUCCEEDED":
        raise RuntimeError(f"Base Runway task failed: {base_task}")
    base_url = output_url(base_task)
    if not base_url:
        raise RuntimeError(f"Base Runway task has no output URL: {base_task}")
    base_path = args.out_dir / "base_landscape.mp4"
    download(base_url, base_path)
    manifest["base"] = {
        "task_id": base_task_id,
        "payload_path": "base_landscape.payload.json",
        "task_path": "base_landscape.task.json",
        "clip_path": str(base_path),
        "output_url_used_for_aleph": base_url,
    }

    pending: list[dict[str, Any]] = []
    for job in aleph_jobs:
        payload = {**job["payload"], "videoUri": base_url}
        payload_path = args.out_dir / f"slot_{job['index']:02d}_{job['slot']}.payload.json"
        write_json(payload_path, payload)
        create = request_json("POST", "/video_to_video", key, args.api_version, payload)
        create_path = args.out_dir / f"slot_{job['index']:02d}_{job['slot']}.create.json"
        write_json(create_path, create)
        pending.append({**job, "task_id": task_id(create), "payload_path": payload_path, "create_path": create_path})
        print(f"submitted {job['slot']}: {pending[-1]['task_id']}")

    for job in pending:
        task_path = args.out_dir / f"slot_{job['index']:02d}_{job['slot']}.task.json"
        task = poll(job["task_id"], key, args.api_version, task_path, args.poll_interval, args.poll_timeout)
        if str(task.get("status") or "").upper() != "SUCCEEDED":
            raise RuntimeError(f"Aleph task failed for {job['slot']}: {task}")
        url = output_url(task)
        if not url:
            raise RuntimeError(f"Aleph task has no output URL for {job['slot']}: {task}")
        clip_path = args.out_dir / f"slot_{job['index']:02d}_{job['slot']}.mp4"
        download(url, clip_path)
        manifest["slots"].append(
            {
                "slot": job["slot"],
                "label": job["label"],
                "task_id": job["task_id"],
                "payload_path": str(job["payload_path"]),
                "task_path": str(task_path),
                "clip_path": str(clip_path),
            }
        )

    write_json(args.out_dir / "runway_video_manifest.json", manifest)
    print(args.out_dir / "runway_video_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
