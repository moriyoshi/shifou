#!/usr/bin/env python3
"""Publish a Qwen session, then measure fresh and cached continuations over a peer socket."""

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import signal
import statistics
import subprocess
import sys
import time


EXPERIMENT = Path(__file__).resolve().parent
YESNO = (EXPERIMENT / "../../../yesno").resolve()
METRICS = (
    "prepared_fresh_total_ms",
    "prepared_hit_total_ms",
    "prepared_host_hit_ms",
    "session_fresh_total_ms",
    "session_hit_total_ms",
    "session_host_hit_ms",
    "prepared_read_ms",
    "session_read_ms",
    "prepared_gpu_restore_ms",
    "session_gpu_restore_ms",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="local Qwen3-0.6B model directory")
    parser.add_argument("corpus", type=Path, help="text corpus to tokenize")
    parser.add_argument("output_dir", type=Path, help="new directory for database and results")
    parser.add_argument("--prefix-tokens", type=int, default=1980)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--yesnod", type=Path, default=YESNO / "target/release/yesnod")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if args.prefix_tokens <= 0 or args.samples <= 0:
        parser.error("prefix tokens and samples must be positive")
    for path in (args.model_dir / "config.json", args.model_dir / "model.safetensors", args.model_dir / "tokenizer.json", args.corpus, args.yesnod):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    return args


def run_logged(command: list[str], log: Path, env: dict[str, str]) -> float:
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(command, cwd=EXPERIMENT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    return (time.monotonic() - start) * 1000


def wait_for_peer(socket: Path, server: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"yesnod exited with status {server.returncode} before peer readiness")
        if socket.exists():
            return
        time.sleep(0.05)
    raise TimeoutError("yesnod peer socket was not ready within 30 seconds")


def checked_result(path: Path, *, peer: bool) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        ("prepared_logit_diff", "session_logit_diff")
        if peer
        else (
            "prepared_max_abs_logit_diff",
            "session_max_abs_logit_diff",
            "generation_two_max_abs_logit_diff",
        )
    )
    for key in keys:
        difference = result[key]
        if not math.isfinite(difference) or difference > 1e-3:
            raise RuntimeError(f"accuracy check failed in {path}: {key}={difference}")
    if peer and not result["external_yesnod"]:
        raise RuntimeError("peer probe did not use the separate yesnod process")
    if not peer and result["verification"] != "Full":
        raise RuntimeError("publisher did not use full verification")
    return result


def main() -> int:
    args = arguments()
    if not args.skip_build:
        subprocess.run(
            ["cargo", "build", "--offline", "--release", "--bin", "session_workflows", "--bin", "peer_session_workflows"],
            cwd=EXPERIMENT,
            check=True,
        )
    binary_dir = (EXPERIMENT / "../../.agents-workspace/tmp/target/release").resolve()
    publisher = binary_dir / "session_workflows"
    reader = binary_dir / "peer_session_workflows"
    for binary in (publisher, reader):
        if not binary.is_file():
            raise FileNotFoundError(binary)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    database = output / "database"
    publish_json = output / "publication.json"
    env = os.environ.copy()
    env.pop("SESSION_TRUSTED_LOCAL", None)
    publication_wall_ms = run_logged(
        [str(publisher), str(args.model_dir.resolve()), str(args.corpus.resolve()), str(database), str(publish_json), str(args.prefix_tokens)],
        output / "publication.log",
        env,
    )
    publication = checked_result(publish_json, peer=False)

    socket_root = YESNO / ".agents-workspace/tmp"
    socket_root.mkdir(parents=True, exist_ok=True)
    socket = socket_root / f"e2e-{os.getpid()}-{secrets.token_hex(4)}.sock"
    if len(os.fsencode(socket)) >= 108:
        raise RuntimeError("peer socket path exceeds Unix address limit")
    config = output / "yesnod.toml"
    config.write_text(
        "[server]\n"
        "role = \"leader\"\n"
        f"data_dir = {json.dumps(str(database / 'data'))}\n"
        "[server.flight]\nlisten = \"127.0.0.1:0\"\n"
        "[server.metrics]\nlisten = \"127.0.0.1:0\"\n"
        "[db]\nshards = 1\n"
        "[plugin]\n"
        f"channel_socket = {json.dumps(str(socket))}\n"
        "channel_max_lanes = 16\n",
        encoding="utf-8",
    )
    peer_env = {**env, "PEER_SOCKET": str(socket)}
    readings = []
    peer_wall_ms = []
    with (output / "yesnod.log").open("w", encoding="utf-8") as log:
        server_start = time.monotonic()
        server = subprocess.Popen(
            [str(args.yesnod.resolve()), "--config", str(config)],
            cwd=YESNO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_peer(socket, server)
            server_start_ms = (time.monotonic() - server_start) * 1000
            for index in range(args.samples):
                result = output / f"peer-{index + 1:02d}.json"
                peer_wall_ms.append(run_logged(
                    [str(reader), str(args.model_dir.resolve()), str(database), str(result), str(args.prefix_tokens)],
                    output / f"peer-{index + 1:02d}.log",
                    peer_env,
                ))
                readings.append(checked_result(result, peer=True))
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            if socket.exists():
                socket.unlink()
    if server.returncode != 0:
        raise RuntimeError(f"yesnod exited with status {server.returncode}")

    medians = {key: statistics.median(reading[key] for reading in readings) for key in METRICS}
    maximum_logit_difference = max(
        reading[key]
        for reading in readings
        for key in ("prepared_logit_diff", "session_logit_diff")
    )
    summary = {
        "benchmark": "qwen-session-e2e-peer/v1",
        "model_dir": str(args.model_dir.resolve()),
        "corpus": str(args.corpus.resolve()),
        "prefix_tokens": args.prefix_tokens,
        "samples": args.samples,
        "publication": publication,
        "process_wall_ms": {
            "publication": publication_wall_ms,
            "yesnod_start": server_start_ms,
            "peer_samples": peer_wall_ms,
        },
        "peer_median_ms": medians,
        "peer_samples": readings,
        "maximum_peer_logit_difference": maximum_logit_difference,
        "scope": "separate yesnod process on the same host; model load, server startup and publication excluded from peer hit times",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output / 'summary.json'}")
    print("metric                               median ms")
    print("-----------------------------------  ---------")
    for key in METRICS:
        print(f"{key:35}  {medians[key]:9.1f}")
    print(f"append publication                    {publication['generation_two_append_publish_ms']:9.1f}")
    print(f"maximum peer logit difference        {maximum_logit_difference:.6g}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError, TimeoutError, KeyError, ValueError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
