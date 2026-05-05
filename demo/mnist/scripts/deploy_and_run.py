#!/usr/bin/env python3
"""
deploy_and_run.py — upload generated MNIST projects to the KV260, build
them on the board, push the MNIST test set, and run bench_mnist for each
model.  Prints a summary table and writes results to results.json.

Pipeline per model:
  upload  → cmake -DINFERENCE_TARGET=LINUX  → make → run bench_mnist
            (the dataset is uploaded once and shared across runs)

The benchmark binary prints a single-line JSON object that we parse.

Reuses the SSH/SFTP helpers from inference-scheduler/src/remote/.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEMO_DIR  = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_DIR.parent.parent
SCHED_DIR = REPO_ROOT / "inference-scheduler"

sys.path.insert(0, str(SCHED_DIR))
from src.remote import (   # noqa: E402
    _green, _red, _yellow, _bold, _dim,
    RemoteSession,
)


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepLog:
    name:     str
    ok:       bool
    duration: float
    output:   str = ""


@dataclass
class ModelResult:
    name:    str
    steps:   List[StepLog]      = field(default_factory=list)
    metrics: Optional[dict]     = None

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)


# ──────────────────────────────────────────────────────────────────────────────
# Upload + build helpers
# ──────────────────────────────────────────────────────────────────────────────

def upload_dataset(session: RemoteSession, local_data: Path,
                    remote_data: str) -> StepLog:
    t0 = time.monotonic()
    out = ""
    try:
        session.exec(f"mkdir -p {shlex.quote(remote_data)}", timeout=15)
        n   = session.upload_dir(local_data, remote_data)
        ok  = True
        out = f"{n} files"
    except Exception as exc:
        ok  = False
        out = str(exc)
    return StepLog("dataset", ok, time.monotonic() - t0, out)


def upload_project(session: RemoteSession, local_proj: Path,
                    remote_proj: str) -> StepLog:
    t0 = time.monotonic()
    try:
        session.exec(f"rm -rf {shlex.quote(remote_proj)}", timeout=30)
        n  = session.upload_dir(local_proj, remote_proj)
        return StepLog("upload", True, time.monotonic() - t0, f"{n} files")
    except Exception as exc:
        return StepLog("upload", False, time.monotonic() - t0, str(exc))


_UIO_CMAKE_DEFINE = {
    "VectorOPKernel": "INFERENCE_VECTOROPKERNEL_INSTANCE",
    "MatmulKernel":   "INFERENCE_MATMULKERNEL_INSTANCE",
    "ConvKernel":     "INFERENCE_CONVKERNEL_INSTANCE",
    "PoolKernel":     "INFERENCE_POOLKERNEL_INSTANCE",
}


def configure_and_build(session: RemoteSession, cfg: dict,
                         remote_proj: str, remote_data: str,
                         active_kernels: List[str]) -> Tuple[StepLog,
                                                              StepLog]:
    build_dir = f"{remote_proj}/build"
    extra     = " ".join(cfg["remote"].get("cmake_args", []))

    # The bench_glue.h header defaults each UIO instance name to "<Kernel>_0".
    # If the loaded overlay uses different node labels (commonly fabric_vecop,
    # fabric_matmul, fabric_conv, fabric_pool), forward them as compile-time
    # defines so inference_init() opens the right /dev/uioN.
    uio_defs = []
    uio_devs = cfg["remote"].get("uio_devices", {})
    for k in active_kernels:
        name = uio_devs.get(k)
        if name:
            macro = _UIO_CMAKE_DEFINE[k]
            uio_defs.append(f'-D{macro}={shlex.quote(name)}')

    cmake_cmd = (
        f"cmake -S {shlex.quote(remote_proj)} -B {shlex.quote(build_dir)} "
        f"-DCMAKE_BUILD_TYPE=Release "
        f"-DINFERENCE_TARGET=LINUX "
        f"-DBENCH_DATA_DIR={shlex.quote(remote_data)} "
        f"{' '.join(uio_defs)} {extra} 2>&1"
    )

    t0 = time.monotonic()
    out, _, rc = session.exec(cmake_cmd, timeout=cfg["build"]["timeout"])
    cmake_step = StepLog("cmake", rc == 0, time.monotonic() - t0, out)
    if rc != 0:
        return cmake_step, StepLog("make", False, 0.0, "skipped (cmake failed)")

    t0 = time.monotonic()
    make_cmd = (f"make -C {shlex.quote(build_dir)} "
                f"-j{cfg['build']['jobs']} bench_mnist 2>&1")
    out, _, rc = session.exec(make_cmd, timeout=cfg["build"]["timeout"])
    make_step = StepLog("make", rc == 0, time.monotonic() - t0, out)
    return cmake_step, make_step


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark execution
# ──────────────────────────────────────────────────────────────────────────────

def _stream_exec(session: RemoteSession, cmd: str, *,
                  on_stderr_line, timeout: int) -> Tuple[str, str, int]:
    """
    Like RemoteSession.exec but streams stderr line-by-line so the caller can
    render progress messages live.  Returns (stdout, stderr, exit_code) once
    the command completes.  Reuses the existing paramiko transport.
    """
    client = session._client          # noqa: SLF001 — internal field reuse
    if client is None:
        raise RuntimeError("RemoteSession: not connected")

    _, stdout, _ = client.exec_command(cmd, timeout=float(timeout),
                                       get_pty=False)
    chan = stdout.channel
    out_chunks: List[str] = []
    err_chunks: List[str] = []
    err_partial = ""

    deadline = time.monotonic() + float(timeout)
    while True:
        progressed = False
        while chan.recv_ready():
            out_chunks.append(chan.recv(4096).decode("utf-8", errors="replace"))
            progressed = True
        while chan.recv_stderr_ready():
            chunk = chan.recv_stderr(4096).decode("utf-8", errors="replace")
            err_chunks.append(chunk)
            err_partial += chunk
            while "\n" in err_partial:
                line, err_partial = err_partial.split("\n", 1)
                on_stderr_line(line)
            progressed = True
        if chan.exit_status_ready() and not chan.recv_ready() \
                and not chan.recv_stderr_ready():
            break
        if time.monotonic() > deadline:
            chan.close()
            raise TimeoutError(f"streaming exec timed out after {timeout}s")
        if not progressed:
            time.sleep(0.05)

    if err_partial:
        on_stderr_line(err_partial)
    return ("".join(out_chunks), "".join(err_chunks), chan.recv_exit_status())


def run_benchmark(session: RemoteSession, cfg: dict,
                   remote_proj: str) -> Tuple[StepLog, Optional[dict]]:
    binary = f"{remote_proj}/build/bench_mnist"

    sudo   = "sudo -n " if cfg["run"].get("use_sudo", True) else ""
    iters  = int(cfg["run"].get("iters", 0))
    warmup = int(cfg["run"].get("warmup", 50))

    cmd = f"{sudo}{shlex.quote(binary)} {iters} {warmup}"

    # Render `progress:` lines in place (CR-overwrite); pass other stderr
    # output through verbatim so any error / setup message is still visible.
    state = {"progress_active": False}
    is_tty = sys.stderr.isatty()

    def _on_line(line: str) -> None:
        line = line.rstrip("\r")
        if line.startswith("progress:"):
            if is_tty:
                # Pad to clear leftovers from a longer previous line.
                sys.stderr.write("\r    " + line.ljust(78))
                sys.stderr.flush()
                state["progress_active"] = True
            else:
                sys.stderr.write("    " + line + "\n")
        else:
            if state["progress_active"]:
                sys.stderr.write("\n")
                state["progress_active"] = False
            if line:
                sys.stderr.write("    " + line + "\n")

    t0 = time.monotonic()
    try:
        out, err, rc = _stream_exec(session, cmd, on_stderr_line=_on_line,
                                     timeout=cfg["run"]["timeout"])
    except TimeoutError as exc:
        if state["progress_active"]:
            sys.stderr.write("\n"); state["progress_active"] = False
        return StepLog("run", False, time.monotonic() - t0, str(exc)), None
    finally:
        if state["progress_active"]:
            sys.stderr.write("\n")
    duration = time.monotonic() - t0

    if rc != 0:
        return StepLog("run", False, duration, (out + err).strip()), None

    # Find the JSON line emitted by bench_mnist (allow stderr noise on stdout).
    metrics = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                metrics = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if metrics is None:
        return StepLog("run", False, duration,
                       f"could not parse JSON output\n{out}{err}"), None
    full_log = "STDOUT:\n" + out.rstrip() + "\nSTDERR:\n" + err.rstrip()
    return StepLog("run", True, duration, full_log), metrics


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def _format_step(s: StepLog) -> str:
    tag = _green("OK") if s.ok else _red("FAIL")
    return f"  {s.name:<8} → {tag:<14}  {s.duration:5.1f}s"


def _print_failure_tail(s: StepLog, *, max_lines: int = 30) -> None:
    """Echo the captured stdout/stderr of a failed step (capped)."""
    text = (s.output or "").rstrip()
    if not text:
        return
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = ["…"] + lines[-max_lines:]
    for ln in lines:
        print(_dim(f"      {ln}"))


def _save_log(log_dir: Path, model: str, step: StepLog) -> None:
    """Persist a step's full output for post-mortem inspection."""
    if not step.output:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{model}.{step.name}.log"
    path.write_text(step.output)


def print_report(results: List[ModelResult]) -> None:
    print()
    print(_bold("  ── MNIST KV260 BENCHMARK ──"))
    print()
    rows = []
    for r in results:
        if r.metrics:
            m = r.metrics
            rows.append((r.name, "OK",
                         f"{m['accuracy_pct']:.2f}%",
                         f"{m['mean_ms']:.3f}",
                         f"{m['p50_ms']:.3f}",
                         f"{m['p99_ms']:.3f}",
                         f"{m['throughput_ips']:.1f}"))
        else:
            rows.append((r.name, "FAIL", "-", "-", "-", "-", "-"))

    name_w = max(len(r[0]) for r in rows)
    hdr = f"  {'Model':<{name_w}}  {'Status':<6}  {'Acc':>8}  " \
          f"{'mean(ms)':>9}  {'p50(ms)':>9}  {'p99(ms)':>9}  {'IPS':>9}"
    sep = "  " + "─" * (len(hdr) - 2)
    print(hdr); print(sep)
    for r in rows:
        status = _green(r[1]) if r[1] == "OK" else _red(r[1])
        print(f"  {r[0]:<{name_w}}  {status:<14}  {r[2]:>8}  "
              f"{r[3]:>9}  {r[4]:>9}  {r[5]:>9}  {r[6]:>9}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def deploy_models(cfg: dict, projects: List[dict],
                   data_dir: Path, *, log_dir: Path,
                   verbose: bool = False) -> List[ModelResult]:
    session = RemoteSession(cfg["ssh"])
    print(f"\n{_bold('Connecting')} to "
          f"{cfg['ssh']['user']}@{cfg['ssh']['host']}:{cfg['ssh']['port']} …")
    session.connect()
    print(_green("  connected"))

    work_dir    = cfg["remote"]["work_dir"].rstrip("/")
    remote_data = f"{work_dir}/data"
    results: List[ModelResult] = []

    def _record(res: ModelResult, step: StepLog) -> None:
        res.steps.append(step)
        print(_format_step(step))
        _save_log(log_dir, res.name, step)
        if not step.ok:
            _print_failure_tail(step)
        # `run` output was already streamed live; don't echo it again on success.
        elif verbose and step.output and step.name != "run":
            _print_failure_tail(step)

    try:
        # Upload dataset once.
        print(f"\n{_bold('Uploading dataset')} → {remote_data}")
        ds_step = upload_dataset(session, data_dir, remote_data)
        print(_format_step(ds_step))
        if not ds_step.ok:
            _print_failure_tail(ds_step)
            print(_red("dataset upload failed; aborting"))
            return results

        for proj in projects:
            name        = proj["model_name"]
            local_proj  = Path(proj["project_dir"])
            remote_proj = f"{work_dir}/projects/{name}"
            print(f"\n{_bold(name)}")
            res = ModelResult(name=name)

            up = upload_project(session, local_proj, remote_proj)
            _record(res, up)
            if not up.ok:
                results.append(res); continue

            cm, mk = configure_and_build(session, cfg, remote_proj, remote_data,
                                          proj.get("active", []))
            _record(res, cm)
            if not cm.ok:
                results.append(res); continue
            _record(res, mk)
            if not mk.ok:
                results.append(res); continue

            run, metrics = run_benchmark(session, cfg, remote_proj)
            _record(res, run)
            if metrics:
                res.metrics = metrics
                print(f"    accuracy = {metrics['accuracy_pct']:.2f}%   "
                      f"mean = {metrics['mean_ms']:.3f} ms   "
                      f"throughput = {metrics['throughput_ips']:.1f} img/s")
            results.append(res)

        if cfg.get("cleanup", True):
            print(f"\n{_dim('cleanup')} {work_dir}")
            session.exec(f"rm -rf {shlex.quote(work_dir)}", timeout=30)
    finally:
        session.close()

    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEMO_DIR / "mnist_config.json"))
    p.add_argument("--projects-dir",
                   default=str(DEMO_DIR / "build" / "projects"),
                   help="root of generated projects (must contain projects.json)")
    p.add_argument("--data-dir",
                   default=str(DEMO_DIR / "assets" / "data"),
                   help="local directory with the unpacked MNIST IDX files")
    p.add_argument("--results",
                   default=str(DEMO_DIR / "build" / "results.json"),
                   help="path to write the JSON results summary")
    p.add_argument("--no-cleanup", action="store_true",
                   help="leave the remote work_dir in place after the run")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    cfg = _load_json(Path(args.config))
    if args.no_cleanup:
        cfg["cleanup"] = False

    projects_summary = Path(args.projects_dir) / "projects.json"
    if not projects_summary.exists():
        print(f"error: {projects_summary} not found — "
              f"run scripts/generate_project.py first", file=sys.stderr)
        return 1
    projects = json.loads(projects_summary.read_text())

    data_dir = Path(args.data_dir)
    for required in ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        if not (data_dir / required).exists():
            print(f"error: {data_dir / required} missing — "
                  f"run scripts/download_assets.py first", file=sys.stderr)
            return 1

    log_dir = Path(args.results).parent / "logs"
    results = deploy_models(cfg, projects, data_dir,
                             log_dir=log_dir, verbose=args.verbose)
    if log_dir.exists():
        print(_dim(f"per-step logs written to {log_dir}"))

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    summary = [{"name":   r.name,
                "ok":     r.ok,
                "metrics": r.metrics,
                "steps":   [{"name": s.name, "ok": s.ok,
                             "duration_s": s.duration} for s in r.steps]}
               for r in results]
    Path(args.results).write_text(json.dumps(summary, indent=2))

    print_report(results)
    n_ok = sum(1 for r in results if r.ok)
    return 0 if n_ok == len(results) and results else 1


if __name__ == "__main__":
    sys.exit(main())
