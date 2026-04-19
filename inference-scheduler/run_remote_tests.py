#!/usr/bin/env python3
"""
run_remote_tests.py — Hardware inference test runner for KV260.

For each ONNX model in the configured list:
  1. Generate a C inference project locally (inference_scheduler.py)
  2. Upload the project to the remote board via SSH/SFTP
  3. Build on the remote machine  (cmake -DINFERENCE_TARGET=LINUX + make)
  4. Run the test binary as root  (sudo ./build/test_inference <instance_name>)
  5. Collect pass/fail results and print a summary report

Prerequisites on the remote machine:
  - gcc/g++, cmake ≥ 3.19, make
  - XRT runtime installed (xrt.h reachable via pkg-config or /opt/xilinx/xrt)
  - A UIO device whose /sys/class/uio/uio*/name matches the configured
    instance name (default "fabric"); verify with:
      cat /sys/class/uio/uio*/name
  - Passwordless sudo for the SSH user, OR log in directly as root

Usage:
  # Run all models listed in the config file
  python run_remote_tests.py --config test/remote_config.json

  # Override model list on the command line
  python run_remote_tests.py --config test/remote_config.json \\
      --models test/models/single_add.onnx test/models/relu_chain.onnx

  # Verify SSH connectivity and remote prerequisites without running tests
  python run_remote_tests.py --config test/remote_config.json --check-only

  # Keep remote directories after the run (for debugging build/run failures)
  python run_remote_tests.py --config test/remote_config.json --no-cleanup

  # Show full build + test output for every model, not only failures
  python run_remote_tests.py --config test/remote_config.json --verbose
"""

import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:
    import paramiko
    import paramiko.sftp_client
except ImportError:
    print("error: paramiko is required.", file=sys.stderr)
    print("  .venv/bin/pip install paramiko", file=sys.stderr)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Terminal colours (disabled when stdout is not a TTY)
# ──────────────────────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s


def _green(s: str)  -> str: return _c("32", s)
def _red(s: str)    -> str: return _c("31", s)
def _yellow(s: str) -> str: return _c("33", s)
def _bold(s: str)   -> str: return _c("1",  s)
def _dim(s: str)    -> str: return _c("2",  s)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "ssh": {
        "host":            "",
        "user":            "root",
        "port":            22,
        "key_file":        None,   # path to private key, e.g. "~/.ssh/id_rsa"
        "password":        None,   # plain-text password (prefer key auth)
        "connect_timeout": 15,
    },
    "remote": {
        "work_dir":   "/tmp/inference_hw_tests",
        "driver_dir": None,        # remote path containing xvectoropkernel*.{c,h}
                                   # used when driver files live on the board already
        "uio_device": "fabric",    # UIO sysfs name (cat /sys/class/uio/uio*/name)
        "cmake_args": [],          # extra -D flags forwarded to cmake
    },
    "local": {
        "driver_dir": None,        # local path; scheduler copies drivers into project
    },
    "build": {
        "jobs":    4,
        "timeout": 180,            # seconds; cmake + make combined
    },
    "run": {
        "timeout":  120,           # seconds for test_inference execution
        "use_sudo": True,          # prefix test binary with "sudo"
    },
    "cleanup": True,               # remove remote build dirs after run
    "models":  [],
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a deep copy of *base*."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str) -> dict:
    """Load JSON config and fill in defaults for missing keys."""
    with open(path) as fh:
        raw = json.load(fh)
    cfg = _deep_merge(_DEFAULTS, raw)
    if not cfg["ssh"]["host"]:
        raise ValueError("config: ssh.host is required")
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Test result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    name:     str
    ok:       bool
    output:   str   = ""
    duration: float = 0.0


@dataclass
class TestResult:
    model_name: str
    steps:      List[StepResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    @property
    def status(self) -> str:
        if not self.steps:
            return "SKIPPED"
        for s in self.steps:
            if not s.ok:
                return f"{s.name.upper()}_ERROR"
        return "PASSED"

    @property
    def total_time(self) -> float:
        return sum(s.duration for s in self.steps)

    @property
    def failed_step(self) -> Optional[StepResult]:
        for s in self.steps:
            if not s.ok:
                return s
        return None


# ──────────────────────────────────────────────────────────────────────────────
# SSH / SFTP session
# ──────────────────────────────────────────────────────────────────────────────

class RemoteSession:
    """
    Persistent SSH+SFTP connection to the remote board.

    All remote commands run via exec(); directory uploads via upload_dir().
    The session re-uses a single transport for the entire test run to avoid
    repeated TCP handshakes.
    """

    def __init__(self, ssh_cfg: dict) -> None:
        self._cfg    = ssh_cfg
        self._client: Optional[paramiko.SSHClient] = None

    # ── connection ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        cfg    = self._cfg
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        kwargs: dict = {
            "hostname":      cfg["host"],
            "username":      cfg["user"],
            "port":          int(cfg["port"]),
            "timeout":       float(cfg["connect_timeout"]),
            "allow_agent":   True,
            "look_for_keys": True,
            "banner_timeout": 30,
        }
        key_file = cfg.get("key_file")
        if key_file:
            kwargs["key_filename"]  = os.path.expanduser(key_file)
            kwargs["look_for_keys"] = False
        password = cfg.get("password")
        if password:
            kwargs["password"] = password

        client.connect(**kwargs)
        self._client = client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # ── remote command execution ─────────────────────────────────────────────

    def exec(self, command: str,
             timeout: int = 120) -> Tuple[str, str, int]:
        """
        Run *command* on the remote host.

        Returns (stdout, stderr, exit_code).
        Raises TimeoutError if the command does not finish within *timeout* s.
        Raises RuntimeError if the session is not connected.
        """
        if self._client is None:
            raise RuntimeError("RemoteSession: not connected")

        try:
            _, stdout_ch, stderr_ch = self._client.exec_command(
                command, timeout=float(timeout),
                get_pty=False,
            )
            stdout = stdout_ch.read().decode("utf-8", errors="replace")
            stderr = stderr_ch.read().decode("utf-8", errors="replace")
            rc     = stdout_ch.channel.recv_exit_status()
        except socket.timeout:
            raise TimeoutError(
                f"Remote command timed out after {timeout}s:\n  {command}"
            )
        return stdout, stderr, rc

    def exec_checked(self, command: str,
                     timeout: int = 120) -> Tuple[str, str]:
        """Like exec() but raises RuntimeError on non-zero exit."""
        out, err, rc = self.exec(command, timeout=timeout)
        if rc != 0:
            raise RuntimeError(
                f"Remote command failed (rc={rc}):\n  {command}\n"
                f"stdout:\n{out}\nstderr:\n{err}"
            )
        return out, err

    # ── file transfer ────────────────────────────────────────────────────────

    def upload_dir(self, local_dir: Path, remote_dir: str,
                   on_file: Optional[Callable[[str], None]] = None) -> int:
        """
        Recursively upload *local_dir* to *remote_dir* via SFTP.

        *on_file* is called with the relative file path before each upload
        (useful for progress reporting).  Returns the number of files uploaded.
        """
        sftp = self._client.open_sftp()
        n    = 0
        try:
            self._mkdir_p(sftp, remote_dir)
            for local_path in sorted(local_dir.rglob("*")):
                rel    = local_path.relative_to(local_dir)
                remote = f"{remote_dir}/{rel.as_posix()}"
                if local_path.is_dir():
                    try:
                        sftp.mkdir(remote)
                    except OSError:
                        pass  # already exists
                else:
                    if on_file:
                        on_file(str(rel))
                    sftp.put(str(local_path), remote)
                    n += 1
        finally:
            sftp.close()
        return n

    @staticmethod
    def _mkdir_p(sftp: paramiko.SFTPClient, remote_path: str) -> None:
        """Create *remote_path* and all missing parent directories."""
        parts   = Path(remote_path).parts
        current = ""
        for part in parts:
            if part == "/":
                current = "/"
                continue
            current = current.rstrip("/") + "/" + part
            try:
                sftp.stat(current)
            except FileNotFoundError:
                try:
                    sftp.mkdir(current)
                except OSError:
                    pass  # race condition or already exists


# ──────────────────────────────────────────────────────────────────────────────
# Project generation (local)
# ──────────────────────────────────────────────────────────────────────────────

def generate_project(model_path: Path, out_dir: Path,
                     driver_dir: Optional[Path]) -> Tuple[bool, str]:
    """
    Invoke inference_scheduler.py to produce the C project under *out_dir*.

    Returns (success, log_text).  The scheduler writes progress to stderr;
    both stdout and stderr are captured and returned as the log.
    """
    import subprocess

    scheduler = Path(__file__).parent / "inference_scheduler.py"
    cmd: List[str] = [
        sys.executable, str(scheduler),
        str(model_path),
        "--out-dir", str(out_dir),
    ]
    if driver_dir:
        cmd += ["--driver-dir", str(driver_dir)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log  = proc.stderr + proc.stdout          # scheduler logs to stderr
    return proc.returncode == 0, log


# ──────────────────────────────────────────────────────────────────────────────
# Remote build
# ──────────────────────────────────────────────────────────────────────────────

def build_remote(session: RemoteSession,
                 project_dir: str,
                 build_dir: str,
                 uio_device: str,
                 extra_cmake: List[str],
                 jobs: int,
                 timeout: int) -> Tuple[bool, str]:
    """
    Configure and compile the inference project on the remote machine.

    Two commands are run sequentially:
      cmake -S <project_dir> -B <build_dir> -DINFERENCE_TARGET=LINUX ...
      make -C <build_dir> -j<jobs> test_inference

    Returns (success, combined_log).
    """
    cmake_flags = [
        "-DINFERENCE_TARGET=LINUX",
        f'-DINFERENCE_TEST_INSTANCE=\\"{uio_device}\\"',
        # weights/ and expected/ subdirs live under the project source tree
        f"-DINFERENCE_WEIGHTS_DIR={project_dir}",
    ] + extra_cmake

    cmake_cmd = (
        f"cmake -S {project_dir} -B {build_dir} "
        + " ".join(cmake_flags)
    )
    make_cmd = f"make -C {build_dir} -j{jobs} test_inference 2>&1"

    log_parts: List[str] = []

    for cmd in (cmake_cmd, make_cmd):
        out, err, rc = session.exec(cmd, timeout=timeout)
        log_parts.append(f"$ {cmd}\n{out}")
        if err.strip():
            log_parts.append(err)
        if rc != 0:
            return False, "\n".join(log_parts)

    return True, "\n".join(log_parts)


# ──────────────────────────────────────────────────────────────────────────────
# Remote test execution
# ──────────────────────────────────────────────────────────────────────────────

def run_test(session: RemoteSession,
             project_dir: str,
             build_dir: str,
             uio_device: str,
             use_sudo: bool,
             timeout: int) -> Tuple[bool, str]:
    """
    Execute test_inference on the remote machine.

    The binary is run from *project_dir* so that relative paths like
    ./weights/*.dat and ./expected/*.dat resolve correctly.

    Root privileges are obtained via "sudo" when *use_sudo* is True and the
    SSH user is not root; if the user is already root, sudo is skipped.
    The test passes when the binary exits with code 0 and prints "PASSED".

    Returns (passed, full_output).
    """
    binary = f"{build_dir}/test_inference"

    # Use sudo -n (non-interactive) so the command fails clearly if
    # passwordless sudo is not configured, rather than hanging for a password.
    prefix = "sudo -n " if use_sudo else ""

    cmd = f"cd {project_dir} && {prefix}{binary} {uio_device}"

    try:
        out, err, rc = session.exec(cmd, timeout=timeout)
    except TimeoutError as exc:
        return False, str(exc)

    combined = out + err
    passed   = (rc == 0) and ("PASSED" in combined)
    return passed, combined


# ──────────────────────────────────────────────────────────────────────────────
# Per-model orchestration
# ──────────────────────────────────────────────────────────────────────────────

def run_model_test(model_path: Path,
                   cfg: dict,
                   session: RemoteSession,
                   tmp_root: Path) -> TestResult:
    """
    Execute the full pipeline (generate → upload → build → run) for one model.

    Each phase records a StepResult; the pipeline aborts on first failure.
    """
    name   = model_path.stem
    result = TestResult(model_name=name)

    local_project = tmp_root / name
    local_driver  = (
        Path(cfg["local"]["driver_dir"]) if cfg["local"].get("driver_dir") else None
    )
    work_dir        = cfg["remote"]["work_dir"].rstrip("/")
    remote_project  = f"{work_dir}/{name}"
    remote_build    = f"{remote_project}/build"

    # ── 1. Generate ─────────────────────────────────────────────────────── #
    _step_print(name, "generate", "generating project...")
    t0    = time.monotonic()
    ok, gen_log = generate_project(model_path, local_project, local_driver)
    dur   = time.monotonic() - t0
    result.steps.append(StepResult("generate", ok, gen_log, dur))
    _step_done(ok, dur)
    if not ok:
        return result

    # Optionally copy drivers from the remote machine's driver_dir later
    # (see _copy_remote_drivers below, invoked after upload).

    # ── 2. Upload ────────────────────────────────────────────────────────── #
    _step_print(name, "upload", "uploading...")
    t0 = time.monotonic()
    try:
        n = session.upload_dir(local_project, remote_project)
        dur = time.monotonic() - t0
        _step_done(True, dur, extra=f"{n} files")
        result.steps.append(StepResult("upload", True,
                                       f"{n} files uploaded in {dur:.1f}s", dur))
    except Exception as exc:
        dur = time.monotonic() - t0
        _step_done(False, dur)
        result.steps.append(StepResult("upload", False, str(exc), dur))
        return result

    # ── 2b. Copy drivers from remote driver_dir (if configured) ─────────── #
    remote_driver_dir = cfg["remote"].get("driver_dir")
    if remote_driver_dir and not local_driver:
        _step_print(name, "drivers", "copying remote drivers...")
        t0 = time.monotonic()
        ok, drv_log = _copy_remote_drivers(
            session, remote_driver_dir, remote_project
        )
        dur = time.monotonic() - t0
        _step_done(ok, dur)
        result.steps.append(StepResult("drivers", ok, drv_log, dur))
        if not ok:
            return result

    # ── 3. Build ─────────────────────────────────────────────────────────── #
    _step_print(name, "build", "cmake + make...")
    t0 = time.monotonic()
    ok, build_log = build_remote(
        session,
        project_dir = remote_project,
        build_dir   = remote_build,
        uio_device  = cfg["remote"]["uio_device"],
        extra_cmake = cfg["remote"].get("cmake_args", []),
        jobs        = cfg["build"]["jobs"],
        timeout     = cfg["build"]["timeout"],
    )
    dur = time.monotonic() - t0
    _step_done(ok, dur)
    result.steps.append(StepResult("build", ok, build_log, dur))
    if not ok:
        return result

    # ── 4. Run ───────────────────────────────────────────────────────────── #
    _step_print(name, "run", "running test...")
    t0 = time.monotonic()
    ok, run_output = run_test(
        session,
        project_dir = remote_project,
        build_dir   = remote_build,
        uio_device  = cfg["remote"]["uio_device"],
        use_sudo    = cfg["run"]["use_sudo"],
        timeout     = cfg["run"]["timeout"],
    )
    dur = time.monotonic() - t0
    _step_done(ok, dur)
    result.steps.append(StepResult("run", ok, run_output, dur))

    return result


def _copy_remote_drivers(session: RemoteSession,
                         src_dir: str, project_dir: str) -> Tuple[bool, str]:
    """
    Copy XVectoropkernel driver files from an existing directory on the remote
    machine into <project_dir>/driver/.  Used when the drivers are already
    installed on the board (not bundled from local).
    """
    driver_files = [
        "xvectoropkernel.h",
        "xvectoropkernel_hw.h",
        "xvectoropkernel.c",
        "xvectoropkernel_sinit.c",
        "xvectoropkernel_linux.c",
    ]
    dst = f"{project_dir}/driver"
    missing = []
    for fname in driver_files:
        src  = f"{src_dir}/{fname}"
        cmd  = f"cp {src} {dst}/{fname} 2>&1"
        _, _, rc = session.exec(cmd, timeout=15)
        if rc != 0:
            missing.append(fname)

    if missing:
        return False, f"Missing driver files in {src_dir}: {', '.join(missing)}"
    return True, f"Copied {len(driver_files)} driver files from {src_dir}"


# ──────────────────────────────────────────────────────────────────────────────
# Remote cleanup
# ──────────────────────────────────────────────────────────────────────────────

def cleanup_remote(session: RemoteSession, cfg: dict,
                   model_names: List[str]) -> None:
    work_dir = cfg["remote"]["work_dir"].rstrip("/")
    for name in model_names:
        session.exec(f"rm -rf {work_dir}/{name}", timeout=30)


# ──────────────────────────────────────────────────────────────────────────────
# Prerequisite check
# ──────────────────────────────────────────────────────────────────────────────

def check_prerequisites(session: RemoteSession, cfg: dict) -> bool:
    """
    Verify that the remote machine has the tools and devices required to
    build and run the inference tests.  Returns True if all checks pass.
    """
    uio = cfg["remote"]["uio_device"]
    checks = [
        # (shell command that exits 0 on success, label, description expr)
        (f"cmake --version 2>&1 | head -1", "cmake"),
        (f"make --version  2>&1 | head -1", "make"),
        (f"gcc  --version  2>&1 | head -1", "gcc"),
        (
            "pkg-config --exists xrt 2>/dev/null && echo 'xrt via pkg-config' || "
            "{ [ -f /opt/xilinx/xrt/include/xrt.h ] && echo 'xrt at /opt/xilinx/xrt'; } || "
            "{ [ -f /usr/include/xrt/xrt.h ]        && echo 'xrt at /usr/include/xrt'; } || "
            "echo '__MISSING__'",
            "xrt headers",
        ),
        # XVectoropkernel_Initialize() scans /sys/class/uio/*/name for a
        # device whose sysfs name matches the instance name string.
        # Check the same way: look for a name file containing the instance name.
        (
            f"match=$(grep -rl '^{uio}$' /sys/class/uio/*/name 2>/dev/null | head -1); "
            f"[ -n \"$match\" ] && echo \"/dev/uio$(echo $match | grep -o 'uio[0-9]*' | tail -1 | sed 's/uio//')\" "
            f"|| echo '__MISSING__'",
            f"uio device ({uio})",
        ),
        (
            "sudo -n true 2>/dev/null && echo 'passwordless sudo OK' || "
            "[ \"$(id -u)\" = '0' ] && echo 'running as root' || "
            "echo '__MISSING__'",
            "sudo / root access",
        ),
    ]

    all_ok = True
    for cmd, label in checks:
        out, _, rc = session.exec(cmd, timeout=15)
        missing = "__MISSING__" in out or (rc != 0 and not out.strip())
        if missing:
            print(f"    {_red('MISSING')} {label}")
            all_ok = False
        else:
            desc = out.strip().splitlines()[0][:70]
            print(f"    {_green('OK')}      {label:<28} {_dim(desc)}")

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _step_print(model: str, step: str, msg: str) -> None:
    print(f"  [{_bold(model)}] {step:<9} {msg}", end="", flush=True)


def _step_done(ok: bool, dur: float, extra: str = "") -> None:
    tag   = _green("OK")  if ok else _red("FAIL")
    extra = f"  {_dim(extra)}" if extra else ""
    print(f"\r  \033[K  " + " " * 40 +  # clear line
          f"  → {tag}  {dur:.1f}s{extra}")


def print_report(results: List[TestResult], verbose: bool = False) -> None:
    """Print a summary table and, when verbose or failed, per-step output."""
    col = max((len(r.model_name) for r in results), default=20) + 2
    hdr = f"  {'Model':<{col}}  {'Status':<18}  {'Time':>7}  Steps"

    print()
    print(_bold(hdr))
    print("  " + "─" * (len(hdr) - 2))

    for r in results:
        status_col = _green("PASSED") if r.ok else _red(r.status)
        steps_str  = "  ".join(
            (_green(s.name) if s.ok else _red(s.name)) for s in r.steps
        )
        print(f"  {r.model_name:<{col}}  {status_col:<27}  "
              f"{r.total_time:>6.1f}s  {steps_str}")

        if verbose or not r.ok:
            failed = r.failed_step
            if failed:
                print(f"\n    ─── {failed.name} output ───")
                for line in failed.output.strip().splitlines():
                    print(f"    {_dim(line)}")
                print()

    print("  " + "─" * (len(hdr) - 2))
    passed  = sum(1 for r in results if r.ok)
    total   = len(results)
    summary = f"  {passed}/{total} passed"
    print(_bold(_green(summary) if passed == total else _red(summary)))
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference hardware tests on a remote KV260 board.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Config file: see remote_config.json.example

            Remote prerequisites:
              cmake ≥ 3.19, make, gcc
              XRT runtime (xrt.h + libxrt_core.so)
              UIO device whose sysfs name matches remote.uio_device (default: "fabric")
                verify: cat /sys/class/uio/uio*/name
              Passwordless sudo  -OR-  SSH as root (user: "root")
        """),
    )
    p.add_argument("--config",    metavar="FILE", required=True,
                   help="JSON config file (see remote_config.json.example)")
    p.add_argument("--models",    metavar="ONNX", nargs="+",
                   help="Override model list; paths relative to this script's dir")
    p.add_argument("--check-only", action="store_true",
                   help="Check SSH connectivity and remote prerequisites, then exit")
    p.add_argument("--no-cleanup", action="store_true",
                   help="Keep remote build directories (for debugging)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print build/run output for every test (not only failures)")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop after the first test failure")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ── Load config ─────────────────────────────────────────────────────── #
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # ── Resolve model list ───────────────────────────────────────────────── #
    base = Path(__file__).parent
    raw_models = args.models if args.models else cfg["models"]
    if not raw_models:
        print("error: no models specified in config or --models", file=sys.stderr)
        return 1

    models: List[Path] = []
    for m in raw_models:
        p = Path(m) if Path(m).is_absolute() else base / m
        models.append(p)

    missing = [m for m in models if not m.exists()]
    if missing:
        for m in missing:
            print(f"error: model not found: {m}", file=sys.stderr)
        return 1

    # ── Connect ─────────────────────────────────────────────────────────── #
    ssh_cfg = cfg["ssh"]
    print(f"\n{_bold('Connecting')} to "
          f"{ssh_cfg['user']}@{ssh_cfg['host']}:{ssh_cfg['port']} …")
    session = RemoteSession(ssh_cfg)
    try:
        session.connect()
        print(f"  {_green('Connected')}")
    except Exception as exc:
        print(f"  {_red('FAILED')}: {exc}", file=sys.stderr)
        return 1

    try:
        # ── Prerequisite check ───────────────────────────────────────────── #
        print(f"\n{_bold('Remote prerequisites')}")
        prereqs_ok = check_prerequisites(session, cfg)

        if args.check_only:
            print(f"\n{_green('All OK') if prereqs_ok else _red('Issues found')} "
                  f"— connectivity check complete.")
            return 0 if prereqs_ok else 1

        if not prereqs_ok:
            print(
                f"\n{_yellow('Warning')}: some prerequisites are missing. "
                "Tests may fail at build or run time.",
                file=sys.stderr,
            )

        # ── Create remote working directory ──────────────────────────────── #
        work_dir = cfg["remote"]["work_dir"]
        session.exec(f"mkdir -p {work_dir}", timeout=10)

        # ── Run tests ────────────────────────────────────────────────────── #
        print(f"\n{_bold('Running tests')} "
              f"({len(models)} model{'s' if len(models) != 1 else ''})\n")

        results: List[TestResult] = []
        tmp_root = Path(tempfile.mkdtemp(prefix="inference_hw_"))

        try:
            for model_path in models:
                result = run_model_test(model_path, cfg, session, tmp_root)
                results.append(result)

                if args.verbose and result.ok:
                    # Show test output even on success
                    run_step = next(
                        (s for s in result.steps if s.name == "run"), None
                    )
                    if run_step and run_step.output.strip():
                        print(f"    ─── run output ───")
                        for line in run_step.output.strip().splitlines():
                            print(f"    {_dim(line)}")
                        print()

                if not result.ok and not args.verbose:
                    # Show failure output immediately (also in the final report)
                    failed = result.failed_step
                    if failed:
                        print(f"\n    ─── {failed.name} output ───")
                        for line in failed.output.strip().splitlines():
                            print(f"    {_dim(line)}")
                        print()

                if args.fail_fast and not result.ok:
                    print(_yellow("  Stopping after first failure (--fail-fast)."))
                    break

        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        # ── Cleanup remote ────────────────────────────────────────────────── #
        if not args.no_cleanup:
            print(f"\n{_bold('Cleaning up')} remote directories…")
            cleanup_remote(session, cfg, [r.model_name for r in results])
            print(f"  {_green('Done')}")
        else:
            print(f"\n{_dim('Remote directories kept (--no-cleanup).')}")
            print(f"  {_dim(work_dir)}/")
            for r in results:
                print(f"    {_dim(r.model_name)}/")

        # ── Report ───────────────────────────────────────────────────────── #
        print_report(results, verbose=args.verbose)

        return 0 if all(r.ok for r in results) else 1

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
