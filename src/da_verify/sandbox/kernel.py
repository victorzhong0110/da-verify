"""Stateful, resource-capped Jupyter-kernel sandbox for running agent code.

WHY a real kernel (not exec()/subprocess-per-call):
  A data-analysis agent works incrementally — load df, look, compute, refine.
  A Jupyter kernel keeps variables alive across calls (stateful), which is the
  natural execution model and lets the agent build on prior steps.

=============================================================================
ISOLATION BOUNDARY — what is and is NOT enforced (read this honestly)
=============================================================================
Model-written code is UNTRUSTED. This portable (single-Mac) sandbox enforces:

  [ENFORCED]
  - Per-cell WALL-CLOCK TIMEOUT: a cell that hangs/loops is interrupted (the
    kernel is interrupted, not the whole process) → no infinite hangs.
  - SEPARATE PROCESS: code runs in a kernel subprocess, not in our process, so
    a crash/segfault takes down the kernel, not the harness.
  - CWD SCOPING: the kernel runs in a fresh temp dir; the task CSV is copied in
    and chmod'd READ-ONLY (0o444) → the agent cannot mutate the source data.
  - MEMORY CAP (best-effort): RLIMIT_AS is set in the kernel. On Linux this
    raises MemoryError past the cap; on macOS it is often NOT honored.

  [NOT ENFORCED here — deferred to container deployment]
  - NETWORK: not hard-blocked. True isolation needs OS-level controls
    (container `--network none`, seccomp, or a firewall). We do NOT pretend a
    Python-level monkeypatch is security — it is bypassable, so we don't ship a
    false sense of safety. For untrusted-at-scale runs, wrap this kernel in a
    container with no network + cgroup CPU/mem limits.
  - FILESYSTEM beyond cwd: the agent could still read/write elsewhere on the
    box. cwd-scoping + read-only data is a speed-bump, not a jail.

This honesty matters: the project's whole point is not over-claiming. The
boundary above is exactly what W2 buys, and what a production deploy must add.
"""

from __future__ import annotations

import queue
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from jupyter_client.manager import KernelManager

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Run in the kernel at startup. Best-effort resource cap + preload libs.
_STARTUP = """
import resource as _r
def _lim(which, soft):
    try:
        _r.setrlimit(which, (soft, soft))
    except Exception:
        pass
_lim(_r.RLIMIT_AS, {mem_bytes})   # memory cap (best-effort; macOS may ignore)
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
"""


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    stdout: str
    result: str  # text/plain of the last expression, if any
    error: str  # cleaned traceback, if any
    timed_out: bool = False

    def as_observation(self, limit: int = 4000) -> str:
        """Render what the agent sees back after a tool call."""
        if self.timed_out:
            return f"[TIMEOUT] cell exceeded the time limit and was interrupted.\nstdout so far:\n{self.stdout[:limit]}"
        parts = []
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.strip()}")
        if self.result.strip():
            parts.append(f"result:\n{self.result.strip()}")
        if self.error.strip():
            parts.append(f"error:\n{self.error.strip()}")
        text = "\n".join(parts) if parts else "(no output)"
        return text[:limit] + ("\n…[truncated]" if len(text) > limit else "")


class KernelSandbox:
    def __init__(self, data_csv: Path | None = None, mem_mb: int = 2048, cell_timeout: float = 30.0):
        self.cell_timeout = cell_timeout
        self.mem_mb = mem_mb
        self.workdir = Path(tempfile.mkdtemp(prefix="da_sbx_"))
        self.csv_path: Path | None = None
        if data_csv is not None:
            dst = self.workdir / Path(data_csv).name
            shutil.copy(data_csv, dst)
            dst.chmod(0o444)  # read-only: agent cannot corrupt source data
            self.csv_path = dst
        self._km: KernelManager | None = None
        self._kc = None

    def __enter__(self) -> "KernelSandbox":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def start(self) -> "KernelSandbox":
        self._km = KernelManager()
        self._km.start_kernel(cwd=str(self.workdir))
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)
        self._raw_exec(_STARTUP.format(mem_bytes=self.mem_mb * 1024 * 1024), timeout=30)
        if self.csv_path is not None:
            # The agent gets the data path as a ready-made variable.
            self._raw_exec(f"CSV_PATH = {str(self.csv_path)!r}", timeout=10)
        return self

    def execute(self, code: str) -> ExecResult:
        return self._raw_exec(code, timeout=self.cell_timeout)

    def _raw_exec(self, code: str, timeout: float) -> ExecResult:
        assert self._kc is not None, "sandbox not started"
        msg_id = self._kc.execute(code)
        stdout: list[str] = []
        result: list[str] = []
        error: list[str] = []
        timed_out = False
        end = time.time() + timeout
        while True:
            remaining = end - time.time()
            if remaining <= 0:
                self._km.interrupt_kernel()  # interrupt the CELL, keep kernel alive
                timed_out = True
                # CRITICAL: drain the interrupted cell's trailing messages
                # (KeyboardInterrupt traceback + idle), else they bleed into
                # the next cell and corrupt its captured output.
                self._drain_until_idle(msg_id, max_wait=5.0)
                break
            try:
                msg = self._kc.get_iopub_msg(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue  # message from an earlier cell — ignore
            mtype, content = msg["msg_type"], msg["content"]
            if mtype == "stream":
                stdout.append(content.get("text", ""))
            elif mtype in ("execute_result", "display_data"):
                result.append(content.get("data", {}).get("text/plain", ""))
            elif mtype == "error":
                error.append(_ANSI.sub("", "\n".join(content.get("traceback", []))))
            elif mtype == "status" and content.get("execution_state") == "idle":
                break
        return ExecResult(
            ok=(not error and not timed_out),
            stdout="".join(stdout),
            result="\n".join(result),
            error="\n".join(error),
            timed_out=timed_out,
        )

    def _drain_until_idle(self, msg_id: str, max_wait: float = 5.0) -> None:
        """Consume and discard the interrupted cell's tail until its 'idle'."""
        end = time.time() + max_wait
        while time.time() < end:
            try:
                msg = self._kc.get_iopub_msg(timeout=min(end - time.time(), 1.0))
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            if msg["msg_type"] == "status" and msg["content"].get("execution_state") == "idle":
                return

    def shutdown(self) -> None:
        try:
            if self._kc is not None:
                self._kc.stop_channels()
            if self._km is not None:
                self._km.shutdown_kernel(now=True)
        finally:
            shutil.rmtree(self.workdir, ignore_errors=True)
