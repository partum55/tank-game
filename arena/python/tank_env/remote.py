"""
An agent running in its own process (own interpreter / venv, own imports).

Why: both players' models run on ONE laptop. A separate process per agent means
no module-name clashes, no peeking into the other agent or the arena, different
dependency sets (torch vs numpy) side by side, and a crashed / hung agent loses
only its own match instead of killing the duel.

POSIX only (uses select on pipes).
"""
import json
import select
import subprocess
import sys
from pathlib import Path

HOST = Path(__file__).resolve().parents[1] / "agent_host.py"


class AgentFailure(Exception):
    """The agent crashed, hung or answered garbage."""


class RemoteAgent:
    def __init__(self, spec: str, python: str = None, timeout_ms: float = 500.0, startup_s: float = 120.0):
        self.spec = spec
        self.python = python or sys.executable
        self.timeout = timeout_ms / 1000.0
        self.startup_s = startup_s
        self.proc = None
        self._start()

    def _start(self):
        self.proc = subprocess.Popen(
            [self.python, str(HOST), self.spec],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        reply = self._read(self.startup_s)
        if "error" in reply:
            self.close()
            raise AgentFailure(f"{self.spec} failed to start:\n{reply['error']}")

    def _read(self, timeout_s: float) -> dict:
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout_s)
        if not ready:
            self.close()
            raise AgentFailure(f"{self.spec}: no answer within {timeout_s * 1000:.0f} ms")
        line = self.proc.stdout.readline()
        if not line:
            code = self.proc.poll()
            raise AgentFailure(f"{self.spec}: agent process died (exit code {code})")
        reply = json.loads(line)
        if "error" in reply:
            self.close()
            raise AgentFailure(f"{self.spec} raised:\n{reply['error']}")
        return reply

    def _ask(self, msg: dict, timeout_s: float) -> dict:
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, AttributeError):
            raise AgentFailure(f"{self.spec}: agent process is not running")
        return self._read(timeout_s)

    def reset(self):
        self._ask({"reset": 1}, max(self.timeout, 5.0))

    def act(self, obs: dict):
        return self._ask({"obs": obs}, self.timeout)["action"]

    def restart(self):
        """Fresh process after a failure (a hung agent may still answer late and desync the pipe)."""
        self.close()
        self._start()

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        if self.proc is not None:
            self.proc.wait()
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except Exception:
                    pass
        self.proc = None
