"""
Prometheus Watchdog — supervises trader.py and three Streamlit command centres.

Runs the trader plus the consolidated Prometheus, Hermes, and Olympus dashboard
applications in parallel.
Either process is restarted automatically on crash. An intentional stop
(Stop button / Ctrl-C) exits everything.

Stop signals:
  stop_flag      → trader.py reads this to exit cleanly
  watchdog_stop  → tells the watchdog to not restart anything

Restart policy per process:
  • cooldown            : 15 s between restarts
  • crash_limit         : 5 crashes within crash_window before halting that process
  • crash_window        : 300 s sliding window
  • Dashboards use a more relaxed crash_limit (10) since they're stateless.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE         = pathlib.Path(__file__).parent
_ROOT         = _HERE.parent                    # Prometheus/
_TRADER       = _HERE / "trader.py"
_PROMETHEUS_DASH = _ROOT / "ui" / "prometheus_command_center.py"
_HERMES_DASH    = _ROOT / "ui" / "hermes_command_center.py"
_OLYMPUS_DASH   = _ROOT / "ui" / "olympus_command_center.py"
_VENV_PY      = pathlib.Path(sys.executable)   # same venv as watchdog

STOP_FLAG        = _HERE / "stop_flag"          # trader.py reads this name
WATCHDOG_STOP    = _HERE / "watchdog_stop"      # dashboard Stop button writes this
WATCHDOG_PID_F   = _HERE / "watchdog.pid"
WATCHDOG_LOG_F   = _HERE / "watchdog.log"
STATUS_FILE      = _HERE / "bot_status.json"

# ── Restart policy ─────────────────────────────────────────────────────────────
RESTART_COOLDOWN_S      = 15
BOT_CRASH_LIMIT         = 5
DASH_CRASH_LIMIT        = 10   # dashboard is stateless — more restarts are fine
CRASH_WINDOW_S          = 300

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [watchdog] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(WATCHDOG_LOG_F), encoding="utf-8"),
    ],
)
log = logging.getLogger("watchdog")

# Shared stop event — set by either thread to tear everything down.
_STOP_EVENT = threading.Event()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _intentional_stop() -> bool:
    return STOP_FLAG.exists() or WATCHDOG_STOP.exists()


def _sleep_interruptible(seconds: int) -> bool:
    """Sleep up to `seconds`, returning True if _STOP_EVENT fired early."""
    return _STOP_EVENT.wait(timeout=seconds)


# ── Generic supervisor ─────────────────────────────────────────────────────────

def _supervise(
    name: str,
    cmd: list[str],
    cwd: pathlib.Path,
    *,
    cooldown: int,
    crash_limit: int,
    crash_window: int,
    pre_launch_hook=None,   # callable() run before every launch (e.g. clear stop_flag)
) -> None:
    """
    Supervise a single subprocess.  Runs in its own thread.
    Stops when _STOP_EVENT is set or an intentional stop flag is detected.
    """
    log.info("[%s] supervisor started.", name)
    crash_times: list[float] = []
    attempt = 0

    while not _STOP_EVENT.is_set():
        if _intentional_stop():
            log.info("[%s] intentional stop detected — supervisor exiting.", name)
            _STOP_EVENT.set()
            break

        attempt += 1
        if pre_launch_hook:
            try:
                pre_launch_hook()
            except Exception:
                pass

        log.info("[%s] launching (attempt #%d): %s", name, attempt, " ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, cwd=str(cwd))
            log.info("[%s] PID %d.", name, proc.pid)
        except Exception as exc:
            log.error("[%s] launch failed: %s — retrying in %ds.", name, exc, cooldown)
            _sleep_interruptible(cooldown)
            continue

        # Poll proc exit so we can react to _STOP_EVENT promptly.
        while True:
            try:
                exit_code = proc.wait(timeout=1)
                break                   # process finished
            except subprocess.TimeoutExpired:
                if _STOP_EVENT.is_set() or _intentional_stop():
                    log.info("[%s] stop event — terminating PID %d.", name, proc.pid)
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    _STOP_EVENT.set()
                    return

        log.info("[%s] exited with code %d.", name, exit_code)

        if exit_code == 0 or _intentional_stop() or _STOP_EVENT.is_set():
            log.info("[%s] clean stop — supervisor exiting.", name)
            _STOP_EVENT.set()
            break

        # Crash accounting
        now = time.monotonic()
        crash_times.append(now)
        crash_times = [t for t in crash_times if now - t <= crash_window]

        if len(crash_times) >= crash_limit:
            log.error(
                "[%s] crashed %d times in %ds — halting this supervisor only. "
                "Fix the error and restart manually if needed.",
                name, crash_limit, crash_window,
            )
            if name == "bot":
                try:
                    STATUS_FILE.write_text(
                        json.dumps({
                            "last_action": "watchdog_halted",
                            "reason": f"bot crashed {crash_limit}x in {crash_window}s",
                            "halted_at": _ts(),
                        }),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            break

        log.warning(
            "[%s] crashed — restarting in %ds (crash %d/%d in window).",
            name, cooldown, len(crash_times), crash_limit,
        )
        _sleep_interruptible(cooldown)

    log.info("[%s] supervisor thread finished.", name)


# ── Entry point ────────────────────────────────────────────────────────────────

def _acquire_pid_lock() -> bool:
    """Return True if this instance is the sole watchdog; False if another is running."""
    if WATCHDOG_PID_F.exists():
        try:
            existing_pid = int(WATCHDOG_PID_F.read_text(encoding="utf-8").strip())
            # Check if a process with that PID actually exists
            try:
                os.kill(existing_pid, 0)   # signal 0 = existence check, no actual signal
                # Process is alive — check it's a watchdog, not a recycled PID
                import subprocess as _sp
                try:
                    out = _sp.check_output(
                        ["wmic", "process", "where", f"ProcessId={existing_pid}",
                         "get", "CommandLine", "/value"],
                        timeout=3, stderr=_sp.DEVNULL,
                    ).decode(errors="replace")
                    if "watchdog" in out.lower():
                        log.warning(
                            "Another watchdog is already running (PID %d) — exiting.",
                            existing_pid,
                        )
                        return False
                except Exception:
                    pass  # wmic not available or timed out — fall through
            except (OSError, ProcessLookupError):
                pass  # stale PID file — proceed
        except (ValueError, OSError):
            pass
    WATCHDOG_PID_F.write_text(str(os.getpid()), encoding="utf-8")
    return True


def run(
    trader_args: list[str],
    *,
    cooldown: int = RESTART_COOLDOWN_S,
    bot_crash_limit: int = BOT_CRASH_LIMIT,
    dash_crash_limit: int = DASH_CRASH_LIMIT,
    crash_window: int = CRASH_WINDOW_S,
    dash_port: int = 8501,
) -> None:
    if not _acquire_pid_lock():
        return   # another watchdog already running

    log.info("Watchdog PID %d — supervising bot + three command centres.", os.getpid())

    # Clear stale stop flags from a previous run.
    STOP_FLAG.unlink(missing_ok=True)
    WATCHDOG_STOP.unlink(missing_ok=True)

    bot_cmd = [str(_VENV_PY), str(_TRADER)] + trader_args
    dash_cmd = [
        str(_VENV_PY), "-m", "streamlit", "run", str(_PROMETHEUS_DASH),
        f"--server.port={dash_port}",
        "--server.headless=true",
        "--server.runOnSave=false",
        "--server.address=0.0.0.0",
    ]
    hermes_dash_cmd = [
        str(_VENV_PY), "-m", "streamlit", "run", str(_HERMES_DASH),
        "--server.port=8503",
        "--server.headless=true",
        "--server.runOnSave=false",
        "--server.address=0.0.0.0",
    ]
    olympus_dash_cmd = [
        str(_VENV_PY), "-m", "streamlit", "run", str(_OLYMPUS_DASH),
        "--server.port=8511",
        "--server.headless=true",
        "--server.runOnSave=false",
        "--server.address=0.0.0.0",
    ]

    def _clear_bot_stop_flag():
        """Remove stop_flag before each bot restart so the bot doesn't immediately exit."""
        STOP_FLAG.unlink(missing_ok=True)

    bot_thread = threading.Thread(
        target=_supervise,
        name="bot-supervisor",
        args=("bot", bot_cmd, _ROOT),
        kwargs=dict(
            cooldown=cooldown,
            crash_limit=bot_crash_limit,
            crash_window=crash_window,
            pre_launch_hook=_clear_bot_stop_flag,
        ),
        daemon=True,
    )
    dash_thread = threading.Thread(
        target=_supervise,
        name="dash-supervisor",
        args=("dashboard", dash_cmd, _ROOT),
        kwargs=dict(
            cooldown=cooldown,
            crash_limit=dash_crash_limit,
            crash_window=crash_window,
        ),
        daemon=True,
    )
    hermes_dash_thread = threading.Thread(
        target=_supervise,
        name="hermes-dashboard-supervisor",
        args=("hermes-dashboard", hermes_dash_cmd, _ROOT),
        kwargs=dict(
            cooldown=cooldown,
            crash_limit=dash_crash_limit,
            crash_window=crash_window,
        ),
        daemon=True,
    )
    olympus_dash_thread = threading.Thread(
        target=_supervise,
        name="olympus-dashboard-supervisor",
        args=("olympus-dashboard", olympus_dash_cmd, _ROOT),
        kwargs=dict(
            cooldown=cooldown,
            crash_limit=dash_crash_limit,
            crash_window=crash_window,
        ),
        daemon=True,
    )

    bot_thread.start()
    dash_thread.start()
    hermes_dash_thread.start()
    olympus_dash_thread.start()

    try:
        # Main thread waits; Ctrl-C sets the stop event and lets threads finish.
        while (
            bot_thread.is_alive()
            or dash_thread.is_alive()
            or hermes_dash_thread.is_alive()
            or olympus_dash_thread.is_alive()
        ):
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Ctrl-C received — stopping all processes.")
        STOP_FLAG.write_text("stop", encoding="utf-8")
        WATCHDOG_STOP.write_text("stop", encoding="utf-8")
        _STOP_EVENT.set()

    bot_thread.join(timeout=30)
    dash_thread.join(timeout=30)
    hermes_dash_thread.join(timeout=30)
    olympus_dash_thread.join(timeout=30)
    WATCHDOG_PID_F.unlink(missing_ok=True)
    log.info("Watchdog exited.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prometheus Watchdog — supervises bot + dashboard",
        add_help=False,
    )
    p.add_argument("--cooldown",          type=int, default=RESTART_COOLDOWN_S)
    p.add_argument("--crash-limit",       type=int, default=BOT_CRASH_LIMIT,
                   help="Max bot crashes in crash-window before halting")
    p.add_argument("--dash-crash-limit",  type=int, default=DASH_CRASH_LIMIT)
    p.add_argument("--crash-window",      type=int, default=CRASH_WINDOW_S)
    p.add_argument("--dash-port",         type=int, default=8501)

    known, trader_args = p.parse_known_args()

    run(
        trader_args,
        cooldown=known.cooldown,
        bot_crash_limit=known.crash_limit,
        dash_crash_limit=known.dash_crash_limit,
        crash_window=known.crash_window,
        dash_port=known.dash_port,
    )


if __name__ == "__main__":
    main()
