"""
run_tests.py — Headless integration test runner for Voice Notes Agent
=====================================================================
Bypasses Whisper and the GUI entirely.
Injects fixed text transcriptions directly into the agent queue and
waits for the agent worker to process each one.

Usage:
    python tests/run_tests.py
    python tests/run_tests.py --scenario "01 - System Design Notes"
    python tests/run_tests.py --list

Output .md files are written to tests/output/ so they never pollute
the project root.  Open them in any Markdown viewer to verify quality.

Requirements:
    llama-server must be running on port 12345 before executing this script.
"""

import sys
import os
import json
import threading
import time
import argparse

# ---------------------------------------------------------------------------
# Path setup — make sure the project root is importable
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Output directory: all generated .md files land here, not in the project root
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Redirect agent config BEFORE importing anything from the agent package,
# so agent_worker writes to tests/output/ instead of the project root.
# ---------------------------------------------------------------------------
import agent.config as _cfg

_cfg.BASE_DIR     = OUTPUT_DIR
_cfg.SESSION_FILE = os.path.join(OUTPUT_DIR, ".session.json")
_cfg.AUDIO_TMP    = os.path.join(OUTPUT_DIR, "rec_temp.wav")

# Now import the worker (it reads BASE_DIR at call time via agent.config)
from agent.agent import agent_worker
import agent.config as agent_config  # single reference for state / queue


# ---------------------------------------------------------------------------
# Stub "app" — replaces NotesApp without touching tkinter
# ---------------------------------------------------------------------------

class StubApp:
    """
    Minimal stand-in for NotesApp.
    Satisfies the interface expected by agent_worker:
        app.log(message)
        app.set_status(text, color)
        app.root.after(delay, fn)
        app.lbl_file.config(**kwargs)
    """

    def __init__(self) -> None:
        self.root = self  # agent_worker calls app.root.after(...)

    def after(self, _delay: int, fn) -> None:
        fn()  # execute immediately — no event loop needed

    def log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        for line in message.splitlines():
            print(f"  [{ts}] {line}")

    def set_status(self, text: str, _color: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] STATUS → {text}")

    lbl_file = type("_Label", (), {"config": staticmethod(lambda **_: None)})()


# ---------------------------------------------------------------------------
# Runner logic
# ---------------------------------------------------------------------------

SCENARIOS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_scenarios.json")


def load_scenarios() -> list[dict]:
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_resume_file(scenario: dict, completed: dict[str, str | None]) -> str | None:
    """
    If the scenario declares 'resume_from_scenario', look up the final file
    produced by that scenario in `completed` and return its path.
    Returns None if the field is absent or the referenced scenario hasn't run yet.
    """
    ref = scenario.get("resume_from_scenario")
    if not ref:
        return None
    path = completed.get(ref)
    if not path:
        print(f"  ⚠️  resume_from_scenario: '{ref}' not found in completed scenarios — skipping resume.")
        return None
    if not os.path.isfile(path):
        print(f"  ⚠️  resume_from_scenario: file '{path}' no longer exists — skipping resume.")
        return None
    return path


def run_scenario(scenario: dict, app: StubApp, completed: dict[str, str | None]) -> bool:
    """
    Run a single scenario sequentially.
    completed: registry {scenario_name: final_file_path} for already-run scenarios.
    Returns True if all steps completed without error, False otherwise.
    """
    name  = scenario["scenario"]
    steps = scenario["steps"]

    print(f"\n{'═' * 60}")
    print(f"  SCENARIO: {name}")
    print(f"  {scenario.get('description', '')}")
    print(f"  Steps: {len(steps)}")
    print(f"{'═' * 60}")

    # Reset agent state, then optionally resume a previous scenario's file
    agent_config.agent_state["current_file"] = None
    agent_config.agent_state["last_section"] = None

    resume_path = resolve_resume_file(scenario, completed)
    if resume_path:
        agent_config.agent_state["current_file"] = resume_path
        print(f"  ⏮  Resuming: {os.path.relpath(resume_path, OUTPUT_DIR)}")

    errors: list[str] = []

    for i, step_text in enumerate(steps, start=1):
        print(f"\n  ── Step {i}/{len(steps)} ──────────────────────────────")
        print(f"  INPUT: \"{step_text[:80]}{'...' if len(step_text) > 80 else ''}\"")

        agent_config.text_queue.put(step_text)

        try:
            agent_config.text_queue.join()  # wait for agent_worker to call task_done()
        except Exception as e:
            errors.append(f"Step {i}: queue join failed — {e}")
            continue

    # Summary
    active = agent_config.agent_state.get("current_file")
    if active:
        rel = os.path.relpath(active, OUTPUT_DIR)
        print(f"\n  ✅ Final active file: {rel}")
        completed[name] = active
    else:
        print(f"\n  ⚠️  No active file at end of scenario.")
        completed[name] = None
        errors.append("No active file at end of scenario")

    if errors:
        print(f"  ❌ Errors encountered:")
        for e in errors:
            print(f"     • {e}")
        return False

    return True


def list_scenarios(scenarios: list[dict]) -> None:
    print("\nAvailable scenarios:")
    for s in scenarios:
        print(f"  • {s['scenario']}")
        print(f"    {s.get('description', '')}")
        print(f"    Steps: {len(s['steps'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Notes Agent — headless test runner")
    parser.add_argument("--scenario", "-s", help="Run only this scenario by name")
    parser.add_argument("--list",     "-l", action="store_true", help="List all scenarios and exit")
    args = parser.parse_args()

    scenarios = load_scenarios()

    if args.list:
        list_scenarios(scenarios)
        return

    if args.scenario:
        scenarios = [s for s in scenarios if s["scenario"] == args.scenario]
        if not scenarios:
            print(f"❌ Scenario not found: {args.scenario}")
            sys.exit(1)

    app = StubApp()

    worker_thread = threading.Thread(
        target=agent_worker, args=(app,), daemon=True
    )
    worker_thread.start()

    print(f"\n🚀 Running {len(scenarios)} scenario(s)  →  output: {OUTPUT_DIR}")
    print(f"   Make sure llama-server is running on port 12345.\n")

    passed   = 0
    failed   = 0
    completed: dict[str, str | None] = {}

    for scenario in scenarios:
        ok = run_scenario(scenario, app, completed)
        if ok:
            passed += 1
        else:
            failed += 1

    # Shut down the worker cleanly
    agent_config.text_queue.put(None)
    worker_thread.join(timeout=5)

    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"  Output files in: {OUTPUT_DIR}")
    print(f"{'═' * 60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
