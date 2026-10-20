"""Import finalized benchmark evidence without starting a trading bot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from olympus.core.execution_benchmark import (
    ExecutionBenchmarkLedger, build_benchmark_report, candidate_from_dict,
    observation_from_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("candidate", type=Path)
    observe = sub.add_parser("observe")
    observe.add_argument("observations", type=Path, help="Finalized JSONL; duplicate exact replays are ignored")
    report = sub.add_parser("report")
    report.add_argument("--source", required=True, choices=("prometheus", "hermes"))
    report.add_argument("--candidate", required=True)
    report.add_argument("--start", required=True)
    report.add_argument("--end", required=True)
    args = parser.parse_args()
    if args.database is None:
        from config import CONFIG, STORAGE_DIR
        args.database = STORAGE_DIR / "olympus" / CONFIG.execution_benchmark.database_name
    ledger = ExecutionBenchmarkLedger(args.database)
    if args.command == "register":
        candidate = candidate_from_dict(json.loads(args.candidate.read_text(encoding="utf-8")))
        print(json.dumps({"candidate_fingerprint": ledger.register(candidate)}))
    elif args.command == "observe":
        added = skipped = 0
        # Each accepted line is committed independently. If a later line fails,
        # earlier lines are retained; exact replay safely resumes the import.
        with args.observations.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    accepted = ledger.append(observation_from_dict(json.loads(line)))
                except (ValueError, TypeError, KeyError) as exc:
                    parser.exit(2, f"Line {line_number}: {exc}; earlier lines remain committed.\n")
                added += int(accepted)
                skipped += int(not accepted)
        print(json.dumps({"added": added, "exact_replays": skipped}))
    else:
        result = build_benchmark_report(ledger, args.source, args.candidate, args.start, args.end)
        print(json.dumps(result.as_dict(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
