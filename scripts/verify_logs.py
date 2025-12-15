#!/usr/bin/env python3
"""
Log verification script.

Analyzes backend.log and provides summary report.
Helps verify logging system works correctly.
"""

import re
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime


def parse_log_file(log_path: Path) -> list[dict]:
    """Parse log file and extract entries."""
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return []

    entries = []
    pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] \[(\w+)\] \[(.+?)\] (.+)"

    with open(log_path) as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                timestamp, level, module, message = match.groups()
                entries.append({
                    "timestamp": timestamp,
                    "level": level,
                    "module": module,
                    "message": message
                })

    return entries


def analyze_logs(entries: list[dict]) -> None:
    """Analyze log entries and print summary."""
    if not entries:
        print("❌ No log entries found")
        return

    print("=" * 80)
    print("📊 LOG ANALYSIS SUMMARY")
    print("=" * 80)

    # Count by level
    levels = Counter(e["level"] for e in entries)
    print(f"\n✅ Total entries: {len(entries)}")
    print(f"   - INFO: {levels.get('INFO', 0)}")
    print(f"   - WARNING: {levels.get('WARNING', 0)}")
    print(f"   - ERROR: {levels.get('ERROR', 0)}")
    print(f"   - CRITICAL: {levels.get('CRITICAL', 0)}")

    # Count by module
    modules = Counter(e["module"] for e in entries)
    print(f"\n📦 Log entries by module:")
    for module, count in modules.most_common(10):
        print(f"   - {module}: {count}")

    # Frontend vs Backend
    frontend_count = sum(1 for e in entries if "frontend" in e["module"])
    backend_count = len(entries) - frontend_count
    print(f"\n🔀 Log source:")
    print(f"   - Backend: {backend_count}")
    print(f"   - Frontend: {frontend_count}")

    # Recent errors
    errors = [e for e in entries if e["level"] in ("ERROR", "CRITICAL")]
    if errors:
        print(f"\n❌ Recent errors ({len(errors)} total):")
        for error in errors[-5:]:  # Last 5 errors
            print(f"   [{error['timestamp']}] {error['module']}")
            print(f"      {error['message'][:100]}...")
    else:
        print("\n✅ No errors found")

    # Recent warnings
    warnings = [e for e in entries if e["level"] == "WARNING"]
    if warnings:
        print(f"\n⚠️  Recent warnings ({len(warnings)} total):")
        for warning in warnings[-5:]:  # Last 5 warnings
            print(f"   [{warning['timestamp']}] {warning['module']}")
            print(f"      {warning['message'][:100]}...")
    else:
        print("\n✅ No warnings found")

    # Check for important operations
    print("\n🔍 Key operations logged:")
    operations = {
        "Project creations": sum(1 for e in entries if "Created project:" in e["message"]),
        "Site creations": sum(1 for e in entries if "Created site:" in e["message"]),
        "Deployment creations": sum(1 for e in entries if "Created deployment" in e["message"]),
        "Folder scans": sum(1 for e in entries if "Scanning folder" in e["message"]),
        "Deletions": sum(1 for e in entries if "Deleted" in e["message"]),
    }
    for op, count in operations.items():
        symbol = "✅" if count > 0 else "⚠️ "
        print(f"   {symbol} {op}: {count}")

    # Time range
    if entries:
        first_time = entries[0]["timestamp"]
        last_time = entries[-1]["timestamp"]
        print(f"\n📅 Time range:")
        print(f"   First entry: {first_time}")
        print(f"   Last entry: {last_time}")

    print("\n" + "=" * 80)


def check_log_health(entries: list[dict]) -> bool:
    """Check if logs look healthy."""
    print("\n🏥 LOG HEALTH CHECK")
    print("=" * 80)

    all_ok = True

    # Check 1: Are there any logs at all?
    if not entries:
        print("❌ FAIL: No log entries found")
        all_ok = False
    else:
        print(f"✅ PASS: Found {len(entries)} log entries")

    # Check 2: Are different log levels present?
    levels = set(e["level"] for e in entries)
    if "INFO" in levels:
        print("✅ PASS: INFO logs present")
    else:
        print("⚠️  WARN: No INFO logs found (might be ok)")

    # Check 3: Do we have both backend and frontend logs?
    has_backend = any("frontend" not in e["module"] for e in entries)
    has_frontend = any("frontend" in e["module"] for e in entries)

    if has_backend:
        print("✅ PASS: Backend logs present")
    else:
        print("❌ FAIL: No backend logs found")
        all_ok = False

    if has_frontend:
        print("✅ PASS: Frontend logs present")
    else:
        print("⚠️  WARN: No frontend logs found (might be ok if frontend not used yet)")

    # Check 4: Are there any critical errors?
    critical = [e for e in entries if e["level"] == "CRITICAL"]
    if critical:
        print(f"❌ FAIL: Found {len(critical)} CRITICAL errors")
        all_ok = False
    else:
        print("✅ PASS: No critical errors")

    # Check 5: Check for silent failure indicators
    silent_failures = [
        e for e in entries
        if any(phrase in e["message"].lower() for phrase in ["exception:", "traceback", "error:"])
        and e["level"] == "INFO"
    ]
    if silent_failures:
        print(f"⚠️  WARN: Found {len(silent_failures)} potential silent failures (errors logged as INFO)")
    else:
        print("✅ PASS: No silent failures detected")

    print("=" * 80)

    return all_ok


def main() -> int:
    """Main entry point."""
    # Find log file
    log_path = Path.home() / "AddaxAI" / "logs" / "backend.log"

    if not log_path.exists():
        print(f"❌ Log file not found at: {log_path}")
        print("\nMake sure the backend has been started at least once.")
        return 1

    print(f"📂 Analyzing log file: {log_path}")
    print(f"📏 File size: {log_path.stat().st_size / 1024:.2f} KB\n")

    # Parse logs
    entries = parse_log_file(log_path)

    # Analyze
    analyze_logs(entries)

    # Health check
    health_ok = check_log_health(entries)

    if health_ok:
        print("\n✅ Overall: Logging system appears healthy!")
        return 0
    else:
        print("\n❌ Overall: Issues found in logging system")
        return 1


if __name__ == "__main__":
    sys.exit(main())
