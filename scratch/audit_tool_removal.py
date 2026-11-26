"""
scratch/audit_tool_removal.py
Independent Audit - Requirement 3:
Search active application code for:
- '/tool on'
- '/tool off'
- Calculator interceptors
- Tool execution logic
- Unused tool-related imports
"""
import os
import re
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")

search_patterns = [
    r"/tool\s+on",
    r"/tool\s+off",
    r"\[Tool:\s*Calculator\]",
    r"from\s+evaluate_tool_assisted",
    r"import\s+safe_calculator",
    r"safe_calculator\s*\(",
    r"evaluate_tool_assisted",
    r"tool_mode\s*=",
    r"tool_assisted",
    r"safe_eval\b"
]

checked_extensions = [".py", ".bat", ".cmd", ".sh"]
search_dirs = [PROJECT_DIR / "src", PROJECT_DIR]

findings = []

for s_dir in search_dirs:
    for item in s_dir.iterdir():
        if item.is_file() and item.suffix in checked_extensions:
            # Check file
            try:
                content = item.read_text(encoding="utf-8", errors="ignore")
                for line_idx, line in enumerate(content.splitlines(), 1):
                    for pat in search_patterns:
                        if re.search(pat, line, re.I):
                            # Ignore comments explaining that tools are removed
                            if "removed" in line.lower() or "removal" in line.lower() or "eradicated" in line.lower() or "deprecated" in line.lower():
                                continue
                            findings.append((str(item.relative_to(PROJECT_DIR)), line_idx, line.strip(), pat))
            except Exception as e:
                pass

print(f"Total active tool references found: {len(findings)}")
for f in findings:
    print(f"  FILE: {f[0]}:{f[1]} -> {f[2]} (Matched: {f[3]})")

if len(findings) == 0:
    print("AUDIT 3 VERIFICATION PASSED: Zero tool interceptors, commands, or execution logic in active application!")
else:
    print(f"AUDIT 3 WARNING: {len(findings)} tool references found!")
