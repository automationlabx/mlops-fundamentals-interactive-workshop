from __future__ import annotations

import os
from pathlib import Path

HOOK_PATH = Path(".git/hooks/pre-commit")


def main() -> None:
    if not Path(".git").exists():
        raise SystemExit(".git directory not found. Run: git init")

    HOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    HOOK_PATH.write_text(
        "#!/bin/sh\n"
        "echo 'Running MLOps workshop local quality gate...'\n"
        "python run_checks.py\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        os.chmod(HOOK_PATH, 0o755)
    except OSError:
        pass
    print(f"Installed pre-commit hook: {HOOK_PATH.as_posix()}")


if __name__ == "__main__":
    main()
