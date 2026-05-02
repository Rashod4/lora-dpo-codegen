"""
Code execution sandbox for evaluating model-generated Python.
Runs candidate code in a subprocess with timeout and captures pass/fail.
"""
import subprocess
import tempfile
import os
from dataclasses import dataclass


@dataclass
class ExecResult:
    passed: bool
    error: str | None
    stdout: str
    stderr: str


def execute_code(code: str, test_code: str, timeout: float = 5.0) -> ExecResult:
    """Execute code + tests in a subprocess. Return pass/fail."""
    full_program = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_program)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return ExecResult(True, None, result.stdout, result.stderr)
        return ExecResult(False, f"non-zero exit {result.returncode}",
                          result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return ExecResult(False, f"timeout after {timeout}s", "", "")
    except Exception as e:
        return ExecResult(False, f"sandbox error: {type(e).__name__}: {e}", "", "")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    print("Test 1: correct code")
    r = execute_code(
        code="def add(a, b):\n    return a + b",
        test_code="assert add(2, 3) == 5",
    )
    print(f"  passed={r.passed}, error={r.error}")
    assert r.passed

    print("Test 2: wrong code")
    r = execute_code(
        code="def add(a, b):\n    return a - b",
        test_code="assert add(2, 3) == 5",
    )
    print(f"  passed={r.passed}, error={r.error}")
    assert not r.passed

    print("Test 3: infinite loop (should timeout)")
    r = execute_code(
        code="def loop():\n    while True: pass",
        test_code="loop()",
        timeout=2.0,
    )
    print(f"  passed={r.passed}, error={r.error}")
    assert not r.passed
    assert "timeout" in r.error

    print("Test 4: syntax error")
    r = execute_code(
        code="def broken(:\n    return 1",
        test_code="broken()",
    )
    print(f"  passed={r.passed}, error={r.error}")
    assert not r.passed

    print("\nAll sandbox self-tests passed.")
