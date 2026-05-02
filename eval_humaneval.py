"""
HumanEval evaluator for code-generation models.
"""
import json
import re
from human_eval.data import read_problems
from sandbox import execute_code


def extract_code(text: str) -> str:
    """Pull the function body out of model output."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def evaluate_completion(problem: dict, completion: str) -> bool:
    """Run HumanEval problem's tests against a model completion."""
    code = extract_code(completion)
    full_code = problem["prompt"] + "\n" + code if not code.startswith("def ") else code
    test_program = problem["test"] + f"\ncheck({problem['entry_point']})"
    result = execute_code(full_code, test_program, timeout=10.0)
    return result.passed


def run_eval(generate_fn, n_samples: int = 1, output_path: str = "results.jsonl"):
    """Run HumanEval. generate_fn(prompt) -> completion str."""
    problems = read_problems()
    results = []
    for task_id, problem in problems.items():
        completions = [generate_fn(problem["prompt"]) for _ in range(n_samples)]
        passes = [evaluate_completion(problem, c) for c in completions]
        results.append({
            "task_id": task_id,
            "completions": completions,
            "passes": passes,
        })
        n_passed = sum(passes)
        print(f"{task_id}: {n_passed}/{n_samples} passed")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(results)
    pass_at_1 = sum(r["passes"][0] for r in results) / n
    pass_at_k = sum(any(r["passes"]) for r in results) / n

    print(f"\nResults over {n} problems:")
    print(f"  pass@1: {pass_at_1:.4f}")
    if n_samples > 1:
        print(f"  pass@{n_samples}: {pass_at_k:.4f}")

    return {"pass@1": pass_at_1, f"pass@{n_samples}": pass_at_k}
