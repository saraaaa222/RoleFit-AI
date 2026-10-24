from pathlib import Path
import ast
import sys

ROOT = Path(__file__).resolve().parent
errors = []

print("RoleFit AI project validation")
print("=" * 32)

required_files = [
    "rolefit_app.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
]

for filename in required_files:
    if (ROOT / filename).is_file():
        print(f"PASS: Found {filename}")
    else:
        print(f"FAIL: Missing {filename}")
        errors.append(filename)

app_path = ROOT / "rolefit_app.py"

if app_path.is_file():
    try:
        ast.parse(app_path.read_text(encoding="utf-8"))
        print("PASS: Python syntax")
    except SyntaxError as exc:
        print(f"FAIL: Python syntax — {exc}")
        errors.append("Python syntax")

data_path = ROOT / "jobs_dataset.csv"

if data_path.is_file():
    try:
        import pandas as pd

        df = pd.read_csv(data_path, nrows=5)
        required_columns = {
            "job_title",
            "job_description",
            "company_name",
        }
        missing = required_columns - set(df.columns)

        if missing:
            print(f"FAIL: Dataset missing columns: {sorted(missing)}")
            errors.append("Dataset schema")
        else:
            print("PASS: Dataset schema")
    except Exception as exc:
        print(f"FAIL: Dataset could not be read — {exc}")
        errors.append("Dataset read")
else:
    print("INFO: Dataset not found locally; dataset check skipped.")

if errors:
    print(f"\nRESULT: FAILED — {len(errors)} issue(s) need attention.")
    sys.exit(1)

print("\nRESULT: ALL CHECKS PASSED")
