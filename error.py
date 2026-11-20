import joblib
from pathlib import Path

files = [
    "model1.pkl",
    "model2.pkl",
    "model3.pkl",
    "model4.pkl",
    "corr_matrix.joblib"
]

for filename in files:

    path = Path(filename)

    if not path.exists():
        print(f"❌ {filename}: FILE NOT FOUND")
        continue

    print(f"\nChecking: {filename}")
    print(f"Size: {path.stat().st_size} bytes")

    try:
        obj = joblib.load(path)
        print(f"✅ Successfully loaded: {type(obj)}")

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")