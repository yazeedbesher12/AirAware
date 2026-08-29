from app.analysis.phase2_pipeline import run_phase2_pipeline


if __name__ == "__main__":
    result = run_phase2_pipeline()
    print(f"Phase 2 complete: {result['validation']['rows']} rows, {len(result['artifacts'])} machine-readable artifacts")
