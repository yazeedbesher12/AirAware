from app.analysis.phase3_pipeline import run_phase3_pipeline


if __name__ == "__main__":
    result = run_phase3_pipeline()
    print(
        f"Phase 3 complete: {result['validation']['ml_feature_rows']} ML rows, "
        f"{result['feature_engineering']['total_candidate_features']} candidate features"
    )
