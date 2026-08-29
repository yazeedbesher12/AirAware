from app.analysis.pipeline import run_pipeline


if __name__ == "__main__":
    analysis = run_pipeline()
    overview = analysis["overview"]
    print(f"Pipeline complete: {overview['row_count']} rows, {overview['sensor_count']} sensors")

