from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# AirAware
# Step 5A-2: TRUE LOCAL MODEL INFERENCE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "backend" / "models"

LGBM_1H_FILE = (
    MODEL_DIR
    / "lightgbm"
    / "pm25_1h.joblib"
)

LGBM_6H_FILE = (
    MODEL_DIR
    / "lightgbm"
    / "pm25_6h.joblib"
)

TFT_3H_CKPT = (
    MODEL_DIR
    / "tft"
    / "pm25_3h.ckpt"
)

TFT_3H_PREPROCESSING = (
    MODEL_DIR
    / "tft"
    / "pm25_3h.preprocessing.joblib"
)

SAVED_TEST_FILE = (
    BASE_DIR
    / "test_predictions_all_models.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ensemble"
    / "real_inference_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LGBM_OUTPUT = (
    OUTPUT_DIR
    / "lightgbm_true_inference_validation.csv"
)

TFT_OUTPUT = (
    OUTPUT_DIR
    / "tft_true_inference_validation.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "true_inference_summary.json"
)


# ============================================================
# Helpers
# ============================================================

def section(title):
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)


def normalize_horizon(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    mapping = {
        "1": "1h",
        "1h": "1h",
        "3": "3h",
        "3h": "3h",
        "6": "6h",
        "6h": "6h",
    }

    return mapping.get(
        value,
        value,
    )


def read_csv_safe(path):

    try:
        df = pd.read_csv(
            path,
            low_memory=False,
        )

        return df

    except Exception:
        return None


def prepare_timestamp(df):

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

    return df


# ============================================================
# Load model bundles
# ============================================================

def load_lightgbm_bundle(path):

    bundle = joblib.load(
        path
    )

    required = {
        "imputer",
        "models",
        "features",
    }

    if not isinstance(
        bundle,
        dict,
    ):
        raise TypeError(
            f"{path.name} is not a dictionary bundle."
        )

    missing = (
        required
        - set(bundle.keys())
    )

    if missing:
        raise ValueError(
            f"{path.name} missing keys: {missing}"
        )

    return bundle


def load_tft_preprocessing():

    preprocessing = joblib.load(
        TFT_3H_PREPROCESSING
    )

    required = {
        "features",
        "imputer",
        "scaler",
        "dataset_parameters",
    }

    missing = (
        required
        - set(preprocessing.keys())
    )

    if missing:
        raise ValueError(
            f"TFT preprocessing missing: {missing}"
        )

    return preprocessing


# ============================================================
# Search for engineered dataset
# ============================================================

def score_candidate_csv(
    path,
    required_features,
    extra_required=None,
):

    extra_required = (
        extra_required or []
    )

    try:

        header = pd.read_csv(
            path,
            nrows=3,
        )

    except Exception:

        return None

    columns = set(
        header.columns
    )

    feature_matches = sum(
        1
        for feature
        in required_features
        if feature in columns
    )

    extra_matches = sum(
        1
        for feature
        in extra_required
        if feature in columns
    )

    return {
        "path":
            path,

        "feature_matches":
            feature_matches,

        "feature_total":
            len(
                required_features
            ),

        "extra_matches":
            extra_matches,

        "columns":
            columns,
    }


def discover_engineered_csv(
    required_features,
    extra_required=None,
):

    section(
        "SEARCHING FOR ENGINEERED FEATURE DATASET"
    )

    ignored_parts = {
        ".venv",
        "node_modules",
        "ensemble",
    }

    candidates = []

    for path in BASE_DIR.rglob(
        "*.csv"
    ):

        if any(
            part in ignored_parts
            for part in path.parts
        ):
            continue

        result = score_candidate_csv(
            path,
            required_features,
            extra_required,
        )

        if result is not None:

            candidates.append(
                result
            )

    candidates.sort(
        key=lambda x: (
            x["feature_matches"],
            x["extra_matches"],
        ),
        reverse=True,
    )

    if not candidates:

        return None, []

    print(
        "\nBest dataset candidates:"
    )

    for item in candidates[:10]:

        print(
            f"{item['feature_matches']:>2}/"
            f"{item['feature_total']} features | "
            f"extra={item['extra_matches']} | "
            f"{item['path']}"
        )

    perfect = [
        item
        for item in candidates
        if (
            item[
                "feature_matches"
            ]
            == item[
                "feature_total"
            ]
        )
    ]

    if not perfect:

        return None, candidates

    return (
        perfect[0]["path"],
        candidates,
    )


# ============================================================
# LightGBM True Inference
# ============================================================
def lightgbm_predict(bundle, df):

    features = bundle["features"]

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing LightGBM features:\n"
            + "\n".join(missing)
        )

    X = df[features].copy()

    for col in features:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    X_imputed = bundle["imputer"].transform(X)

    # IMPORTANT:
    # models[0] = PM2.5 concentration target
    # models[1] = WHO 24h average target
    concentration_model = bundle["models"][0]

    prediction = concentration_model.predict(
        X_imputed
    )

    return np.asarray(
        prediction,
        dtype=float,
    )
# ============================================================
# Merge with saved TEST predictions
# ============================================================

def prepare_saved_test():

    df = pd.read_csv(
        SAVED_TEST_FILE
    )

    df = prepare_timestamp(
        df
    )

    df["horizon"] = (
        df["horizon"]
        .apply(
            normalize_horizon
        )
    )

    if "sensor_id" in df.columns:

        df["sensor_id"] = (
            df["sensor_id"]
            .astype(str)
        )

    return df


def detect_merge_keys(
    engineered,
    saved,
):

    preferred = [
        "timestamp",
        "sensor_id",
    ]

    if all(
        col in engineered.columns
        and col in saved.columns
        for col in preferred
    ):

        return preferred

    if (
        "timestamp"
        in engineered.columns
        and "timestamp"
        in saved.columns
    ):

        return [
            "timestamp"
        ]

    return []


def validate_lightgbm(
    engineered,
    saved,
    horizon,
    bundle,
):

    section(
        f"TRUE LIGHTGBM INFERENCE — {horizon}"
    )

    local = engineered.copy()

    local = prepare_timestamp(
        local
    )

    if "sensor_id" in local.columns:

        local["sensor_id"] = (
            local[
                "sensor_id"
            ]
            .astype(str)
        )

    local[
        "true_prediction"
    ] = lightgbm_predict(
        bundle,
        local,
    )

    saved_h = saved[
        saved[
            "horizon"
        ]
        == horizon
    ].copy()

    prediction_col = (
        "prediction_lightgbm"
    )

    merge_keys = (
        detect_merge_keys(
            local,
            saved_h,
        )
    )

    if not merge_keys:

        print(
            "Could not identify safe merge keys."
        )

        return None, None

    keep_cols = (
        merge_keys
        + [
            prediction_col,
            "actual",
        ]
    )

    keep_cols = [
        col
        for col in keep_cols
        if col in saved_h.columns
    ]

    merged = local.merge(
        saved_h[
            keep_cols
        ],
        on=merge_keys,
        how="inner",
    )

    if merged.empty:

        print(
            "No overlapping rows found."
        )

        return None, None

    merged[
        prediction_col
    ] = pd.to_numeric(
        merged[
            prediction_col
        ],
        errors="coerce",
    )

    valid = merged.dropna(
        subset=[
            "true_prediction",
            prediction_col,
        ]
    ).copy()

    valid[
        "difference"
    ] = (
        valid[
            "true_prediction"
        ]
        - valid[
            prediction_col
        ]
    )

    valid[
        "absolute_difference"
    ] = np.abs(
        valid[
            "difference"
        ]
    )

    metrics = {
        "horizon":
            horizon,

        "rows":
            int(
                len(valid)
            ),

        "mean_absolute_difference":
            float(
                valid[
                    "absolute_difference"
                ].mean()
            ),

        "max_absolute_difference":
            float(
                valid[
                    "absolute_difference"
                ].max()
            ),

        "mean_signed_difference":
            float(
                valid[
                    "difference"
                ].mean()
            ),

        "within_1e_6_fraction":
            float(
                (
                    valid[
                        "absolute_difference"
                    ]
                    <= 1e-6
                ).mean()
            ),

        "within_1e_4_fraction":
            float(
                (
                    valid[
                        "absolute_difference"
                    ]
                    <= 1e-4
                ).mean()
            ),

        "within_1e_2_fraction":
            float(
                (
                    valid[
                        "absolute_difference"
                    ]
                    <= 1e-2
                ).mean()
            ),
    }

    print(
        f"Compared rows          : "
        f"{metrics['rows']:,}"
    )

    print(
        f"Mean absolute diff     : "
        f"{metrics['mean_absolute_difference']:.8f}"
    )

    print(
        f"Maximum absolute diff  : "
        f"{metrics['max_absolute_difference']:.8f}"
    )

    print(
        f"Within 0.01            : "
        f"{metrics['within_1e_2_fraction'] * 100:.2f}%"
    )

    return metrics, valid


# ============================================================
# TFT Smoke Test
# ============================================================

def run_tft_smoke_test(
    engineered,
    preprocessing,
):

    section(
        "TRUE TFT 3h MODEL RELOAD"
    )

    try:

        import torch

        from pytorch_forecasting import (
            TimeSeriesDataSet,
        )

        from pytorch_forecasting.models import (
            TemporalFusionTransformer,
        )

    except Exception as exc:

        return {
            "status":
                "failed_import",

            "error":
                str(exc),
        }, None

    params = preprocessing[
        "dataset_parameters"
    ]

    required_columns = set()

    required_columns.update(
        params.get(
            "group_ids",
            [],
        )
    )

    time_idx = params.get(
        "time_idx"
    )

    if time_idx:

        required_columns.add(
            time_idx
        )

    required_columns.update(
        params.get(
            "static_categoricals",
            [],
        )
        or []
    )

    required_columns.update(
        params.get(
            "time_varying_known_reals",
            [],
        )
        or []
    )

    targets = params.get(
        "target",
        []
    )

    if isinstance(
        targets,
        str,
    ):
        targets = [
            targets
        ]

    required_columns.update(
        targets
    )

    missing = [
        col
        for col in required_columns
        if col not in engineered.columns
    ]

    if missing:

        print(
            "Dataset has LightGBM features, "
            "but is not sufficient for TFT."
        )

        print(
            "\nMissing TFT columns:"
        )

        for col in sorted(
            missing
        ):

            print(
                f"  - {col}"
            )

        return {
            "status":
                "missing_dataset_columns",

            "missing_columns":
                sorted(
                    missing
                ),
        }, None

    data = (
        engineered.copy()
    )

    data = prepare_timestamp(
        data
    )

    # --------------------------------------------------------
    # Correct data types
    # --------------------------------------------------------

    for categorical in (
        params.get(
            "static_categoricals",
            [],
        )
        or []
    ):

        data[
            categorical
        ] = (
            data[
                categorical
            ]
            .astype(str)
        )

    for group_id in (
        params.get(
            "group_ids",
            []
        )
    ):

        data[
            group_id
        ] = (
            data[
                group_id
            ]
            .astype(str)
        )

    # --------------------------------------------------------
    # Reconstruct dataset from SAVED parameters
    # --------------------------------------------------------

    try:

        prediction_dataset = (
            TimeSeriesDataSet.from_parameters(
                params,
                data,
                predict=True,
                stop_randomization=True,
            )
        )

    except Exception as exc:

        print(
            "TFT dataset reconstruction FAILED:"
        )

        print(
            exc
        )

        return {
            "status":
                "dataset_reconstruction_failed",

            "error":
                str(exc),
        }, None

    # --------------------------------------------------------
    # Reload actual checkpoint
    # --------------------------------------------------------

    try:

        model = (
            TemporalFusionTransformer
            .load_from_checkpoint(
                str(
                    TFT_3H_CKPT
                ),
                map_location="cpu",
            )
        )

        model.eval()

    except Exception as exc:

        print(
            "TFT checkpoint reload FAILED:"
        )

        print(
            exc
        )

        return {
            "status":
                "checkpoint_reload_failed",

            "error":
                str(exc),
        }, None

    dataloader = (
        prediction_dataset
        .to_dataloader(
            train=False,
            batch_size=64,
            num_workers=0,
        )
    )

    # --------------------------------------------------------
    # Actual model inference
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            prediction = model.predict(
                dataloader,
                mode="prediction",
                return_x=True,
            )

    except Exception as exc:

        print(
            "TFT inference FAILED:"
        )

        print(
            exc
        )

        return {
            "status":
                "inference_failed",

            "error":
                str(exc),
        }, None

    print(
        "TFT checkpoint reload: OK"
    )

    print(
        "TFT dataset rebuild   : OK"
    )

    print(
        "TFT model.predict()    : OK"
    )

    try:

        raw_output = (
            prediction.output
        )

        print(
            f"TFT output type       : "
            f"{type(raw_output)}"
        )

        print(
            f"TFT output shape/info : "
            f"{getattr(raw_output, 'shape', None)}"
        )

    except Exception:

        raw_output = None

    return {
        "status":
            "success",

        "dataset_sequences":
            int(
                len(
                    prediction_dataset
                )
            ),
    }, prediction


# ============================================================
# Main
# ============================================================

def main():

    section(
        "AIRAWARE — STEP 5A-2 TRUE LOCAL INFERENCE"
    )

    required_files = [
        LGBM_1H_FILE,
        LGBM_6H_FILE,
        TFT_3H_CKPT,
        TFT_3H_PREPROCESSING,
        SAVED_TEST_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing:\n{path}"
            )

    # ========================================================
    # Load packages
    # ========================================================

    lgbm_1h = (
        load_lightgbm_bundle(
            LGBM_1H_FILE
        )
    )

    lgbm_6h = (
        load_lightgbm_bundle(
            LGBM_6H_FILE
        )
    )

    tft_preprocessing = (
        load_tft_preprocessing()
    )

    print(
        f"LightGBM 1h features: "
        f"{len(lgbm_1h['features'])}"
    )

    print(
        f"LightGBM 6h features: "
        f"{len(lgbm_6h['features'])}"
    )

    print(
        f"TFT 3h features     : "
        f"{len(tft_preprocessing['features'])}"
    )

    # ========================================================
    # Discover feature dataset
    # ========================================================

    required_lgbm_features = (
        sorted(
            set(
                lgbm_1h[
                    "features"
                ]
            )
            |
            set(
                lgbm_6h[
                    "features"
                ]
            )
        )
    )

    tft_extra = [
        "series_id",
        "sensor_key",
        "tft_time_idx",
        "target_pm25_3h",
        "target_pm25_who_24h_avg_3h",
    ]

    dataset_path, candidates = (
        discover_engineered_csv(
            required_features=
                required_lgbm_features,

            extra_required=
                tft_extra,
        )
    )

    if dataset_path is None:

        section(
            "ENGINEERED DATASET NOT FOUND"
        )

        print(
            "No CSV currently contains all "
            "saved LightGBM training features."
        )

        print(
            "\nThis means we should NOT recreate "
            "the features manually yet."
        )

        print(
            "\nFind the Phase 3 engineered/forecast "
            "dataset or its feature-building code."
        )

        print(
            "\nTop candidate was:"
        )

        if candidates:

            top = candidates[0]

            print(
                top[
                    "path"
                ]
            )

            missing = [
                feature
                for feature
                in required_lgbm_features
                if feature
                not in top[
                    "columns"
                ]
            ]

            print(
                f"\nMissing "
                f"{len(missing)} features:"
            )

            for feature in missing:

                print(
                    f"  - {feature}"
                )

        return

    print(
        f"\nSelected feature dataset:\n"
        f"{dataset_path}"
    )

    engineered = (
        read_csv_safe(
            dataset_path
        )
    )

    engineered = (
        prepare_timestamp(
            engineered
        )
    )

    print(
        f"Rows loaded: "
        f"{len(engineered):,}"
    )

    saved_test = (
        prepare_saved_test()
    )

    # ========================================================
    # True LightGBM inference
    # ========================================================

    lgbm_1h_metrics, lgbm_1h_df = (
        validate_lightgbm(
            engineered=
                engineered,

            saved=
                saved_test,

            horizon=
                "1h",

            bundle=
                lgbm_1h,
        )
    )

    lgbm_6h_metrics, lgbm_6h_df = (
        validate_lightgbm(
            engineered=
                engineered,

            saved=
                saved_test,

            horizon=
                "6h",

            bundle=
                lgbm_6h,
        )
    )

    validation_frames = []

    if lgbm_1h_df is not None:

        temp = (
            lgbm_1h_df.copy()
        )

        temp[
            "horizon"
        ] = "1h"

        validation_frames.append(
            temp
        )

    if lgbm_6h_df is not None:

        temp = (
            lgbm_6h_df.copy()
        )

        temp[
            "horizon"
        ] = "6h"

        validation_frames.append(
            temp
        )

    if validation_frames:

        pd.concat(
            validation_frames,
            ignore_index=True,
        ).to_csv(
            LGBM_OUTPUT,
            index=False,
        )

    # ========================================================
    # TFT actual checkpoint smoke test
    # ========================================================

    tft_result, tft_prediction = (
        run_tft_smoke_test(
            engineered=
                engineered,

            preprocessing=
                tft_preprocessing,
        )
    )

    # ========================================================
    # Save summary
    # ========================================================

    summary = {
        "project":
            "AirAware",

        "phase":
            "Step 5A-2 True Local Inference",

        "engineered_dataset":
            str(
                dataset_path
            ),

        "lightgbm_1h":
            lgbm_1h_metrics,

        "lightgbm_6h":
            lgbm_6h_metrics,

        "tft_3h":
            tft_result,

        "important_note":
            (
                "This step reloads the actual saved "
                "model artifacts. LightGBM validation "
                "compares fresh model.predict() outputs "
                "against previously saved TEST predictions."
            ),
    }

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Final report
    # ========================================================

    section(
        "TRUE INFERENCE SUMMARY"
    )

    if lgbm_1h_metrics:

        print(
            "\n1h → LightGBM"
        )

        print(
            f"Mean diff: "
            f"{lgbm_1h_metrics['mean_absolute_difference']:.8f}"
        )

        print(
            f"Max diff : "
            f"{lgbm_1h_metrics['max_absolute_difference']:.8f}"
        )

    if tft_result:

        print(
            "\n3h → TFT"
        )

        print(
            f"Status: "
            f"{tft_result['status']}"
        )

    if lgbm_6h_metrics:

        print(
            "\n6h → LightGBM"
        )

        print(
            f"Mean diff: "
            f"{lgbm_6h_metrics['mean_absolute_difference']:.8f}"
        )

        print(
            f"Max diff : "
            f"{lgbm_6h_metrics['max_absolute_difference']:.8f}"
        )

    section(
        "FILES CREATED"
    )

    print(
        SUMMARY_OUTPUT
    )

    if validation_frames:

        print(
            LGBM_OUTPUT
        )

    section(
        "STEP 5A-2 COMPLETE"
    )


if __name__ == "__main__":
    main()