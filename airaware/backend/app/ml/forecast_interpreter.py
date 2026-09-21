from __future__ import annotations


WHO_PM25_24H_GUIDELINE = 15.0


def interpret_who_pm25(
    pm25_24h_average: float,
) -> dict:

    value = float(
        pm25_24h_average
    )

    ratio = (
        value
        / WHO_PM25_24H_GUIDELINE
    )

    if value <= WHO_PM25_24H_GUIDELINE:

        status = (
            "Meets WHO Guideline"
        )

    else:

        status = (
            "Above WHO Guideline"
        )

    return {
        "pollutant":
            "PM2.5",

        "averaging_period":
            "24h",

        "forecast_24h_average":
            value,

        "guideline":
            WHO_PM25_24H_GUIDELINE,

        "unit":
            "µg/m³",

        "ratio_to_guideline":
            float(ratio),

        "status":
            status,

        "note":
            (
                "WHO interpretation is based on "
                "the predicted PM2.5 24-hour average, "
                "not the instantaneous PM2.5 forecast."
            ),
    }