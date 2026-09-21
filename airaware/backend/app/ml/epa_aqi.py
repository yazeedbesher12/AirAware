import math


PM25_BREAKPOINTS = [
    (0.0,   9.0,   0,   50,  "Good"),
    (9.1,   35.4,  51,  100, "Moderate"),
    (35.5,  55.4,  101, 150, "Unhealthy for Sensitive Groups"),
    (55.5,  125.4, 151, 200, "Unhealthy"),
    (125.5, 225.4, 201, 300, "Very Unhealthy"),
    (225.5, 325.4, 301, 500, "Hazardous"),
]


def truncate_pm25(value: float) -> float:
    return math.floor(float(value) * 10.0) / 10.0


def pm25_to_aqi(pm25_24h: float) -> dict:

    concentration = max(
        0.0,
        truncate_pm25(pm25_24h)
    )

    for c_low, c_high, i_low, i_high, category in PM25_BREAKPOINTS:

        if c_low <= concentration <= c_high:

            aqi = (
                (i_high - i_low)
                / (c_high - c_low)
                * (concentration - c_low)
                + i_low
            )

            return {
                "aqi": int(round(aqi)),
                "category": category,
                "dominant_pollutant": "PM2.5",
                "pm25_24h_average": concentration,
                "unit": "µg/m³",
            }

    return {
        "aqi": 500,
        "category": "Hazardous",
        "dominant_pollutant": "PM2.5",
        "pm25_24h_average": concentration,
        "unit": "µg/m³",
    }