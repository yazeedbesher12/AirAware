# AirAware Phase 3 - WHO 2021 Integration Report

Source: WHO global air quality guidelines: particulate matter, ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide (WHO 2021)  
Official publication: https://www.who.int/publications/i/item/9789240034228

## Guideline configuration

The short-term configurations use PM2.5 24-hour 15 ug/m3, PM10 24-hour 45 ug/m3, NO2 24-hour 25 ug/m3, O3 daily maximum 8-hour 100 ug/m3, SO2 24-hour 40 ug/m3, and CO 24-hour 4 mg/m3. Interim targets are stored separately in the versioned JSON configuration.

## Unit and availability audit

- Supported comparison: PM25.
- Present but not comparable because source units are undocumented: NO2, O3.
- PM10, SO2, and CO are not present in this dataset.
- NO2/O3 remain structurally unavailable for Nablus and are never fabricated.

## Completeness rule

A rolling WHO reference window requires at least 75% of expected 15-minute readings and cannot cross a major outage. Early and fragmented windows are `Insufficient Data`.

## PM2.5 findings

- Valid 24-hour sensor windows: 5,280.
- Windows above the 15 ug/m3 reference: 3,826 (72.46%).

These are overlapping rolling-window health-reference screening results. WHO's short-term AQG also includes annual exceedance-frequency context, which this 17-day dataset cannot evaluate. AirAware does not call this a WHO AQI or a legal compliance result.
