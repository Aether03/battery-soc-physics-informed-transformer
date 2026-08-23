# Data

## What these files are

Four CSVs, one per cell, each a **12-cycle lifecycle sample** stitched from the
NASA PCoE Li-ion Battery Aging Dataset. Every row is real measured data. Nothing
here is synthetic or simulated.

| File | Cell | Rows | Span | Role |
|---|---|---|---|---|
| `Merged_B0005_Lifecycle_Sample.csv` | B0005 | 14,400 | 19.0 h | Training |
| `Merged_B0006_Lifecycle_Sample.csv` | B0006 | 14,400 | 19.0 h | Training |
| `Merged_B0007_Lifecycle_Sample.csv` | B0007 | 14,400 | 19.0 h | Validation |
| `Merged_B0018_Lifecycle_Sample.csv` | B0018 | 15,180 | 21.9 h | **Test, never seen during training or model selection** |

Columns: `Voltage_measured` (V), `Current_measured` (A), `Temperature_measured` (°C),
`Time` (s, monotonic across the stitched sequence).

## How they were built

`src/data_prep/build_lifecycle_sample.m` reads the cell's `metadata_<cell>.csv`
logbook, drops impedance cycles, and selects twelve charge/discharge cycles:
four from beginning-of-life, four from middle-of-life, four from end-of-life.
It then concatenates them on a continuous time axis, offsetting each file's
timestamps by one second past the end of the previous one.

Sampling across the life of the cell rather than taking the first N cycles is
deliberate: the physics constraint recalibrates capacity and internal resistance
as the cell ages, so the training data has to actually contain aged behaviour.
B0005's capacity falls from roughly 2.0 Ah fresh to the 1.4 Ah end-of-life
threshold across its 338 cycles, and a first-N-cycles sample would only ever
show the model a fresh cell.

`src/data_prep/merge_full_lifecycle.m` builds the larger continuous files used
for the full-life stress test. `Merged_B0005_338_Cycles.csv` (338 cycles, ~38 MB)
is **not** committed here. Regenerate it by setting `numFilesToMerge` to
`height(cycleData)` and running that script against the raw B0005 files.

## Getting the raw data

The originals are not redistributed here. Download them from NASA:

- Landing page: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Dataset record: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

You need the per-cycle CSV exports plus each cell's `metadata_<cell>.csv` in the
working directory before running the data-prep scripts.

## Experimental conditions

18650 lithium-ion cells, 2 Ah rated capacity, cycled in a 24 °C thermal chamber.
Charge at 1.5 A constant current to 4.2 V then constant voltage until current
falls below 20 mA; discharge at 2 A constant current to a per-cell voltage cutoff.
Cells are cycled to failure, defined as a 30% capacity fade (2.0 Ah → 1.4 Ah).

## Provenance and terms

Source: NASA Ames Prognostics Center of Excellence, Prognostics Data Repository.

The dataset carries no explicit licence tag on either NASA landing page. It is a
work of the United States Government, which under 17 U.S.C. § 105 is not subject
to copyright protection in the United States, and NASA publishes it as open data.
Attribution is expected. If you use this data, cite the original:

> B. Saha and K. Goebel, "Battery Data Set," NASA Ames Prognostics Data
> Repository, NASA Ames Research Center, Moffett Field, CA, 2007.

The derived CSVs in this directory are redistributed with the transformation
script alongside them so the derivation is fully reproducible and auditable
against the originals. They are **not** covered by this repository's MIT or
CC BY 4.0 licences.
