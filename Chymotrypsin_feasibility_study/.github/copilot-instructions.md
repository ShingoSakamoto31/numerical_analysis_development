# Chymotrypsin Feasibility Study - AI Coding Agent Instructions

## Project Overview
This is a clinical data analysis pipeline for Chymotrypsin feasibility studies. It processes fluorescence microscopy data from multi-well imaging experiments to classify cells into groups (CTRB1, CTRB2) and perform quality control on lane/sample data.

## Architecture

### Data Flow
1. **Raw Data Input**: `all_raw_data_clinical_study.csv` (master reference dataset with FITC and mCherry channel measurements)
2. **Processing Pipeline** (`main.py`):
   - Filters fields/wells by intensity thresholds (>5000 wells)
   - Computes derived metrics (green/red fluorescence from FITC/mCherry channels)
   - Runs K-means clustering in polar coordinate space to assign samples to groups
   - Applies QC thresholds to validate results
3. **Output**: `pd.Series` with 40+ metrics including clustering results, QC pass/fail status, and coefficient of variation values

### Key Components

- **`chymotrypsin_output.py`**: QC threshold standards (e.g., `LAMBDA_CV_STANDARD = 0.22`, `RED_MEAN_INTENSITY_STANDARD = 54000`) and result formatting into pandas Series
- **`clustering.py`**: K-means clustering with 5 clusters mapped to 3 biological groups via reference point matching; converts FITC/mCherry columns to green/red coordinates; handles both master reference and new sample data
- **`qc_conductor.py`**: Validates results against standards; generates diagnostic plots (scatter plots, CV trends, well counts) for visual QC assessment
- **`main.py`**: Orchestrates the full pipeline using `AnalyzeArgs`, `ParallelRunner` from external `analysisrun` package (git dependency)

## Critical Patterns

### Fluorescence Channel Transformation
- **Source columns**: `FITC_Sum/Max/Min`, `mCherry_Sum/Max/Min`
- **Derived values**: `green = FITC_Sum - (FITC_Max + FITC_Min)`, `red = mCherry_Sum - (mCherry_Max + mCherry_Min)`
- Applied consistently in clustering and QC logic

### Clustering Approach
- Uses **polar coordinates** (theta, r) from green/red values: `theta = arctan2(green, red)`, `r = sqrt(green² + red²)`
- K-means with 5 clusters; clusters mapped to reference points to standardize labeling across runs
- Output groups: 1=CTRB1 (cluster 0), 2=CTRB2_lambda (clusters 1-3), 3=CTRB2_sub (cluster 4)

### QC Standards Structure
- **Lane-level QC**: `LAMBDA_CV_STANDARD`, `INSUFFICIENT_WELL_FIELD_COUNT_STANDARD`, `RED_CV_STANDARD`, etc.
- **Reference QC (rQC)**: `RQC_TOTAL_LAMBDA_LOW/HIGH_STANDARD`, `RQC_CTRB1_PER_CTRB2_LOW/HIGH_STANDARD`
- **Positive QC (pQC)**: Separate thresholds for positive control samples
- All thresholds centralized in `chymotrypsin_output.py` for maintainability

### Data Filtering Rules
1. Drop rows with missing FITC/mCherry columns
2. Filter fields by well count: only fields with `All_well_count > 5000` are analyzed
3. Skip samples entirely if no fields meet the well count threshold (returns NG result)

## Dependencies

- **External**: `analysisrun` (v0.0.6, git dependency from GitHub)
- **Analysis Libraries**: pandas, numpy, scikit-learn, scipy, matplotlib
- **Python**: 3.13+

## Naming Conventions

- **Japanese comments** used in clustering and some utility functions (preserved for domain expertise)
- **Group numbering**: 1-based (group 1, 2, 3), not zero-indexed
- **Field/Well terminology**: "field" = image region; "well" = individual sample well within a field
- **QC results**: `"NG"` for fail, `None` for pass (not boolean)

## Common Tasks

**Adding new QC metrics**: Define threshold constant in `chymotrypsin_output.py`, compute in `qc_conductor.py`, add to `result()` Series

**Adjusting clustering**: Modify reference points array in `clustering.py`, `pairwise_distances_argmin()` call, or group assignment logic

**Changing filter thresholds**: Update well count or intensity cutoffs in `main.py` filter logic

**Debug tip**: Master CSV column validation uses explicit checks for `{green, red}` OR `{FITC_Sum/Max/Min, mCherry_Sum/Max/Min}` sets—if columns don't match, raises descriptive ValueError in Japanese
