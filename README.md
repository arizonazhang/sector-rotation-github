## Folder Composition
- weights: monthly and weekly allocation calculated by our programs
- plots: plots
- regre: regression results from our programs
- performance: portfolio performance based on monthly and weekly allocation calculated by our programs
- input: factor return, sector return data (cleaned data are stored in input/factor and input/sector)

## Codes
- data cleaning: `weekly_data_convert.ipynb` (factor and sector data cleaning)
- porfolio allocation: `portfolio_quadprog.py` (imports `utils.py`)
- visualization: `factor_summary.ipynb` and `allocation_summary.ipynb`
- database upload: `db_upload_helper.ipynb` and (imports `utils_db_upload.py`)
- model validation: `model validation.ipynb` (for my own use, no comments added yet)

common variables:
- `sector_indices`: dj (Dow Jones), csi (China Security Index), hsci (Hong Kong Composite Index)
- `markets`: us (U.S.), cn (Shanghai-Shenzhen), hk (Hong Kong)
- `window`: number of records to use for regression

## Plot Naming Rules:
- factor summary: `factor_(market code)_5.png `
- factor summary for long, short: `factor_(market code)_(long/short)_5.png `
- performance summary: `performance_(market code)_5.png`
- allocation summary: `allocation_(market code)_(method)_weekly.png`
- allocation comparison: `allocation_cmp_(market code)_(method)_weekly.png`

## TODO later:
- remove `monthly` folder and related variables
