# WRDS Local Data Policy

WRDS/CRSP files are licensed academic data. Keep them local/private.

Do not commit these files to GitHub:

```text
data/raw/point_in_time_market_caps.csv
data/raw/wrds_crsp_daily_stock_file.csv
data/raw/wrds_crsp_monthly_stock_file.csv
data/raw/wrds_crsp_names.csv
data/raw/wrds_crsp_delisting_information.csv
data/processed/*.parquet
data/cache/*.parquet
```

Safe to commit:

```text
data/raw/README.md
data/raw/example_point_in_time_market_caps.csv
.gitkeep files
source code
configs
docs
tests
```

The real WRDS-derived files should remain on your local machine or private encrypted storage only.
