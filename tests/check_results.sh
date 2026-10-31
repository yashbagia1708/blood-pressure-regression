#!/usr/bin/env bash
# Run the analysis headless and check the headline results haven't drifted.
set -euo pipefail
out=$(MPLBACKEND=Agg python regression.py 2>/dev/null)
echo "$out" | grep -qx "Best Degree: 2"                          || { echo "FAIL best degree"; exit 1; }
echo "$out" | grep -q  "Mean RMSE(Multiple Linear Regression): 8.50" || { echo "FAIL linear RMSE"; exit 1; }
echo "$out" | grep -q  "Mean RMSE (Ridge Regression): 8.50"        || { echo "FAIL ridge RMSE"; exit 1; }
test -s rmse_by_degree.png                                       || { echo "FAIL plot not saved"; exit 1; }
echo "ok    best degree 2, linear and ridge RMSE 8.51, plot saved"
