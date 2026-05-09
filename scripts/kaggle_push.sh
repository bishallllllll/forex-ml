#!/usr/bin/env bash
set -euo pipefail

# Kaggle Push Script — pushes all 5 notebooks sequentially
# Requires: pip install kaggle, ~/.kaggle/kaggle.json configured
#
# Usage:
#   First:  Edit scripts/kaggle_metadata/*.json — replace "your-kaggle-username" with your actual username
#   Then:   bash scripts/kaggle_push.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METADATA_DIR="$SCRIPT_DIR/kaggle_metadata"

# Check kaggle CLI
if ! command -v kaggle &>/dev/null; then
    echo "Installing kaggle CLI..."
    pip install kaggle --quiet
fi

# Check credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found"
    echo "Download from https://www.kaggle.com/settings -> API -> Create New Token"
    exit 1
fi

# Push order: N1 -> N2 -> N3 -> N4 -> N5 (sequential due to kernel_sources dependency)
NOTEBOOKS=(
    "01_Data_Ingestion"
    "02_Feature_Engineering"
    "03_Model_Training"
    "04_Backtesting_Evaluation"
    "05_Daily_Signal"
)

for notebook in "${NOTEBOOKS[@]}"; do
    metadata="$METADATA_DIR/$notebook.json"
    if [ ! -f "$metadata" ]; then
        echo "WARNING: Metadata not found for $notebook — skipping"
        continue
    fi

    echo ""
    echo "=============================================="
    echo "Pushing: $notebook"
    echo "=============================================="

    # Create a temp dir so kaggle CLI picks up kernel-metadata.json as the metadata file
    TMP_DIR=$(mktemp -d)
    cp "$metadata" "$TMP_DIR/kernel-metadata.json"
    # Copy the notebook with a fixed name (remove ../.. prefix from code_file path)
    code_file=$(python3 -c "import json; print(json.load(open('$metadata'))['code_file'])")
    notebook_path="$SCRIPT_DIR/../$code_file"
    cp "$notebook_path" "$TMP_DIR/"

    cd "$TMP_DIR"
    kaggle kernels push
    cd "$OLDPWD"
    rm -rf "$TMP_DIR"

    echo "✅ $notebook pushed"
done

echo ""
echo "=============================================="
echo "All 5 notebooks pushed!"
echo "=============================================="
echo ""
echo "Next steps on Kaggle:"
echo "  1. Run N1 → wait for completion"
echo "  2. Run N2 → wait for completion (reads N1 output)"
echo "  3. Run N3 (GPU) → wait for completion (reads N2 output)"
echo "  4. Run N4 → wait for completion (reads N2+N3 outputs)"
echo "  5. Run N5 daily (reads N3 model)"
