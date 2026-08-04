#!/bin/bash
# Download RoboCasa assets required for the inhand_transfer environment.
# Only fetches the two packs needed (~220MB unzipped):
#   - objects_lightwheel  (dish_brush, whisk, salt_and_pepper_shaker, ...)
#   - objaverse           (rolling_pin, water_bottle, can, ladle)
#
# Run from this directory:
#   cd yam_sim/models/assets_robocasa
#   bash download_assets.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PACK_NAMES=(
    "objects_lightwheel"
    "objaverse"
)

PACK_IDS=(
    "vckqvvkh1z8t69k8qcpcmee6k66stii4"
    "03eionyo8fk3a9dsksq9jb8du5lqfw8h"
)

download_file() {
    local url="$1"
    local output="$2"

    if command -v wget >/dev/null 2>&1; then
        wget -O "$output" "$url"
        return
    fi

    if command -v curl >/dev/null 2>&1; then
        curl -L "$url" -o "$output"
        return
    fi

    echo "Error: need either wget or curl to download assets." >&2
    exit 1
}

for i in "${!PACK_NAMES[@]}"; do
    name="${PACK_NAMES[$i]}"
    id="${PACK_IDS[$i]}"
    zip_file="${name}.zip"
    url="https://utexas.box.com/shared/static/${id}.zip"

    if [ -d "$name" ]; then
        echo "--- Skipping ${name} (already exists) ---"
        continue
    fi

    echo "--- Downloading ${name} ---"
    download_file "$url" "$zip_file"

    echo "--- Unzipping ${name} ---"
    mkdir -p "$name"
    unzip "$zip_file" -d "$name"
    rm "$zip_file"
done

echo "Done."
