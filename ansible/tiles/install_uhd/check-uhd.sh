#!/bin/bash

# Check if uhd_images_downloader is available
if ! command -v uhd_images_downloader >/dev/null 2>&1; then
    echo "Error: could not find 'uhd_images_downloader'."
    echo "Is UHD installed?"
    exit 1
fi

# If found, print the path for confirmation
echo "Running image downloader for B200-series..."

# Run the tool
uhd_images_downloader --types b2.*

# Check exit code
if [ $? -ne 0 ]; then
    echo "Error: uhd_images_downloader failed!"
    exit 2
fi

echo "Done downloading B200/B210 images."
exit 0
