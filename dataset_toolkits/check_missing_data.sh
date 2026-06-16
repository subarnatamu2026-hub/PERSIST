
#!/bin/bash

# Initialize variables
DATASET_PATH=""
DELETE_FLAG=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--delete) DELETE_FLAG=true ;;
        *) DATASET_PATH="$1" ;;
    esac
    shift
done

# Check if dataset path is provided
if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH is required"
    echo "Usage: $0 DATASET_PATH [-d]"
    exit 1
fi

# Check if raw/subdataset1 directory exists
RAW_DIR="$DATASET_PATH/raw"
if [ ! -d "$RAW_DIR" ]; then
    echo "Error: Directory $RAW_DIR does not exist"
    exit 1
fi

# Counter for missing files
MISSING_COUNT=0

# Check each subdirectory
for SUBDIR in "$RAW_DIR"/*/*/; do
    if [ -d "$SUBDIR" ]; then
        # Extract directory name
        DIR_NAME=$(basename "$SUBDIR")
        
        # Check if required files exist
        if [ ! -f "$SUBDIR/data.npz" ] || [ ! -f "$SUBDIR/level_metadata.json" ]; then
            echo "Warning: Directory $DIR_NAME is missing required files"
            MISSING_COUNT=$((MISSING_COUNT + 1))
            
            # Delete directory if flag is set
            if [ "$DELETE_FLAG" = true ]; then
                echo "Deleting directory $DIR_NAME"
                rm -rf "$SUBDIR"
            fi
        fi
    fi
done

# Summary
if [ $MISSING_COUNT -eq 0 ]; then
    echo "All directories contain required files"
else
    echo "Found $MISSING_COUNT directories with missing files"
    if [ "$DELETE_FLAG" = true ]; then
        echo "Deleted $MISSING_COUNT directories"
    else
        echo "Run with -d flag to delete these directories"
    fi
fi