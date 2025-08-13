#!/bin/bash

# Batch crystal structure generation script
# Usage: ./batch_generate.sh <folder_path> [num_samples] [batch_size]

# Check parameters
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <folder_path> [num_samples] [batch_size]"
    echo ""
    echo "Parameters:"
    echo "  folder_path  - Path to folder containing .pt model files"
    echo "  num_samples  - Number of samples to generate per model (default: 1000)"
    echo "  batch_size   - Batch size for generation (default: 32)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/models                    # Use default parameters"
    echo "  $0 /path/to/models 500               # Generate 500 samples"
    echo "  $0 /path/to/models 1000 64           # Generate 1000 samples with batch size 64"
    exit 1
fi

# Get parameters
MODEL_DIR="$1"
NUM_SAMPLES="${2:-1000}"
BATCH_SIZE="${3:-32}"

# Check if folder exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Folder '$MODEL_DIR' does not exist"
    exit 1
fi

# Check if generate_crystals.py exists
if [ ! -f "generate_crystals.py" ]; then
    echo "Error: generate_crystals.py not found. Please ensure script runs in correct directory"
    exit 1
fi

# Create output directory
FOLDER_NAME=$(basename "$MODEL_DIR")
OUTPUT_BASE_DIR="${FOLDER_NAME}_crystals"

if [ -d "$OUTPUT_BASE_DIR" ]; then
    echo "Output directory exists, will check and skip completed files: $OUTPUT_BASE_DIR"
else
    mkdir -p "$OUTPUT_BASE_DIR"
    echo "Created output directory: $OUTPUT_BASE_DIR"
fi

echo "Processing folder: $MODEL_DIR"
echo "Number of samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Finding all .pt files..."

# Find all .pt files
PT_FILES=($(find "$MODEL_DIR" -name "*.pt" -type f))

# Check if .pt files found
if [ ${#PT_FILES[@]} -eq 0 ]; then
    echo "Warning: No .pt files found in '$MODEL_DIR'"
    exit 1
fi

echo "Found ${#PT_FILES[@]} .pt files:"
for file in "${PT_FILES[@]}"; do
    echo " - $(basename "$file")"
done
echo ""

# Check current progress
echo "Checking current progress..."
completed_tasks=0
total_tasks=${#PT_FILES[@]}

for pt_file in "${PT_FILES[@]}"; do
    filename=$(basename "$pt_file" .pt)
    output_dir="$OUTPUT_BASE_DIR/${filename}_generated"
    
    if [ -d "$output_dir" ]; then
        completed_tasks=$((completed_tasks + 1))
    fi
done

remaining_tasks=$((total_tasks - completed_tasks))

echo "Progress summary:"
echo " - Total models: ${#PT_FILES[@]}"
echo " - Completed tasks: $completed_tasks"
echo " - Remaining tasks: $remaining_tasks"

if [ $remaining_tasks -eq 0 ]; then
    echo ""
    echo "🎉 All tasks completed! No further processing needed."
    exit 0
fi

echo ""
echo "Starting crystal structure generation..."

# Process each .pt file
current_index=0
total_files=${#PT_FILES[@]}

for pt_file in "${PT_FILES[@]}"; do
    current_index=$((current_index + 1))
    
    filename=$(basename "$pt_file" .pt)
    output_dir="$OUTPUT_BASE_DIR/${filename}_generated"
    
    echo "=========================================="
    echo "[$current_index/$total_files] Processing model: $filename"
    echo "Model file: $pt_file"
    echo "Output directory: $output_dir"
    
    # Check if already generated
    if [ -d "$output_dir" ]; then
        echo "⚠ Model already processed, skipping: $output_dir"
        echo ""
        continue
    fi
    
    echo "Generating $NUM_SAMPLES crystal structures..."
    echo "Command: python generate_crystals.py --checkpoint \"$pt_file\" --output_dir \"$output_dir\" --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE"
    
    # Record start time
    start_time=$(date +%s)
    
    # Execute generation command
    if python generate_crystals.py \
        --checkpoint "$pt_file" \
        --output_dir "$output_dir" \
        --num_samples $NUM_SAMPLES \
        --batch_size $BATCH_SIZE \
        --use_multi_gpu; then
        
        # Calculate duration
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✓ Model $filename completed! Duration: ${duration}s"
        echo "  Output directory: $output_dir"
        
        # Count generated files
        if [ -d "$output_dir" ]; then
            file_count=$(find "$output_dir" -name "*.cif" | wc -l)
            echo "  Generated CIF files: $file_count"
        fi
    else
        echo "✗ Model $filename generation failed"
        # Remove empty directory if generation failed
        [ -d "$output_dir" ] && rmdir "$output_dir" 2>/dev/null
    fi
    
    echo ""
done

echo "=========================================="
echo "All models processed!"

# Show final statistics
echo ""
echo "Generation results:"
echo "Total models processed: ${#PT_FILES[@]}"

success_count=0
total_structures=0

for pt_file in "${PT_FILES[@]}"; do
    filename=$(basename "$pt_file" .pt)
    output_dir="$OUTPUT_BASE_DIR/${filename}_generated"
    
    if [ -d "$output_dir" ]; then
        file_count=$(find "$output_dir" -name "*.cif" | wc -l 2>/dev/null || echo "0")
        if [ "$file_count" -gt 0 ]; then
            echo "✓ $filename: $file_count structures"
            success_count=$((success_count + 1))
            total_structures=$((total_structures + file_count))
        else
            echo "✗ $filename: Generation failed or no valid structures"
        fi
    else
        echo "✗ $filename: Output directory does not exist"
    fi
done

echo ""
echo "Successfully generated structures for: $success_count/${#PT_FILES[@]} models"
echo "Total crystal structures generated: $total_structures"
echo "All output files saved in: $OUTPUT_BASE_DIR"
echo ""
echo "Script execution completed!"
