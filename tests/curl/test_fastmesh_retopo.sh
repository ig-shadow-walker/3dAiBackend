#!/bin/bash
# Test script for FastMesh Retopology API
# This script tests mesh retopology using FastMesh V1K and V4K variants

set -e  # Exit on error

# Configuration
SERVER_URL="${SERVER_URL:-http://localhost:7842}"
API_BASE="$SERVER_URL/api/v1"
TEST_MESH="${TEST_MESH:-assets/example_retopo/001.obj}"
OUTPUT_DIR="./test_outputs/fastmesh_retopology"
POLL_INTERVAL=5  # seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}=== FastMesh Retopology API Test ===${NC}"
echo "Server: $SERVER_URL"
echo "Test mesh: $TEST_MESH"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if test mesh exists
if [ ! -f "$TEST_MESH" ]; then
    echo -e "${RED}Error: Test mesh not found: $TEST_MESH${NC}"
    echo "Please set TEST_MESH environment variable to a valid mesh file"
    exit 1
fi

# Function to poll job status
poll_job() {
    local job_id=$1
    local timeout=600  # 10 minutes
    local elapsed=0
    
    echo "Polling job status (job_id: $job_id)..."
    
    while [ $elapsed -lt $timeout ]; do
        response=$(curl -s "$API_BASE/system/jobs/$job_id")
        status=$(echo "$response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        
        echo "  Status: $status (${elapsed}s elapsed)"
        
        if [ "$status" = "completed" ]; then
            echo -e "${GREEN}Job completed successfully!${NC}"
            return 0
        elif [ "$status" = "failed" ] || [ "$status" = "error" ]; then
            echo -e "${RED}Job failed!${NC}"
            echo "Response: $response"
            return 1
        fi
        
        sleep $POLL_INTERVAL
        elapsed=$((elapsed + POLL_INTERVAL))
    done
    
    echo -e "${RED}Timeout waiting for job completion${NC}"
    return 1
}

# Function to download result
download_result() {
    local job_id=$1
    local output_file=$2
    
    echo "Downloading result..."
    curl -s "$API_BASE/system/jobs/$job_id/download" -o "$output_file"
    
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        echo -e "${GREEN}Downloaded: $output_file${NC}"
        ls -lh "$output_file"
        return 0
    else
        echo -e "${RED}Failed to download result${NC}"
        return 1
    fi
}

# Test 1: Upload mesh file
echo -e "\n${YELLOW}=== Step 1: Upload Mesh File ===${NC}"
upload_response=$(curl -s -X POST "$API_BASE/file-upload/mesh" \
    -F "file=@$TEST_MESH")

echo "Upload response: $upload_response"

file_id=$(echo "$upload_response" | grep -o '"file_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$file_id" ]; then
    echo -e "${RED}Failed to upload mesh file${NC}"
    exit 1
fi

echo -e "${GREEN}File uploaded successfully: $file_id${NC}"

# Test 2: FastMesh V1K Retopology
echo -e "\n${YELLOW}=== Step 2: FastMesh V1K Retopology (~1000 vertices) ===${NC}"
retopo_v1k_response=$(curl -s -X POST "$API_BASE/mesh-retopology/retopologize-mesh" \
    -H "Content-Type: application/json" \
    -d "{
        \"mesh_file_id\": \"$file_id\",
        \"model_preference\": \"fastmesh_v1k_retopology\",
        \"output_format\": \"obj\",
        \"seed\": 42
    }")

echo "Retopology V1K response: $retopo_v1k_response"

job_id_v1k=$(echo "$retopo_v1k_response" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$job_id_v1k" ]; then
    echo -e "${RED}Failed to submit V1K retopology job${NC}"
    exit 1
fi

echo -e "${GREEN}V1K Retopology job submitted: $job_id_v1k${NC}"

# Poll job status
if poll_job "$job_id_v1k"; then
    download_result "$job_id_v1k" "$OUTPUT_DIR/retopo_v1k.obj"
else
    echo -e "${RED}V1K Retopology job failed${NC}"
    exit 1
fi

# Test 3: FastMesh V4K Retopology
echo -e "\n${YELLOW}=== Step 3: FastMesh V4K Retopology (~4000 vertices) ===${NC}"
retopo_v4k_response=$(curl -s -X POST "$API_BASE/mesh-retopology/retopologize-mesh" \
    -H "Content-Type: application/json" \
    -d "{
        \"mesh_file_id\": \"$file_id\",
        \"model_preference\": \"fastmesh_v4k_retopology\",
        \"output_format\": \"glb\",
        \"seed\": 42
    }")

echo "Retopology V4K response: $retopo_v4k_response"

job_id_v4k=$(echo "$retopo_v4k_response" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$job_id_v4k" ]; then
    echo -e "${RED}Failed to submit V4K retopology job${NC}"
    exit 1
fi

echo -e "${GREEN}V4K Retopology job submitted: $job_id_v4k${NC}"

# Poll job status
if poll_job "$job_id_v4k"; then
    download_result "$job_id_v4k" "$OUTPUT_DIR/retopo_v4k.glb"
else
    echo -e "${RED}V4K Retopology job failed${NC}"
    exit 1
fi

# Test 4: Check available models
echo -e "\n${YELLOW}=== Step 4: Check Available Models ===${NC}"
models_response=$(curl -s "$API_BASE/mesh-retopology/available-models")
echo "Available models:"
echo "$models_response" | python3 -m json.tool 2>/dev/null || echo "$models_response"

# Test 5: Check supported formats
echo -e "\n${YELLOW}=== Step 5: Check Supported Formats ===${NC}"
formats_response=$(curl -s "$API_BASE/mesh-retopology/supported-formats")
echo "Supported formats:"
echo "$formats_response" | python3 -m json.tool 2>/dev/null || echo "$formats_response"

# Summary
echo -e "\n${GREEN}=== Test Summary ===${NC}"
echo "✓ Mesh file uploaded"
echo "✓ V1K retopology completed: $OUTPUT_DIR/retopo_v1k.obj"
echo "✓ V4K retopology completed: $OUTPUT_DIR/retopo_v4k.glb"
echo "✓ Available models retrieved"
echo "✓ Supported formats retrieved"
echo ""
echo -e "${GREEN}All FastMesh retopology tests passed!${NC}"
