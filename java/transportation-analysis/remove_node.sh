#!/bin/bash

# Script to remove a node with coordinates "26.643221256222105, 38.319147501994145" from a graph JSON file
# and also remove all edges connected to that node

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq to run this script."
    echo "You can install it with: brew install jq"
    exit 1
fi

# Check if input file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <json_file>"
    echo "Example: $0 graph_data/graph_100_gabriel.json"
    exit 1
fi

INPUT_FILE=$1

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

echo "Starting to process file: $INPUT_FILE"

# Create a timestamp for the new file name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${INPUT_FILE%.*}_without_node_${TIMESTAMP}.json"

echo "Will create new file: $OUTPUT_FILE"

# Find node ID with the specific coordinates
NODE_ID=$(jq -r '.nodes[] | select(.x == 26.643221256222105 and .y == 38.319147501994145) | .id' "$INPUT_FILE")

if [ -z "$NODE_ID" ]; then
    echo "Error: Node with coordinates (26.643221256222105, 38.319147501994145) not found in the file."
    exit 1
fi

echo "Found node with ID: $NODE_ID to remove"

# Get original counts for logging
ORIGINAL_NODE_COUNT=$(jq '.nodeCount' "$INPUT_FILE")
ORIGINAL_EDGE_COUNT=$(jq '.edges | length' "$INPUT_FILE")

# Create a new JSON with the node removed and edges filtered
jq --arg node_id "$NODE_ID" '
    # Remove the node with the given ID
    .nodes = [.nodes[] | select(.id != $node_id)] |
    
    # Remove edges connected to the node
    .edges = [.edges[] | select(.sourceId != $node_id and .targetId != $node_id)] |
    
    # Update node count
    .nodeCount = (.nodeCount - 1)
' "$INPUT_FILE" > "$OUTPUT_FILE"

# Check if the operation was successful
if [ $? -eq 0 ]; then
    NEW_NODE_COUNT=$(jq '.nodeCount' "$OUTPUT_FILE")
    NEW_EDGE_COUNT=$(jq '.edges | length' "$OUTPUT_FILE")
    
    REMOVED_EDGES=$((ORIGINAL_EDGE_COUNT - NEW_EDGE_COUNT))
    
    echo "Operation completed successfully."
    echo "Original node count: $ORIGINAL_NODE_COUNT"
    echo "New node count: $NEW_NODE_COUNT"
    echo "Original edge count: $ORIGINAL_EDGE_COUNT"
    echo "New edge count: $NEW_EDGE_COUNT"
    echo "Removed $REMOVED_EDGES edges connected to node $NODE_ID"
    echo "New file created: $OUTPUT_FILE"
else
    echo "Error: Failed to process the file."
    exit 1
fi

echo "Done!" 