#!/bin/bash
# Ask questions to the REM RAG system

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the ask script with all arguments passed through
python scripts/query/ask_rag.py "$@"