#!/bin/bash

# Determine directory where run.sh is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment if present
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Run the RAG orchestrator CLI with any arguments passed
python3 -m phase10.web_grounded_rag "$@"
