#!/bin/bash

# start_application.sh
# This script starts the FinGenius application.

set -e

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting the application..."
python src/main.py

echo "Deactivating virtual environment..."
deactivate
