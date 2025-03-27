#!/bin/bash

# install_dependencies.sh
# This script installs necessary system packages and Python dependencies for FinGenius.

set -e

echo "Updating package lists..."
sudo apt-get update

echo "Installing system dependencies..."
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev build-essential libssl-dev libffi-dev

echo "Creating virtual environment..."
python3.9 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Deactivating virtual environment..."
deactivate

echo "Installation complete."
