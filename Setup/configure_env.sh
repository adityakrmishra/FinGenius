#!/bin/bash

# configure_env.sh
# This script sets up environment variables for FinGenius.

set -e

echo "Creating .env file..."

cat <<EOL > .env
# FinGenius Environment Variables

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=fingenius_db

# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
OTHER_API_KEY=your_other_api_key

# Application Settings
DEBUG=True
SECRET_KEY=your_secret_key
EOL

echo ".env file created successfully."
