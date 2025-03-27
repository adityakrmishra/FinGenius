#!/bin/bash
# scripts/start_simulator.sh

# Launch simulation environment
set -e

source venv/bin/activate

echo "🌌 Starting FinGenius simulator..."

# Configuration
NUM_SIMULATIONS=${1:-1000}
CONFIG_FILE=${2:-"config/simulation_defaults.yaml"}
LOG_DIR="logs/simulations"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/simulation_$TIMESTAMP.log"

# Create simulation directory
mkdir -p $LOG_DIR

# Run Docker services if available
if docker compose version &> /dev/null; then
    echo "🐳 Starting background services..."
    docker compose -f docker/docker-compose.yml up -d
fi

# Run simulation
python -m src.simulation.backtester \
    --config $CONFIG_FILE \
    --simulations $NUM_SIMULATIONS \
    | tee -a $LOG_FILE

echo "📈 Simulation completed! Results logged to $LOG_FILE"
