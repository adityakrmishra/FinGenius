# Simulation Results

- Backtesting results (CSV)
- Portfolio performance metrics (JSON)
- Risk analysis reports (PDF/HTML)

- ```
  import os

required_dirs = [
    'data/raw/stocks',
    'data/raw/crypto',
    'data/processed',
    'data/simulations'
]

def check_directory_structure():
    missing = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(dir_path)
    
    if missing:
        print("Missing directories:")
        print('\n'.join(missing))
        raise SystemExit(1)
    print("All required directories exist")

if __name__ == "__main__":
    check_directory_structure()
    ```
