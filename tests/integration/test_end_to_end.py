# tests/integration/test_end_to_end.py
import pytest
import subprocess
import json

def test_full_workflow(tmp_path):
    # Test main pipeline execution
    result = subprocess.run(
        ['python', '-m', 'src.main', 
         '--goal', 'Save $50000 in 5 years',
         '--output', str(tmp_path / 'portfolio.json')],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    output_file = tmp_path / 'portfolio.json'
    assert output_file.exists()
    
    with open(output_file) as f:
        portfolio = json.load(f)
        assert 'allocations' in portfolio
        assert 'projections' in portfolio
