# tests/unit/test_nlp.py
import pytest
from src.nlp_processing import goal_parser, risk_assessor

@pytest.mark.parametrize("input_text,expected", [
    ("I want to save $50k in 3 years", (50000, 3, 'moderate')),
    ("Retire with 1 million in 15 years (low risk)", (1000000, 15, 'conservative')),
    ("Grow 500k to 2M quickly with high risk", (500000, 5, 'aggressive')),
])
def test_goal_parsing(input_text, expected):
    parser = goal_parser.GoalInterpreter()
    result = parser.parse(input_text)
    
    assert result['target_amount'] == expected[0]
    assert result['time_horizon'] == expected[1]
    assert result['risk_level'] == expected[2]

def test_risk_assessment_scoring():
    questionnaire = risk_assessor.RiskQuestionnaire()
    test_answers = {
        'time_horizon': '10+ years',
        'risk_tolerance': 'High',
        'loss_reaction': 'Invest more'
    }
    
    score = questionnaire.calculate_score(test_answers)
    profile = questionnaire.determine_profile(score)
    
    assert score >= 25
    assert profile == 'aggressive'
