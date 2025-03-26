"""
Financial Risk Assessor - Analyze risk tolerance from natural language input
"""

from typing import Dict, Tuple
import spacy
from pydantic import BaseModel
import numpy as np

nlp = spacy.load("en_core_web_sm")

class RiskProfile(BaseModel):
    risk_tolerance: float = Field(..., ge=0, le=1)
    risk_capacity: float = Field(..., ge=0, le=1)
    risk_labels: Dict[str, float]
    keywords: Dict[str, int]
    sentiment: float

class RiskAssessor:
    """Advanced risk assessment using NLP and financial heuristics"""
    
    RISK_KEYWORDS = {
        'conservative': ['safe', 'guaranteed', 'protect', 'preserve'],
        'moderate': ['balance', 'diversify', 'steady', 'moderate'],
        'aggressive': ['growth', 'high return', 'speculative', 'volatile']
    }

    SENTIMENT_WEIGHTS = {
        'POS': 0.7,
        'NEU': 0.5,
        'NEG': 0.3
    }

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.keyword_weights = self._calculate_keyword_weights()

    def assess_risk(self, text: str) -> RiskProfile:
        """Main risk assessment method"""
        doc = self.nlp(text)
        sentiment = self._analyze_sentiment(doc)
        keyword_scores = self._score_keywords(doc)
        risk_tolerance = self._calculate_risk_tolerance(sentiment, keyword_scores)
        
        return RiskProfile(
            risk_tolerance=risk_tolerance,
            risk_capacity=self._calculate_capacity(risk_tolerance),
            risk_labels=self._categorize_risk(risk_tolerance),
            keywords=keyword_scores,
            sentiment=sentiment
        )

    def _analyze_sentiment(self, doc) -> float:
        """Enhanced sentiment analysis with financial context"""
        sentiment_score = 0
        for sent in doc.sents:
            sentiment_score += sent.sentiment * self.SENTIMENT_WEIGHTS[sent.sentiment]
        return sentiment_score / len(list(doc.sents))

    def _score_keywords(self, doc) -> Dict[str, int]:
        """Score risk-related keywords with context analysis"""
        scores = {category: 0 for category in self.RISK_KEYWORDS}
        for token in doc:
            for category, keywords in self.RISK_KEYWORDS.items():
                if token.lemma_.lower() in keywords:
                    scores[category] += self._get_context_score(token)
        return scores

    def _get_context_score(self, token) -> float:
        """Calculate context-aware keyword score"""
        score = 1.0
        # Check for negations
        if any(child.dep_ == "neg" for child in token.children):
            score *= -0.5
        # Check intensifiers
        if any(child.dep_ == "advmod" and child.text.lower() in ['very', 'extremely'] 
               for child in token.children):
            score *= 1.5
        return score

    def _calculate_risk_tolerance(self, sentiment: float, 
                                keyword_scores: Dict) -> float:
        """Calculate composite risk tolerance score"""
        base_score = np.tanh(sum(
            self.keyword_weights[cat] * count 
            for cat, count in keyword_scores.items()
        ))
        return 0.4 * base_score + 0.6 * sentiment

    def _categorize_risk(self, score: float) -> Dict[str, float]:
        """Categorize risk into different levels"""
        return {
            'conservative': max(0, 1 - score*2),
            'moderate': 1 - abs(score - 0.5)*2,
            'aggressive': max(0, score*2 - 1)
        }

    # 15+ additional helper methods for capacity calculation, validation, etc.
