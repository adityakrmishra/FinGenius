"""
Financial Goal Parser - Extract and classify financial objectives from natural language
"""

import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dateutil.parser import parse
import spacy
from pydantic import BaseModel, ValidationError

nlp = spacy.load("en_core_web_sm")

class FinancialGoal(BaseModel):
    goal_type: str
    amount: Optional[float]
    target_date: Optional[datetime]
    priority: int = 3
    confidence: float = Field(..., ge=0, le=1)
    raw_text: str

class GoalParser:
    """Advanced NLP parser for financial goals extraction"""
    
    GOAL_PATTERNS = {
        'savings': r'(save|accumulate|set aside|reserve)\s+\$?(\d+[\d,\.]*)\s+(\bby\b|\bwithin\b|\bin\b)?',
        'retirement': r'(retire|retirement)\s+(?:by|at|before)\s+(\d+)\s+years?\s+old',
        'debt': r'(pay off|clear|eliminate)\s+(?:my\s+)?(debt|loan|mortgage|credit card)',
        'investment': r'(invest|allocate)\s+\$?(\d+[\d,\.]*)\s+in\s+(\w+)',
        'education': r'(education|college|university)\s+fund\s+(?:of|with)\s+\$?(\d+[\d,\.]*)',
        'emergency_fund': r'emergency fund\s+(?:of|with)\s+\$?(\d+[\d,\.]*)'
    }

    GOAL_TYPES = {
        'savings': ['save', 'accumulate', 'set aside'],
        'retirement': ['retire', 'retirement'],
        'debt': ['pay off', 'clear debt'],
        'investment': ['invest', 'allocate'],
        'education': ['education fund'],
        'emergency': ['emergency fund']
    }

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.date_parser = DateParser()

    def parse_goals(self, text: str) -> List[FinancialGoal]:
        """Main parsing method with validation"""
        goals = []
        clean_text = self.preprocess_text(text)
        
        # Pattern-based extraction
        pattern_matches = self._extract_with_patterns(clean_text)
        goals.extend(pattern_matches)
        
        # NLP-based extraction
        nlp_matches = self._extract_with_nlp(clean_text)
        goals.extend(nlp_matches)
        
        # Remove duplicates and validate
        return self._deduplicate_and_validate(goals)

    def _extract_with_patterns(self, text: str) -> List[FinancialGoal]:
        """Regex-based goal extraction"""
        matches = []
        for goal_type, pattern in self.GOAL_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                amount = self._parse_amount(match.group(2)) if match.groups() >= 2 else None
                date = self.date_parser.parse_date(match.group(3)) if match.groups() >= 3 else None
                
                matches.append(FinancialGoal(
                    goal_type=goal_type,
                    amount=amount,
                    target_date=date,
                    confidence=0.85,
                    raw_text=match.group(0)
                ))
        return matches

    def _extract_with_nlp(self, text: str) -> List[FinancialGoal]:
        """SpaCy-based goal extraction"""
        goals = []
        doc = self.nlp(text)
        
        # Check for monetary entities and dates
        money_ents = [ent for ent in doc.ents if ent.label_ == 'MONEY']
        date_ents = [ent for ent in doc.ents if ent.label_ == 'DATE']
        
        for sent in doc.sents:
            goal_type = self._classify_goal_type(sent.text)
            if goal_type:
                amount = self._find_closest_amount(sent, money_ents)
                date = self._find_closest_date(sent, date_ents)
                
                goals.append(FinancialGoal(
                    goal_type=goal_type,
                    amount=amount,
                    target_date=date,
                    confidence=self._calculate_confidence(sent),
                    raw_text=sent.text
                ))
        return goals

    def _classify_goal_type(self, text: str) -> Optional[str]:
        """Classify goal type using semantic analysis"""
        doc = self.nlp(text)
        verb_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        for goal_type, keywords in self.GOAL_TYPES.items():
            if any(keyword in verb_phrases for keyword in keywords):
                return goal_type
        return None

    def _deduplicate_and_validate(self, goals: List[FinancialGoal]) -> List[FinancialGoal]:
        """Validate and deduplicate goals"""
        unique_goals = {}
        for goal in goals:
            try:
                key = f"{goal.goal_type}-{goal.amount}-{goal.target_date}"
                if key not in unique_goals or unique_goals[key].confidence < goal.confidence:
                    unique_goals[key] = goal
            except ValidationError as e:
                logging.error(f"Invalid goal detected: {str(e)}")
        return list(unique_goals.values())

    # Helper methods for amount/date extraction and confidence calculation
    # ... (20+ additional helper methods)

class DateParser:
    """Advanced financial date parser with relative date handling"""
    def parse_date(self, date_str: str) -> Optional[datetime]:
        try:
            if re.match(r'in\s+\d+', date_str):
                return self._parse_relative_date(date_str)
            return parse(date_str, fuzzy=True)
        except:
            return None

    def _parse_relative_date(self, text: str) -> Optional[datetime]:
        match = re.search(r'in\s+(\d+)\s+(days?|weeks?|months?|years?)', text)
        if match:
            quantity = int(match.group(1))
            unit = match.group(2)
            delta = self._get_timedelta(quantity, unit)
            return datetime.now() + delta
        return None

    # Additional date parsing utilities...
