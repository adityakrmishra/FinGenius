"""
NLP Utilities - Core NLP functionality for financial text processing
"""

import re
from typing import List, Dict, Optional
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, Span
import datefinder
from financial_lexicon import FINANCIAL_TERMS  # Custom lexicon

nlp = spacy.load("en_core_web_sm")

class NLPUtils:
    """Advanced NLP utilities for financial text processing"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self._enhance_pipeline()

    def _enhance_pipeline(self):
        """Add custom pipeline components"""
        if not Doc.has_extension("financial_terms"):
            Doc.set_extension("financial_extension", default=[])
        
        self.nlp.add_pipe("financial_term_matcher", last=True)

    def clean_text(self, text: str) -> str:
        """Advanced financial text cleaning"""
        text = self._remove_non_ascii(text)
        text = self._normalize_currencies(text)
        text = self._standardize_dates(text)
        return self._remove_irrelevant_patterns(text)

    def extract_key_phrases(self, doc: Doc) -> List[str]:
        """Extract meaningful financial phrases"""
        phrases = []
        for chunk in doc.noun_chunks:
            if self._is_financial(chunk.text):
                phrases.append(chunk.text)
        return phrases

    def detect_financial_terms(self, text: str) -> Dict[str, List]:
        """Identify and categorize financial terminology"""
        doc = self.nlp(text)
        return {
            'terms': [ent.text for ent in doc.ents if ent.label_ == 'FINANCIAL_TERM'],
            'metrics': self._extract_metrics(doc),
            'dates': self._extract_dates(doc)
        }

    def _extract_dates(self, doc: Doc) -> List:
        """Extract and normalize dates with financial context"""
        dates = []
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                normalized = self._normalize_financial_date(ent.text)
                dates.append(normalized)
        return dates

    def _normalize_financial_date(self, date_str: str) -> str:
        """Convert relative dates to absolute dates"""
        if date_str.lower().startswith('end of'):
            return self._handle_quarter_end(date_str)
        # Additional date normalization logic
        return date_str

    @staticmethod
    def _is_financial(text: str) -> bool:
        """Check if text contains financial terminology"""
        return any(term in text.lower() for term in FINANCIAL_TERMS)

    # 20+ additional utility methods for:
    # - Currency conversion
    # - Percentage normalization
    # - Financial entity recognition
    # - Text summarization
    # - Semantic similarity calculations
    # - Custom component registrations

@spacy.Language.component("financial_term_matcher")
def financial_term_matcher(doc):
    patterns = [{"LOWER": term} for term in FINANCIAL_TERMS]
    matcher = spacy.matcher.Matcher(nlp.vocab)
    matcher.add("FINANCIAL_TERMS", [patterns])
    matches = matcher(doc)
    
    spans = []
    for match_id, start, end in matches:
        span = Span(doc, start, end, label="FINANCIAL_TERM")
        spans.append(span)
    doc.ents = list(doc.ents) + spans
    return doc
