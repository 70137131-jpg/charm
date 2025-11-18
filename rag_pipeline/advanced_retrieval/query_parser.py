"""Query parsing and expansion"""
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class ParsedQuery:
    """Represents a parsed query"""
    original: str
    expanded: List[str]
    keywords: List[str]
    entities: List[str]
    intent: Optional[str] = None


class QueryParser:
    """Parse and expand user queries"""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize query parser

        Args:
            spacy_model: Spacy model for NLP
        """
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"Spacy model {spacy_model} not found. Some features will be limited.")

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a query and extract information

        Args:
            query: User query

        Returns:
            ParsedQuery object
        """
        # Basic parsing without spacy
        keywords = self.extract_keywords_simple(query)
        entities = []
        expanded = [query]

        # Advanced parsing with spacy if available
        if self.nlp:
            doc = self.nlp(query)
            entities = [ent.text for ent in doc.ents]
            keywords = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "VERB", "ADJ"]
            ]

        # Query expansion
        expanded.extend(self.expand_query(query, keywords))

        return ParsedQuery(
            original=query,
            expanded=expanded,
            keywords=keywords,
            entities=entities
        )

    def extract_keywords_simple(self, query: str) -> List[str]:
        """Extract keywords using simple heuristics"""
        # Remove punctuation and convert to lowercase
        clean_query = re.sub(r'[^\w\s]', '', query.lower())

        # Simple stopword list
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how'
        }

        # Extract keywords
        keywords = [
            word for word in clean_query.split()
            if word not in stopwords and len(word) > 2
        ]

        return keywords

    def expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """
        Expand query with variations

        Args:
            query: Original query
            keywords: Extracted keywords

        Returns:
            List of expanded queries
        """
        expansions = []

        # Add query with just keywords
        if keywords:
            keyword_query = ' '.join(keywords)
            if keyword_query != query:
                expansions.append(keyword_query)

        # Add synonyms (simplified - in production use WordNet or word embeddings)
        synonym_map = {
            'find': ['search', 'locate', 'discover'],
            'show': ['display', 'present', 'demonstrate'],
            'explain': ['describe', 'clarify', 'elaborate'],
            'best': ['top', 'optimal', 'excellent'],
            'how': ['what is the method', 'what is the way'],
            'why': ['what is the reason', 'what causes'],
        }

        # Create synonym expansions
        query_lower = query.lower()
        for word, synonyms in synonym_map.items():
            if word in query_lower:
                for syn in synonyms:
                    expanded = query_lower.replace(word, syn)
                    if expanded != query_lower:
                        expansions.append(expanded)

        return expansions[:3]  # Limit expansions

    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract metadata filters from query

        Examples:
            "papers from 2023" -> {"year": 2023}
            "articles by John Doe" -> {"author": "John Doe"}

        Args:
            query: User query

        Returns:
            Dictionary of filters
        """
        filters = {}

        # Year extraction
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            filters['year'] = int(year_match.group())

        # Date range extraction
        date_range = re.search(r'from (\d{4}) to (\d{4})', query)
        if date_range:
            filters['year'] = {
                'gte': int(date_range.group(1)),
                'lte': int(date_range.group(2))
            }

        # Author extraction
        author_match = re.search(r'by ([A-Z][a-z]+ [A-Z][a-z]+)', query)
        if author_match:
            filters['author'] = author_match.group(1)

        # Source extraction
        source_match = re.search(r'from ([a-zA-Z]+)', query)
        if source_match and source_match.group(1).lower() not in ['the', 'a', 'an']:
            filters['source'] = source_match.group(1)

        return filters

    def detect_intent(self, query: str) -> str:
        """
        Detect query intent

        Returns:
            Intent string: 'search', 'question', 'comparison', 'definition', etc.
        """
        query_lower = query.lower().strip()

        # Question patterns
        if query_lower.startswith(('what', 'who', 'when', 'where', 'why', 'how')):
            if 'compare' in query_lower or 'difference' in query_lower:
                return 'comparison'
            elif 'define' in query_lower or query_lower.startswith('what is'):
                return 'definition'
            else:
                return 'question'

        # Search patterns
        if any(word in query_lower for word in ['find', 'search', 'look for', 'show me']):
            return 'search'

        # Comparison
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference between']):
            return 'comparison'

        # Definition
        if any(word in query_lower for word in ['define', 'meaning of', 'what is']):
            return 'definition'

        # Default
        return 'search'

    def reformulate_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Reformulate query for better retrieval

        Args:
            query: Original query
            context: Optional conversation context

        Returns:
            Reformulated query
        """
        # Remove filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
        reformulated = query

        for filler in filler_words:
            reformulated = re.sub(rf'\b{filler}\b', '', reformulated, flags=re.IGNORECASE)

        # Clean up whitespace
        reformulated = ' '.join(reformulated.split())

        # If context is provided, incorporate it
        if context:
            # Simple context incorporation - in production, use more sophisticated methods
            parsed = self.parse(reformulated)
            if not parsed.entities:
                # Try to extract entities from context
                if self.nlp:
                    context_doc = self.nlp(context)
                    context_entities = [ent.text for ent in context_doc.ents]
                    if context_entities:
                        reformulated = f"{reformulated} {context_entities[0]}"

        return reformulated.strip()


class MultiQueryGenerator:
    """Generate multiple query variations for better retrieval"""

    @staticmethod
    def generate_variations(query: str, num_variations: int = 3) -> List[str]:
        """
        Generate query variations

        Args:
            query: Original query
            num_variations: Number of variations to generate

        Returns:
            List of query variations
        """
        variations = [query]

        # Variation 1: Rephrase as a question
        if not query.strip().endswith('?'):
            question_words = ['What', 'How', 'Why', 'When', 'Where', 'Who']
            # Simple heuristic to convert to question
            if any(query.lower().startswith(w.lower()) for w in question_words):
                variations.append(query)
            else:
                variations.append(f"What is {query}?")

        # Variation 2: Make more specific
        variations.append(f"{query} explanation with examples")

        # Variation 3: Make more general
        parser = QueryParser()
        keywords = parser.extract_keywords_simple(query)
        if len(keywords) > 2:
            variations.append(' '.join(keywords[:2]))

        # Return requested number of unique variations
        unique_variations = []
        seen = set()

        for var in variations:
            if var not in seen:
                unique_variations.append(var)
                seen.add(var)
                if len(unique_variations) >= num_variations:
                    break

        return unique_variations
