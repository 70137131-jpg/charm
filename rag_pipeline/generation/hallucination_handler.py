"""Hallucination detection and handling"""
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass


@dataclass
class HallucinationCheck:
    """Result of hallucination check"""
    is_hallucination: bool
    confidence: float
    reasons: List[str]
    evidence: Optional[Dict[str, Any]] = None


class HallucinationHandler:
    """Detect and handle hallucinations in LLM outputs"""

    def __init__(self, llm=None):
        """
        Initialize hallucination handler

        Args:
            llm: Optional LLM instance for self-verification
        """
        self.llm = llm

    def check_factual_consistency(
        self,
        answer: str,
        context: str,
        threshold: float = 0.7
    ) -> HallucinationCheck:
        """
        Check if answer is factually consistent with context

        Args:
            answer: Generated answer
            context: Source context
            threshold: Confidence threshold

        Returns:
            HallucinationCheck result
        """
        reasons = []
        is_hallucination = False
        confidence = 1.0

        # Extract claims from answer
        claims = self._extract_claims(answer)

        # Check each claim against context
        unsupported_claims = []
        for claim in claims:
            if not self._is_supported_by_context(claim, context):
                unsupported_claims.append(claim)
                is_hallucination = True
                reasons.append(f"Unsupported claim: {claim}")

        # Calculate confidence
        if claims:
            support_ratio = 1 - (len(unsupported_claims) / len(claims))
            confidence = support_ratio

        if confidence < threshold:
            is_hallucination = True
            reasons.append(f"Support ratio {confidence:.2f} below threshold {threshold}")

        return HallucinationCheck(
            is_hallucination=is_hallucination,
            confidence=confidence,
            reasons=reasons,
            evidence={
                "total_claims": len(claims),
                "unsupported_claims": len(unsupported_claims),
                "unsupported": unsupported_claims
            }
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Simple heuristic: split by sentences
        sentences = re.split(r'[.!?]+', text)
        claims = [s.strip() for s in sentences if s.strip()]

        # Filter out very short or question sentences
        claims = [
            c for c in claims
            if len(c.split()) >= 3 and not c.endswith('?')
        ]

        return claims

    def _is_supported_by_context(self, claim: str, context: str) -> bool:
        """
        Check if a claim is supported by context

        Simple heuristic: check for keyword overlap
        For production, use NLI models
        """
        # Normalize
        claim_lower = claim.lower()
        context_lower = context.lower()

        # Extract keywords from claim
        claim_words = set(re.findall(r'\b\w+\b', claim_lower))

        # Remove common words
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that'
        }
        claim_words = claim_words - stopwords

        # Check if significant portion of keywords appear in context
        if not claim_words:
            return True

        matching_words = sum(1 for word in claim_words if word in context_lower)
        overlap_ratio = matching_words / len(claim_words)

        # Require at least 50% overlap
        return overlap_ratio >= 0.5

    def self_check(
        self,
        answer: str,
        context: str,
        question: str
    ) -> HallucinationCheck:
        """
        Use the LLM itself to check for hallucinations

        Args:
            answer: Generated answer
            context: Source context
            question: Original question

        Returns:
            HallucinationCheck result
        """
        if not self.llm:
            raise ValueError("LLM not provided for self-check")

        verification_prompt = f"""You are a fact-checker. Verify if the answer is supported by the context.

Context:
{context}

Question: {question}

Answer: {answer}

Is this answer fully supported by the context? Check for:
1. Factual accuracy
2. Claims not in the context
3. Misinterpretations
4. Exaggerations

Respond in this format:
VERDICT: [SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED]
CONFIDENCE: [0.0-1.0]
ISSUES: [List any issues found]

Verification:"""

        response = self.llm.generate(verification_prompt, temperature=0.0)
        result_text = response.text

        # Parse response
        verdict_match = re.search(r'VERDICT:\s*(\w+)', result_text, re.IGNORECASE)
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', result_text, re.IGNORECASE)
        issues_match = re.search(r'ISSUES:\s*(.+?)(?=\n\n|\Z)', result_text, re.IGNORECASE | re.DOTALL)

        verdict = verdict_match.group(1).upper() if verdict_match else "UNKNOWN"
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        issues = [issues_match.group(1).strip()] if issues_match else []

        is_hallucination = verdict in ["PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]

        return HallucinationCheck(
            is_hallucination=is_hallucination,
            confidence=confidence,
            reasons=issues,
            evidence={"verdict": verdict}
        )

    def check_for_hedging(self, answer: str) -> bool:
        """
        Check if answer contains appropriate hedging/uncertainty markers

        Returns True if answer appropriately expresses uncertainty
        """
        hedging_phrases = [
            "i don't know",
            "not sure",
            "unclear",
            "cannot determine",
            "insufficient information",
            "not mentioned",
            "doesn't say",
            "appears to",
            "seems to",
            "might",
            "may",
            "possibly",
            "perhaps",
            "according to",
        ]

        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in hedging_phrases)

    def detect_fabricated_details(
        self,
        answer: str,
        context: str
    ) -> List[str]:
        """
        Detect specific details in answer that aren't in context

        Returns:
            List of potentially fabricated details
        """
        fabricated = []

        # Extract numbers, dates, names (simple heuristics)
        import re

        # Numbers
        answer_numbers = set(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', answer))
        context_numbers = set(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', context))
        fabricated_numbers = answer_numbers - context_numbers

        if fabricated_numbers:
            fabricated.append(f"Numbers not in context: {', '.join(fabricated_numbers)}")

        # Dates
        answer_dates = set(re.findall(r'\b\d{4}\b', answer))
        context_dates = set(re.findall(r'\b\d{4}\b', context))
        fabricated_dates = answer_dates - context_dates

        if fabricated_dates:
            fabricated.append(f"Dates not in context: {', '.join(fabricated_dates)}")

        # Capitalized words (potential proper nouns)
        answer_proper = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer))
        context_proper = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context))
        fabricated_proper = answer_proper - context_proper

        # Filter out common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'However', 'Therefore', 'Additionally'}
        fabricated_proper = fabricated_proper - common_words

        if fabricated_proper:
            fabricated.append(f"Proper nouns not in context: {', '.join(list(fabricated_proper)[:5])}")

        return fabricated

    def handle_hallucination(
        self,
        answer: str,
        context: str,
        question: str,
        strategy: str = "filter"
    ) -> str:
        """
        Handle detected hallucinations

        Args:
            answer: Generated answer
            context: Source context
            question: Original question
            strategy: Handling strategy ('filter', 'hedge', 'regenerate')

        Returns:
            Corrected or filtered answer
        """
        check = self.check_factual_consistency(answer, context)

        if not check.is_hallucination:
            return answer

        if strategy == "filter":
            # Return a safe response
            return "I cannot provide a reliable answer based on the available context. The information needed to answer this question is not sufficiently present in the provided sources."

        elif strategy == "hedge":
            # Add uncertainty markers
            hedged = "Based on the available context, " + answer
            if check.evidence and check.evidence.get('unsupported_claims'):
                hedged += "\n\nNote: Some claims may not be fully supported by the provided context."
            return hedged

        elif strategy == "regenerate":
            # Suggest regeneration (requires LLM)
            if self.llm:
                prompt = f"""The previous answer contained unsupported claims. Generate a new answer that only uses information explicitly stated in the context.

Context:
{context}

Question: {question}

Previous answer (contained errors):
{answer}

Issues found:
{', '.join(check.reasons)}

Generate a new, factually accurate answer using ONLY information from the context:"""

                response = self.llm.generate(prompt, temperature=0.3)
                return response.text
            else:
                return answer

        return answer

    def attribution_check(
        self,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Check if answer can be attributed to specific parts of context

        Returns:
            Dictionary with attribution information
        """
        claims = self._extract_claims(answer)
        attributions = {}

        for claim in claims:
            # Find sentences in context that support this claim
            context_sentences = re.split(r'[.!?]+', context)
            supporting_sentences = []

            for sent in context_sentences:
                if self._is_supported_by_context(claim, sent):
                    supporting_sentences.append(sent.strip())

            attributions[claim] = {
                "supported": len(supporting_sentences) > 0,
                "sources": supporting_sentences
            }

        return attributions
