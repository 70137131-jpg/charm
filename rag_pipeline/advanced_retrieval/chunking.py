"""Document chunking strategies"""
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class Chunk:
    """Represents a document chunk"""
    text: str
    metadata: Dict[str, Any]
    start_char: int
    end_char: int
    chunk_id: int


class DocumentChunker:
    """Basic document chunking strategies"""

    @staticmethod
    def chunk_by_tokens(
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = " "
    ) -> List[str]:
        """
        Chunk text by token count with overlap

        Args:
            text: Input text
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            separator: Token separator

        Returns:
            List of text chunks
        """
        tokens = text.split(separator)
        chunks = []

        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk = separator.join(tokens[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

            # Stop if we've reached the end
            if i + chunk_size >= len(tokens):
                break

        return chunks

    @staticmethod
    def chunk_by_characters(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        Chunk text by character count with overlap

        Args:
            text: Input text
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters

        Returns:
            List of text chunks
        """
        chunks = []

        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

            if i + chunk_size >= len(text):
                break

        return chunks

    @staticmethod
    def chunk_by_sentences(
        text: str,
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1
    ) -> List[str]:
        """
        Chunk text by sentences

        Args:
            text: Input text
            sentences_per_chunk: Number of sentences per chunk
            overlap_sentences: Number of overlapping sentences

        Returns:
            List of text chunks
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []

        for i in range(0, len(sentences), sentences_per_chunk - overlap_sentences):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk)

            if i + sentences_per_chunk >= len(sentences):
                break

        return chunks

    @staticmethod
    def chunk_by_paragraphs(
        text: str,
        max_chunk_size: Optional[int] = None
    ) -> List[str]:
        """
        Chunk text by paragraphs

        Args:
            text: Input text
            max_chunk_size: Maximum chunk size in characters (None for no limit)

        Returns:
            List of text chunks
        """
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            if max_chunk_size and current_size + para_size > max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks


class AdvancedChunker:
    """Advanced chunking techniques using semantic and structural information"""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize advanced chunker

        Args:
            spacy_model: Spacy model to use for NLP
        """
        if not SPACY_AVAILABLE:
            raise ImportError("Spacy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            import subprocess
            print(f"Downloading spacy model: {spacy_model}")
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)

    def chunk_by_semantic_similarity(
        self,
        text: str,
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500
    ) -> List[Chunk]:
        """
        Chunk text based on semantic similarity between sentences

        Args:
            text: Input text
            similarity_threshold: Minimum similarity to keep sentences together
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters

        Returns:
            List of Chunk objects
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return []

        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0].text)
        chunk_id = 0

        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_size = len(sent.text)

            # Check semantic similarity with previous sentence
            similarity = current_chunk[-1].similarity(sent)

            # Decide whether to add to current chunk or start new one
            should_split = (
                similarity < similarity_threshold or
                current_size + sent_size > max_chunk_size
            )

            if should_split and current_size >= min_chunk_size:
                # Save current chunk
                chunk_text = ' '.join(s.text for s in current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={"chunk_type": "semantic"},
                    start_char=current_chunk[0].start_char,
                    end_char=current_chunk[-1].end_char,
                    chunk_id=chunk_id
                ))
                chunk_id += 1

                # Start new chunk
                current_chunk = [sent]
                current_size = sent_size
            else:
                # Add to current chunk
                current_chunk.append(sent)
                current_size += sent_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(s.text for s in current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={"chunk_type": "semantic"},
                start_char=current_chunk[0].start_char,
                end_char=current_chunk[-1].end_char,
                chunk_id=chunk_id
            ))

        return chunks

    def chunk_by_topic(
        self,
        text: str,
        max_chunk_size: int = 500
    ) -> List[Chunk]:
        """
        Chunk text by detecting topic changes using noun phrases and entities

        Args:
            text: Input text
            max_chunk_size: Maximum chunk size in characters

        Returns:
            List of Chunk objects
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return []

        chunks = []
        current_chunk = [sentences[0]]
        current_entities = set(ent.text.lower() for ent in sentences[0].ents)
        current_nouns = set(token.lemma_.lower() for token in sentences[0] if token.pos_ == "NOUN")
        current_size = len(sentences[0].text)
        chunk_id = 0

        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_size = len(sent.text)

            # Extract entities and nouns from current sentence
            sent_entities = set(ent.text.lower() for ent in sent.ents)
            sent_nouns = set(token.lemma_.lower() for token in sent if token.pos_ == "NOUN")

            # Calculate topic overlap
            entity_overlap = len(current_entities & sent_entities) / max(len(current_entities | sent_entities), 1)
            noun_overlap = len(current_nouns & sent_nouns) / max(len(current_nouns | sent_nouns), 1)
            topic_similarity = (entity_overlap + noun_overlap) / 2

            # Decide whether to add to current chunk or start new one
            should_split = (
                topic_similarity < 0.3 or
                current_size + sent_size > max_chunk_size
            )

            if should_split and len(current_chunk) > 0:
                # Save current chunk
                chunk_text = ' '.join(s.text for s in current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={"chunk_type": "topic"},
                    start_char=current_chunk[0].start_char,
                    end_char=current_chunk[-1].end_char,
                    chunk_id=chunk_id
                ))
                chunk_id += 1

                # Start new chunk
                current_chunk = [sent]
                current_entities = sent_entities
                current_nouns = sent_nouns
                current_size = sent_size
            else:
                # Add to current chunk
                current_chunk.append(sent)
                current_entities |= sent_entities
                current_nouns |= sent_nouns
                current_size += sent_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(s.text for s in current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={"chunk_type": "topic"},
                start_char=current_chunk[0].start_char,
                end_char=current_chunk[-1].end_char,
                chunk_id=chunk_id
            ))

        return chunks

    def recursive_character_split(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ) -> List[str]:
        """
        Recursively split text using multiple separators

        Tries to split on paragraphs first, then sentences, then words

        Args:
            text: Input text
            chunk_size: Target chunk size
            chunk_overlap: Overlap size
            separators: List of separators to try (in order)

        Returns:
            List of text chunks
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        def split_text(text: str, sep_idx: int) -> List[str]:
            if sep_idx >= len(separators):
                # No more separators, just split by character
                return DocumentChunker.chunk_by_characters(text, chunk_size, chunk_overlap)

            separator = separators[sep_idx]

            if not separator:
                # Empty separator means split by character
                return DocumentChunker.chunk_by_characters(text, chunk_size, chunk_overlap)

            # Split by current separator
            splits = text.split(separator)

            chunks = []
            current_chunk = []
            current_size = 0

            for split in splits:
                split_size = len(split)

                if split_size > chunk_size:
                    # This split is too large, recursively split it
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                        current_chunk = []
                        current_size = 0

                    # Recursively split the large piece
                    sub_chunks = split_text(split, sep_idx + 1)
                    chunks.extend(sub_chunks)
                elif current_size + split_size > chunk_size:
                    # Adding this would exceed chunk size
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                    current_chunk = [split]
                    current_size = split_size
                else:
                    # Add to current chunk
                    current_chunk.append(split)
                    current_size += split_size + len(separator)

            # Add remaining chunk
            if current_chunk:
                chunks.append(separator.join(current_chunk))

            return chunks

        return split_text(text, 0)


class MarkdownChunker:
    """Specialized chunker for Markdown documents"""

    @staticmethod
    def chunk_by_headers(text: str, max_chunk_size: Optional[int] = None) -> List[Chunk]:
        """
        Chunk markdown by headers, preserving document structure

        Args:
            text: Markdown text
            max_chunk_size: Maximum chunk size in characters

        Returns:
            List of Chunk objects
        """
        # Split by headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')

        chunks = []
        current_chunk = []
        current_headers = []
        current_size = 0
        chunk_id = 0
        start_char = 0

        for line in lines:
            header_match = re.match(header_pattern, line)

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2)

                # Check if we should start a new chunk
                if max_chunk_size and current_size > max_chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={
                            "headers": current_headers.copy(),
                            "chunk_type": "markdown"
                        },
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                    start_char += len(chunk_text) + 1

                    current_chunk = []
                    current_size = 0

                # Update header context
                current_headers = current_headers[:level-1] + [title]

            current_chunk.append(line)
            current_size += len(line) + 1

        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "headers": current_headers.copy(),
                    "chunk_type": "markdown"
                },
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                chunk_id=chunk_id
            ))

        return chunks
