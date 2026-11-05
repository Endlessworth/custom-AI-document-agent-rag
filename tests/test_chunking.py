"""Unit tests for the chunking module (src/chunking.py).

Verifies splitting logic, boundary preservation, overlap retention, parameter
validation, and independent section processing without any external dependencies.
"""

import unittest
from src.loaders import DocumentSection
from src.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DocumentChunk,
    validate_chunk_params,
    split_text,
    chunk_section,
    chunk_sections,
    make_chunk_id,
)


class TestChunkingModule(unittest.TestCase):
    """Test suite covering text chunking and section processing."""

    # ------------------------------------------------------------------------
    # 1. Parameter Validation Tests
    # ------------------------------------------------------------------------

    def test_parameter_validation_success(self):
        """Valid parameters should pass without error."""
        try:
            validate_chunk_params(1000, 150)
            validate_chunk_params(500, 0)
            validate_chunk_params(100, 99)
        except (ValueError, TypeError) as exc:
            self.fail(f"validate_chunk_params unexpectedly raised: {exc}")

    def test_parameter_validation_invalid_chunk_size(self):
        """Zero or negative chunk_size should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_chunk_params(0, 50)
        self.assertIn("chunk_size must be a positive integer", str(ctx.exception))

        with self.assertRaises(ValueError):
            validate_chunk_params(-100, 50)

    def test_parameter_validation_negative_overlap(self):
        """Negative chunk_overlap should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_chunk_params(500, -10)
        self.assertIn("chunk_overlap must be a non-negative integer", str(ctx.exception))

    def test_parameter_validation_overlap_greater_or_equal_to_size(self):
        """Overlap >= size should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_chunk_params(500, 500)
        self.assertIn("must be strictly less than chunk_size", str(ctx.exception))

        with self.assertRaises(ValueError):
            validate_chunk_params(500, 600)

    def test_parameter_validation_types(self):
        """Non-integer parameters should raise TypeError."""
        with self.assertRaises(TypeError):
            validate_chunk_params("1000", 150)  # type: ignore

        with self.assertRaises(TypeError):
            validate_chunk_params(1000, 15.5)  # type: ignore

    # ------------------------------------------------------------------------
    # 2. Text Splitting & Boundary Tests
    # ------------------------------------------------------------------------

    def test_short_text_remains_single_chunk(self):
        """Short text (<= chunk_size) must remain a single, intact chunk."""
        text = "This is a brief policy document that fits easily inside one chunk."
        chunks = split_text(text, chunk_size=200, chunk_overlap=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_empty_and_whitespace_text_handled_safely(self):
        """Empty or whitespace strings should return an empty list of chunks."""
        self.assertEqual(split_text(""), [])
        self.assertEqual(split_text("   \n\t  "), [])
        self.assertEqual(split_text(None), [])  # type: ignore

    def test_long_text_split_into_multiple_chunks(self):
        """Long text must be split into multiple chunks, each <= chunk_size."""
        p1 = "Paragraph 1 explains the first set of operational guidelines in detail."
        p2 = "Paragraph 2 details the deployment criteria and server security configuration."
        p3 = "Paragraph 3 concludes the report with next steps and future roadmap items."
        doc = f"{p1}\n\n{p2}\n\n{p3}"

        chunk_size = 80
        chunk_overlap = 20
        chunks = split_text(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreater(len(chunks), 1)
        for idx, chunk in enumerate(chunks):
            self.assertLessEqual(
                len(chunk),
                chunk_size,
                f"Chunk {idx} exceeds chunk_size ({len(chunk)} > {chunk_size})"
            )

    def test_paragraph_and_sentence_boundaries_preferred(self):
        """Splits should prefer paragraph boundaries when available."""
        p1 = "Sentence A in first paragraph. Sentence B in first paragraph."
        p2 = "Sentence C in second paragraph. Sentence D in second paragraph."
        text = f"{p1}\n\n{p2}"

        # Size large enough for p1 + boundary, but not both p1 and p2
        chunk_size = len(p1) + 20
        chunk_overlap = 15
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreaterEqual(len(chunks), 2)
        # Chunk 0 should end cleanly with paragraph 1 rather than cutting into paragraph 2
        self.assertEqual(chunks[0], p1)

    def test_sentence_boundary_splitting(self):
        """Splits should prefer sentence ends (. ) over arbitrary cuts."""
        s1 = "The first sentence establishes the context."
        s2 = "The second sentence introduces the core technical premise."
        s3 = "The third sentence offers supporting details."
        text = f"{s1} {s2} {s3}"

        chunk_size = len(s1) + len(s2) + 5
        chunk_overlap = 10
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreaterEqual(len(chunks), 2)
        # Chunk 0 should end with a period
        self.assertTrue(chunks[0].endswith("."))

    def test_overlap_preserved_between_consecutive_chunks(self):
        """Consecutive chunks must share overlapping words/context."""
        p1 = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."
        p2 = "Mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
        text = f"{p1} {p2}"

        chunk_size = 70
        chunk_overlap = 25
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreaterEqual(len(chunks), 2)
        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Find common substring between tail of curr_chunk and head of next_chunk
            # The last few words of curr_chunk should appear near the start of next_chunk
            curr_words = curr_chunk.split()
            last_word = curr_words[-1]
            self.assertIn(
                last_word,
                next_chunk,
                f"Overlap missing: last word '{last_word}' of chunk {i} not in chunk {i+1}"
            )

    def test_unbreakable_string_fallback(self):
        """A string with no spaces or punctuation must be split at chunk_size without error."""
        long_hash = "abcdefghijklmnopqrstuvwxyz0123456789" * 4  # 144 chars
        chunk_size = 50
        chunk_overlap = 10
        chunks = split_text(long_hash, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreaterEqual(len(chunks), 3)
        for c in chunks:
            self.assertLessEqual(len(c), chunk_size)
        # Overlap check
        self.assertEqual(chunks[0][-10:], chunks[1][:10])

    # ------------------------------------------------------------------------
    # 3. DocumentSection Chunking & Metadata Preservation
    # ------------------------------------------------------------------------

    def test_pdf_section_chunking_metadata_preserved(self):
        """PDF DocumentSection chunking preserves page, total_chunks, and metadata."""
        sec = DocumentSection(
            text="Long PDF text on page 3. " * 30,  # ~780 chars
            source="annual_report.pdf",
            file_type="pdf",
            page=3,
            metadata={"department": "Finance", "confidential": True}
        )

        chunks = chunk_section(sec, chunk_size=300, chunk_overlap=50)
        self.assertGreater(len(chunks), 1)

        for idx, ch in enumerate(chunks):
            self.assertEqual(ch.source, "annual_report.pdf")
            self.assertEqual(ch.file_type, "pdf")
            self.assertEqual(ch.page, 3)
            self.assertEqual(ch.chunk_index, idx)
            self.assertEqual(ch.total_chunks, len(chunks))
            self.assertIn("p3:c", ch.chunk_id)
            self.assertEqual(ch.metadata["department"], "Finance")
            self.assertTrue(ch.metadata["confidential"])
            self.assertIn("chunk_size", ch.metadata)

    def test_excel_sections_remain_separate(self):
        """Excel rows must remain distinct chunks and NEVER be merged together."""
        row1_sec = DocumentSection(
            text="Sheet: Q4 | Row 2 | Dept: R&D | Budget: 1200000 | Contact: Alice",
            source="finances.xlsx",
            file_type="xlsx",
            sheet_name="Q4",
            row_index=2,
            metadata={"sheet": "Q4"}
        )
        row2_sec = DocumentSection(
            text="Sheet: Q4 | Row 3 | Dept: Sales | Budget: 850000 | Contact: Bob",
            source="finances.xlsx",
            file_type="xlsx",
            sheet_name="Q4",
            row_index=3,
            metadata={"sheet": "Q4"}
        )

        chunks = chunk_sections([row1_sec, row2_sec], chunk_size=1000, chunk_overlap=150)

        # Must have exactly 2 chunks, one per row
        self.assertEqual(len(chunks), 2)

        # Chunk 1 verifies Row 2
        self.assertEqual(chunks[0].sheet_name, "Q4")
        self.assertEqual(chunks[0].row_index, 2)
        self.assertIn("Dept: R&D", chunks[0].text)
        self.assertNotIn("Dept: Sales", chunks[0].text)
        self.assertEqual(chunks[0].chunk_id, "finances.xlsx:sheet_Q4:r2:c0")

        # Chunk 2 verifies Row 3
        self.assertEqual(chunks[1].sheet_name, "Q4")
        self.assertEqual(chunks[1].row_index, 3)
        self.assertIn("Dept: Sales", chunks[1].text)
        self.assertNotIn("Dept: R&D", chunks[1].text)
        self.assertEqual(chunks[1].chunk_id, "finances.xlsx:sheet_Q4:r3:c0")

    def test_powerpoint_slide_section_chunking(self):
        """PowerPoint slide sections retain slide_number and slide metadata."""
        sec = DocumentSection(
            text="Slide Title: APAC Strategy\n\nExpansion planned for Q1 2025 across regional hubs.",
            source="strategy.pptx",
            file_type="pptx",
            slide_number=4,
            metadata={"slide_title": "APAC Strategy", "total_slides": 10}
        )

        chunks = chunk_section(sec, chunk_size=500, chunk_overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source, "strategy.pptx")
        self.assertEqual(chunks[0].file_type, "pptx")
        self.assertEqual(chunks[0].slide_number, 4)
        self.assertEqual(chunks[0].chunk_id, "strategy.pptx:s4:c0")
        self.assertEqual(chunks[0].metadata["slide_title"], "APAC Strategy")

    def test_chunk_to_dict_serialization(self):
        """DocumentChunk.to_dict produces clean dictionary with all attributes."""
        chunk = DocumentChunk(
            text="Sample chunk content",
            source="guide.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=1,
            chunk_id="guide.pdf:p1:c0",
            page=1,
            metadata={"test_key": "test_val"}
        )
        d = chunk.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["chunk_id"], "guide.pdf:p1:c0")
        self.assertEqual(d["text"], "Sample chunk content")
        self.assertEqual(d["page"], 1)
        self.assertIsNone(d["sheet_name"])
        self.assertEqual(d["metadata"]["test_key"], "test_val")

    # ------------------------------------------------------------------------
    # 4. Focused Verification: Awkward Patterns, Size Bounds & Overlap
    # ------------------------------------------------------------------------

    def test_awkward_long_sentence_without_punctuation(self):
        """Long sentence exceeding chunk_size with no punctuation splits cleanly at word boundaries."""
        sentence = (
            "this is an exceptionally long continuous English sentence containing no punctuation marks "
            "or comma breaks whatsoever that is designed specifically to test how our text splitting "
            "algorithm handles clauses that far exceed the configured chunk size limit without cutting "
            "words in half or generating malformed fragments"
        )
        chunk_size = 80
        chunk_overlap = 20
        chunks = split_text(sentence, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreater(len(chunks), 2)
        for idx, c in enumerate(chunks):
            self.assertLessEqual(len(c), chunk_size, f"Chunk {idx} exceeded chunk_size")
            # Must not start or end with broken partial whitespace
            self.assertEqual(c, c.strip())

        # Verify all original words are present across the chunks
        orig_words = sentence.split()
        all_chunk_words = set(" ".join(chunks).split())
        for w in orig_words:
            self.assertIn(w, all_chunk_words, f"Word '{w}' was lost in chunking")

    def test_awkward_repeated_spaces_and_tabs(self):
        """Text with repeated spaces, tabs, and mixed whitespace chunks cleanly without text loss."""
        awkward = (
            "Item One    with   lots   of   spaces.   \t\tItem Two\tfollows with tab indents.  \n\n\n\n"
            "Item Three      has     massive     inter-word      spacing      that     continues     on."
        )
        chunk_size = 70
        chunk_overlap = 15
        chunks = split_text(awkward, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreater(len(chunks), 1)
        for idx, c in enumerate(chunks):
            self.assertLessEqual(len(c), chunk_size)

        # All non-whitespace words must be preserved
        orig_words = awkward.split()
        all_chunk_words = set(" ".join(chunks).split())
        for w in orig_words:
            self.assertIn(w, all_chunk_words, f"Word '{w}' was lost")

    def test_awkward_irregular_paragraph_breaks(self):
        """Text with clusters of irregular newlines (\\r\\n and \\n\\n\\n) handles breaks gracefully."""
        text = "Paragraph Alpha.\r\n\r\n\r\n\r\nParagraph Beta has details.\n\n\n\nParagraph Gamma concludes."
        chunk_size = 50
        chunk_overlap = 10
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertLessEqual(len(c), chunk_size)
            self.assertNotIn("\r", c, "Carriage returns must be normalized")

        orig_words = text.split()
        all_chunk_words = set(" ".join(chunks).split())
        for w in orig_words:
            self.assertIn(w, all_chunk_words)

    def test_awkward_long_unbroken_string(self):
        """Continuous long string with zero separators splits safely without infinite loops."""
        unbroken = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/" * 6  # 384 chars
        chunk_size = 60
        chunk_overlap = 15
        chunks = split_text(unbroken, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreaterEqual(len(chunks), 6)
        for idx, c in enumerate(chunks):
            self.assertLessEqual(len(c), chunk_size)

        # Verify forward progress and exact character coverage
        # Every character in unbroken must appear in at least one chunk
        for i in range(len(chunks) - 1):
            curr_c = chunks[i]
            next_c = chunks[i + 1]
            # Must overlap by exactly chunk_overlap on continuous string
            self.assertEqual(curr_c[-chunk_overlap:], next_c[:chunk_overlap])

    def test_every_chunk_strictly_respects_max_chunk_size(self):
        """Property test: No chunk across varied configurations ever exceeds chunk_size."""
        sample_corpus = [
            "Short sentence.",
            "Normal paragraph with several sentences. Each sentence covers an important architectural topic.",
            "Mixed. \n\nPunctuation! And? Questions... \t Along with   extra spaces.",
            "SuperLongUnbrokenTokenWithNoSpacesInsideItWhatsoeverRepeated" * 5,
        ]

        test_configs = [
            (30, 5),
            (50, 10),
            (100, 20),
            (250, 50),
            (1000, 150),
        ]

        for text in sample_corpus:
            for size, overlap in test_configs:
                chunks = split_text(text, chunk_size=size, chunk_overlap=overlap)
                for idx, c in enumerate(chunks):
                    self.assertLessEqual(
                        len(c),
                        size,
                        f"Chunk {idx} (len {len(c)}) exceeded chunk_size {size} for text: {repr(text[:30])}"
                    )

    def test_overlap_never_collapses_to_zero_when_configured(self):
        """Verify that bounded word-boundary snapping never collapses overlap to zero."""
        text = (
            "The quick brown fox jumps over the lazy dog repeatedly to demonstrate "
            "that sentence boundaries and word boundaries remain aligned across every "
            "segment without losing context or dropping overlapping words."
        )

        for overlap in [10, 20, 30, 50]:
            chunk_size = 60
            chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=overlap)
            self.assertGreater(len(chunks), 1)

            for i in range(len(chunks) - 1):
                c1 = chunks[i]
                c2 = chunks[i + 1]

                # Check that at least one word from the end of c1 appears in c2
                words_c1 = c1.split()
                has_common_word = any(w in c2 for w in words_c1[-3:])
                self.assertTrue(
                    has_common_word,
                    f"Overlap collapsed: no shared words between chunk {i} ({repr(c1[-20:])}) "
                    f"and chunk {i+1} ({repr(c2[:20])}) at overlap={overlap}"
                )

    # ------------------------------------------------------------------------
    # 5. Targeted Final Verifications: Exact Overlap, Earliest Split & Lossless
    # ------------------------------------------------------------------------

    @staticmethod
    def _compute_shared_overlap(c1: str, c2: str) -> str:
        """Find the longest common substring that is a suffix of c1 and a prefix of c2."""
        max_k = min(len(c1), len(c2))
        for k in range(max_k, 0, -1):
            if c1[-k:] == c2[:k]:
                return c1[-k:]
        return ""

    def test_measured_overlap_substrings_for_configurations(self):
        """Measure actual shared substrings between consecutive chunks across overlap configurations.

        Verifies that for chunk_overlap values 1, 10, 20, and 50:
        1. A true shared boundary substring exists (suffix of c_i == prefix of c_{i+1}).
        2. Word-boundary snapping never reduces overlap to zero.
        3. The actual measured overlap matches bounded snapping expectations:
           - overlap 1: exactly 1 char (no word snapping applied when max_snap=0)
           - overlap 10: bounded within [5, 15] (max snap = 5)
           - overlap 20: bounded within [10, 30] (max snap = 10)
           - overlap 50: bounded within [30, 70] (max snap = 20)
        """
        prose = (
            "The quick brown fox jumps over the lazy dog repeatedly to test natural overlap behavior "
            "and verify word boundaries. Each chunk should retain useful context and smooth transitions "
            "across boundaries without dropping important words or creating unnecessary micro-chunks."
        )

        overlap_configs = [
            (1, 70, 1, 1),
            (10, 70, 5, 15),
            (20, 70, 10, 30),
            (50, 90, 30, 70),
        ]

        measured_results = {}

        for target_overlap, chunk_size, min_exp, max_exp in overlap_configs:
            chunks = split_text(prose, chunk_size=chunk_size, chunk_overlap=target_overlap)
            self.assertGreater(len(chunks), 1)

            pair_overlaps = []
            for i in range(len(chunks) - 1):
                c1 = chunks[i]
                c2 = chunks[i + 1]
                shared = self._compute_shared_overlap(c1, c2)
                shared_len = len(shared)
                pair_overlaps.append((shared_len, shared))

                # Must not collapse to 0
                self.assertGreater(
                    shared_len,
                    0,
                    f"Overlap collapsed to 0 between chunk {i} and {i+1} at target_overlap={target_overlap}"
                )
                # Must be bounded within snapping limits
                self.assertGreaterEqual(
                    shared_len,
                    min_exp,
                    f"Measured overlap {shared_len} was below minimum {min_exp} at target_overlap={target_overlap}"
                )
                self.assertLessEqual(
                    shared_len,
                    max_exp,
                    f"Measured overlap {shared_len} exceeded maximum {max_exp} at target_overlap={target_overlap}"
                )
                # Every chunk must respect chunk_size
                self.assertLessEqual(len(c1), chunk_size)
                self.assertLessEqual(len(c2), chunk_size)

            measured_results[target_overlap] = pair_overlaps

        # Confirm all 4 configurations recorded valid non-empty measurements
        self.assertEqual(len(measured_results), 4)

    def test_paragraph_breaks_near_earliest_split_position(self):
        """Paragraph breaks positioned near min_split make reasonable forward progress without duplicate chunks."""
        # chunk_size = 100, chunk_overlap = 20
        # min_advance = (100 - 20) // 4 = 20 characters
        # min_split = 20 + 20 = 40 characters
        # Create paragraphs of ~35-40 chars so paragraph breaks occur near min_split
        paragraphs = [f"Section {i:02d} detailed operational guidance notes." for i in range(12)]
        text = "\n\n".join(paragraphs)

        chunk_size = 100
        chunk_overlap = 20
        min_advance = (chunk_size - chunk_overlap) // 4  # 20

        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Should produce a proportionate number of chunks (~6-8), not dozens of duplicate micro-slivers
        self.assertGreaterEqual(len(chunks), 4)
        self.assertLessEqual(len(chunks), 12)

        for i in range(len(chunks)):
            # Respect chunk_size
            self.assertLessEqual(len(chunks[i]), chunk_size)

        for i in range(len(chunks) - 1):
            c1 = chunks[i]
            c2 = chunks[i + 1]

            # Consecutive chunks must not be duplicates
            self.assertNotEqual(c1, c2, f"Chunks {i} and {i+1} are identical duplicates")

            # Shared overlap must not exceed 80% of chunk length (preventing near-duplicate slivers)
            shared = self._compute_shared_overlap(c1, c2)
            max_allowed_duplicate = int(len(c1) * 0.8)
            self.assertLess(
                len(shared),
                max_allowed_duplicate,
                f"Chunk {i+1} is a near-duplicate of chunk {i} ({len(shared)} / {len(c1)} chars shared)"
            )

    def test_long_unbroken_string_character_preservation_and_integrity(self):
        """Long unbroken string has zero missing characters, no duplicate chunks, and terminates safely."""
        # 900-character unbroken string with unique sequential token markers
        unbroken_text = "".join(f"TOKEN{i:04d}" for i in range(100))
        chunk_size = 75
        chunk_overlap = 15

        chunks = split_text(unbroken_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 1. No infinite loop - returns expected chunk count
        expected_chunks = (len(unbroken_text) - chunk_overlap) // (chunk_size - chunk_overlap)
        self.assertGreaterEqual(len(chunks), 10)

        # 2. Every chunk respects chunk_size
        for idx, c in enumerate(chunks):
            self.assertLessEqual(
                len(c),
                chunk_size,
                f"Chunk {idx} (len {len(c)}) exceeded chunk_size {chunk_size}"
            )

        # 3. No duplicate chunks
        self.assertEqual(
            len(chunks),
            len(set(chunks)),
            "Duplicate chunks detected in unbroken string splitting"
        )

        # 4. Zero text loss: stitch chunks along overlap to reconstruct original text exactly
        reconstructed = chunks[0]
        for c in chunks[1:]:
            # The overlap between unbroken chunks must be exactly chunk_overlap
            self.assertEqual(
                c[:chunk_overlap],
                reconstructed[-chunk_overlap:],
                "Unbroken chunk overlap mismatch"
            )
            reconstructed += c[chunk_overlap:]

        self.assertEqual(
            reconstructed,
            unbroken_text,
            "Reconstructed text does not match original; characters were dropped or corrupted"
        )


if __name__ == "__main__":
    unittest.main()

