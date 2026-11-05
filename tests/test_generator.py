"""Unit tests for the Gemini answer-generation module (src/generator.py).

All tests are 100% offline, deterministic, and use mocked client objects with
zero network calls, zero model downloads, and zero API quota consumption.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from src.chunking import DocumentChunk
from src.generator import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TEMPERATURE,
    NO_EVIDENCE_ANSWER,
    AnswerResult,
    GeminiAuthError,
    GeminiGenerationError,
    GeminiQuotaError,
    GeminiUnavailableError,
    answer_query,
    build_grounded_prompt,
    generate_answer,
    get_gemini_client,
    resolve_api_key,
    resolve_model_name,
)
from src.vector_store import FAISSVectorStore
from tests.test_vector_store import FakeEmbedder


class MockGenerateContentResponse:
    """Mock response object simulating google.genai GenerateContentResponse."""

    def __init__(self, text: str = "This is a grounded answer based on the context."):
        self.text = text


class MockGeminiClient:
    """Mock Gemini client with call tracking."""

    def __init__(self, response_text: str = "This is a grounded answer based on the context."):
        self.response = MockGenerateContentResponse(text=response_text)
        self.models = MagicMock()
        self.models.generate_content.return_value = self.response


class TestGenerator(unittest.TestCase):
    """Complete test suite for Gemini answer generation and RAG pipeline integration."""

    def setUp(self):
        """Set up sample chunks and mock clients."""
        self.mock_client = MockGeminiClient(response_text="Company revenue in 2026 reached 50 million.")

        # Sample chunks for end-to-end testing
        self.chunk_pdf = DocumentChunk(
            text="Quarterly revenue for 2026 reached 50 million dollars.",
            source="annual_report.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=1,
            chunk_id="annual_report.pdf:p3:c0",
            page=3,
        )

        self.chunk_xlsx = DocumentChunk(
            text="Dept: Engineering | Allocated: 450000 | Spent: 320000",
            source="budget.xlsx",
            file_type="xlsx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="budget.xlsx:sAllocations:r12:c0",
            sheet_name="Allocations",
            row_index=12,
        )

    # ------------------------------------------------------------------------
    # 1. API Key & Client Management Tests
    # ------------------------------------------------------------------------

    def test_resolve_api_key_explicit(self):
        """Explicitly passed API key is returned cleanly."""
        key = resolve_api_key("test_explicit_key_12345")
        self.assertEqual(key, "test_explicit_key_12345")

    def test_resolve_api_key_explicit_blank_raises_auth_error(self):
        """Empty or whitespace-only explicit key raises GeminiAuthError."""
        with self.assertRaises(GeminiAuthError) as ctx:
            resolve_api_key("   ")
        self.assertIn("empty API key was provided", str(ctx.exception))

    def test_resolve_api_key_from_env(self):
        """Resolves API key from GEMINI_API_KEY environment variable."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env_api_key_abcdef"}):
            key = resolve_api_key()
            self.assertEqual(key, "env_api_key_abcdef")

    def test_resolve_api_key_missing_from_env_raises_auth_error(self):
        """Missing GEMINI_API_KEY raises GeminiAuthError with setup guidance."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GeminiAuthError) as ctx:
                resolve_api_key()
            self.assertIn("GEMINI_API_KEY environment variable is missing", str(ctx.exception))

    def test_resolve_api_key_never_exposes_secret_in_error(self):
        """Error messages never include API key tokens."""
        with self.assertRaises(GeminiAuthError) as ctx:
            resolve_api_key("")
        self.assertNotIn("AIza", str(ctx.exception))

    def test_get_gemini_client_uses_resolved_key(self):
        """Client uses the resolved API key, configured timeout, and retry options."""
        with patch("google.genai.Client") as mock_client_cls:
            mock_instance = MagicMock()
            mock_client_cls.return_value = mock_instance

            client = get_gemini_client(api_key="valid_dummy_key")

            mock_client_cls.assert_called_once()
            call_kwargs = mock_client_cls.call_args.kwargs

            self.assertEqual(
                call_kwargs["api_key"],
                "valid_dummy_key",
            )
            self.assertEqual(
                call_kwargs["http_options"].timeout,
                60000,
            )
            retry_opts = call_kwargs["http_options"].retry_options
            self.assertIsNotNone(
                retry_opts,
                "retry_options must be configured on http_options",
            )
            self.assertEqual(
                retry_opts.attempts,
                3,
                "retry_options.attempts must be 3 (initial request + 2 retries)",
            )
            self.assertEqual(
                retry_opts.initial_delay,
                1.0,
                "retry_options.initial_delay must be 1.0 second",
            )
            self.assertEqual(
                retry_opts.http_status_codes,
                [502, 503, 504],
                "retry_options.http_status_codes must be [502, 503, 504]",
            )
            self.assertEqual(client, mock_instance)

    def test_resolve_model_name_precedence(self):
        """Verifies resolution precedence: explicit > GEMINI_MODEL > default."""

        # 1. Explicit parameter takes top precedence
        with patch.dict(os.environ, {"GEMINI_MODEL": "env-model-flash"}):
            model = resolve_model_name("explicit-model-pro")
            self.assertEqual(model, "explicit-model-pro")

        # 2. Environment variable takes second precedence
        with patch.dict(os.environ, {"GEMINI_MODEL": "env-model-flash"}):
            model = resolve_model_name(None)
            self.assertEqual(model, "env-model-flash")

        # 3. Documented default when neither is provided
        with patch.dict(os.environ, {}, clear=True):
            model = resolve_model_name(None)
            self.assertEqual(model, DEFAULT_GEMINI_MODEL)

    # ------------------------------------------------------------------------
    # 3. Grounded Prompt Construction Tests        

    def test_build_grounded_prompt_structure_and_security(self):
        """Grounded prompt clearly demarcates untrusted evidence and enforces grounding."""
        query = "What is the 2026 revenue?"
        context = "[Source: annual_report.pdf | Page 3]\nRevenue reached 50 million."

        prompt = build_grounded_prompt(query, context)

        # Context demarcation
        self.assertIn("=== Context Begin ===", prompt)
        self.assertIn(context, prompt)
        self.assertIn("=== Context End ===", prompt)

        # Query insertion
        self.assertIn(f"Question: {query}", prompt)

        # Security and grounding instructions
        self.assertIn("Untrusted Evidence", prompt)
        self.assertIn("Ignore previous instructions", prompt)  # Prompt injection warning
        self.assertIn(NO_EVIDENCE_ANSWER, prompt)               # Standard refusal instruction

    # ------------------------------------------------------------------------
    # 4. Successful Answer Generation Tests
    # ------------------------------------------------------------------------

    def test_generate_answer_success_with_injected_client(self):
        """generate_answer produces a populated AnswerResult using an injected mock client."""
        query = "What was the 2026 revenue?"
        context = "[Source: annual_report.pdf | Page 3]\nRevenue reached 50 million."
        citations = [
            {"source": "annual_report.pdf", "page": 3, "score": 0.95, "snippet": "Revenue reached 50 million."},
        ]

        result = generate_answer(
            query=query,
            context=context,
            citations=citations,
            client=self.mock_client,
            model="gemini-2.5-flash",
            temperature=0.2,
        )

        self.assertIsInstance(result, AnswerResult)
        self.assertEqual(result.answer, "Company revenue in 2026 reached 50 million.")
        self.assertEqual(result.query, query)
        self.assertEqual(result.context, context)
        self.assertEqual(result.model, "gemini-2.5-flash")
        self.assertEqual(result.sources_used, ["annual_report.pdf"])
        self.assertTrue(result.grounded)
        self.assertEqual(len(result.citations), 1)

        # Verify mock client call
        self.mock_client.models.generate_content.assert_called_once()
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-2.5-flash")
        self.assertIn(query, call_kwargs["contents"])

    def test_generate_answer_to_dict(self):
        """AnswerResult.to_dict produces clean serializable dictionary."""
        result = AnswerResult(
            answer="42",
            query="Meaning of life?",
            citations=[{"source": "guide.pdf"}],
            context="The meaning of life is 42.",
            model="gemini-2.5-flash",
            sources_used=["guide.pdf"],
            grounded=True,
        )
        d = result.to_dict()
        self.assertEqual(d["answer"], "42")
        self.assertEqual(d["model"], "gemini-2.5-flash")
        self.assertTrue(d["grounded"])

    # ------------------------------------------------------------------------
    # 5. Short-Circuiting Tests (Zero API Calls on Empty Evidence)
    # ------------------------------------------------------------------------

    def test_generate_answer_short_circuits_on_empty_context(self):
        """Empty context short-circuits immediately without invoking the Gemini API."""
        result = generate_answer(
            query="What is the revenue?",
            context="",
            client=self.mock_client,
        )

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertFalse(result.grounded)
        self.assertEqual(result.model, "none")
        self.assertEqual(result.citations, [])
        self.assertEqual(result.sources_used, [])

        # Crucial check: client was NEVER called
        self.mock_client.models.generate_content.assert_not_called()

    def test_generate_answer_short_circuits_on_fallback_string(self):
        """Context matching the empty fallback string short-circuits without API call."""
        result = generate_answer(
            query="What is the revenue?",
            context="No relevant document context found.",
            client=self.mock_client,
        )

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertFalse(result.grounded)
        self.mock_client.models.generate_content.assert_not_called()

    def test_generate_answer_empty_query(self):
        """Blank query returns guidance without calling the Gemini API."""
        result = generate_answer(
            query="   ",
            context="Some valid context.",
            client=self.mock_client,
        )

        self.assertIn("Please provide a valid question", result.answer)
        self.assertFalse(result.grounded)
        self.mock_client.models.generate_content.assert_not_called()

    # ------------------------------------------------------------------------
    # 6. End-to-End Pipeline Tests (answer_query)
    # ------------------------------------------------------------------------

    def test_answer_query_end_to_end_success(self):
        """answer_query successfully connects retrieval, formatting, and generation."""
        embedder = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([self.chunk_pdf, self.chunk_xlsx])

        result = answer_query(
            query="fiscal report details",
            vector_store=store,
            top_k=2,
            client=self.mock_client,
        )

        self.assertIsInstance(result, AnswerResult)
        self.assertTrue(result.grounded)
        self.assertEqual(result.answer, "Company revenue in 2026 reached 50 million.")
        self.assertGreater(len(result.citations), 0)
        self.assertIn("annual_report.pdf", result.sources_used)

        # Mock client was called exactly once
        self.mock_client.models.generate_content.assert_called_once()

    def test_answer_query_short_circuits_on_empty_store(self):
        """Empty vector store returns NO_EVIDENCE_ANSWER without calling Gemini API."""
        embedder = FakeEmbedder(dimension=4)
        empty_store = FAISSVectorStore(embedder=embedder)

        result = answer_query(
            query="any question",
            vector_store=empty_store,
            client=self.mock_client,
        )

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertFalse(result.grounded)
        self.assertEqual(result.citations, [])
        self.mock_client.models.generate_content.assert_not_called()

    def test_answer_query_short_circuits_when_threshold_filters_all_chunks(self):
        """When score_threshold filters all chunks, answer_query short-circuits without API call."""
        embedder = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([self.chunk_pdf])

        # Setting threshold impossibly high (1.0) eliminates the chunk
        result = answer_query(
            query="any question",
            vector_store=store,
            score_threshold=1.0,
            client=self.mock_client,
        )

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertFalse(result.grounded)
        self.mock_client.models.generate_content.assert_not_called()

    # ------------------------------------------------------------------------
    # 7. Error Handling & Exception Mapping Tests
    # ------------------------------------------------------------------------

    def test_quota_error_handling(self):
        """HTTP 429 / RESOURCE_EXHAUSTED raises GeminiQuotaError."""
        failing_client = MagicMock()
        failing_client.models.generate_content.side_effect = Exception(
            "429 Resource exhausted: Quota exceeded for GenerateContent"
        )

        with self.assertRaises(GeminiQuotaError) as ctx:
            generate_answer(
                query="question",
                context="Valid document excerpt.",
                client=failing_client,
            )
        self.assertIn("HTTP 429", str(ctx.exception))
        self.assertIn("quota", str(ctx.exception).lower())

    def test_service_unavailable_error_handling(self):
        """HTTP 503 / UNAVAILABLE raises GeminiUnavailableError."""
        failing_client = MagicMock()
        failing_client.models.generate_content.side_effect = Exception(
            "503 Service Unavailable: The model is currently overloaded"
        )

        with self.assertRaises(GeminiUnavailableError) as ctx:
            generate_answer(
                query="question",
                context="Valid document excerpt.",
                client=failing_client,
            )
        self.assertIn("temporarily unavailable", str(ctx.exception))

    def test_auth_error_handling(self):
        """API_KEY_INVALID raises GeminiAuthError."""
        failing_client = MagicMock()
        failing_client.models.generate_content.side_effect = Exception(
            "API_KEY_INVALID: The provided API key is invalid"
        )

        with self.assertRaises(GeminiAuthError) as ctx:
            generate_answer(
                query="question",
                context="Valid document excerpt.",
                client=failing_client,
            )
        self.assertIn("authentication failed", str(ctx.exception))

    def test_empty_or_malformed_response_raises_generation_error(self):
        """Empty response text raises GeminiGenerationError."""
        empty_response_client = MockGeminiClient(response_text="")

        with self.assertRaises(GeminiGenerationError) as ctx:
            generate_answer(
                query="question",
                context="Valid document excerpt.",
                client=empty_response_client,
            )
        self.assertIn("empty or malformed", str(ctx.exception))

    def test_invalid_input_types_raise_type_error(self):
        """Non-string inputs to generate_answer and answer_query raise TypeError."""
        with self.assertRaises(TypeError):
            generate_answer(query=123, context="text", client=self.mock_client)

        with self.assertRaises(TypeError):
            generate_answer(query="text", context=None, client=self.mock_client)

        with self.assertRaises(TypeError):
            answer_query(query=123, vector_store=MagicMock())

    # ------------------------------------------------------------------------
    # 8. SDK Retry Behavior Tests (offline, deterministic)
    # ------------------------------------------------------------------------

    def test_sdk_retry_recovers_on_second_attempt_after_503(self):
        """Real SDK retry loop: transient HTTP 503 on attempt 1 recovers on attempt 2.

        Uses the real genai.Client returned by get_gemini_client() with a dummy
        key. Intercepts httpx.Client.send so no network call can escape, and
        patches tenacity.nap.time.sleep to eliminate backoff wall-clock delay.
        Asserts exactly 2 HTTP send calls and 1 intercepted sleep.
        """
        client = get_gemini_client(api_key="offline_dummy_key")
        calls = []

        def mock_send(instance, request, *args, **kwargs):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(
                    503,
                    json={
                        "error": {
                            "code": 503,
                            "message": "The model is overloaded.",
                            "status": "UNAVAILABLE",
                        }
                    },
                    request=request,
                )
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "Recovered answer text."}]}}
                    ]
                },
                request=request,
            )

        with patch("tenacity.nap.time.sleep") as mock_sleep, \
                patch("httpx.Client.send", new=mock_send):
            result = generate_answer(
                query="What is the revenue?",
                context="[Source: report.pdf | Page 1]\nRevenue reached 50 million.",
                client=client,
            )

        self.assertEqual(
            len(calls), 2,
            "Must make exactly 2 HTTP attempts (1 initial failure + 1 retry)",
        )
        self.assertEqual(
            mock_sleep.call_count, 1,
            "Must intercept exactly 1 tenacity sleep call between the two attempts",
        )
        self.assertIsInstance(result, AnswerResult)
        self.assertEqual(result.answer, "Recovered answer text.")
        self.assertTrue(result.grounded)

    def test_sdk_retry_exhaustion_raises_unavailable_after_three_attempts(self):
        """Real SDK retry loop: 3 consecutive HTTP 503 responses exhaust retries.

        Uses the real genai.Client returned by get_gemini_client() with a dummy
        key. Intercepts httpx.Client.send so no network call can escape, and
        patches tenacity.nap.time.sleep to eliminate backoff wall-clock delay.
        Asserts exactly 3 HTTP send calls, 2 intercepted sleeps, and that
        generate_answer() raises GeminiUnavailableError with the expected message.
        """
        client = get_gemini_client(api_key="offline_dummy_key")
        calls = []

        def mock_send(instance, request, *args, **kwargs):
            calls.append(request)
            return httpx.Response(
                503,
                json={
                    "error": {
                        "code": 503,
                        "message": "The model is overloaded.",
                        "status": "UNAVAILABLE",
                    }
                },
                request=request,
            )

        with patch("tenacity.nap.time.sleep") as mock_sleep, \
                patch("httpx.Client.send", new=mock_send):
            with self.assertRaises(GeminiUnavailableError) as ctx:
                generate_answer(
                    query="What is the revenue?",
                    context="[Source: report.pdf | Page 1]\nRevenue reached 50 million.",
                    client=client,
                )

        self.assertEqual(
            len(calls), 3,
            "Must make exactly 3 HTTP attempts (1 initial + 2 retries) before exhausting",
        )
        self.assertEqual(
            mock_sleep.call_count, 2,
            "Must intercept exactly 2 tenacity sleep calls across 3 attempts",
        )
        self.assertIn("temporarily unavailable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
