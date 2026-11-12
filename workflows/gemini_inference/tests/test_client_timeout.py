import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gemini_interiority.gemini_client import GeminiConfig, GeminiRunnerClient


class ClientTimeoutTests(unittest.TestCase):
    def test_default_config_keeps_thinking_enabled_with_large_output_budget(self):
        config = GeminiConfig(api_key="test-key")

        self.assertEqual(config.classify_thinking_level, "high")
        self.assertEqual(config.max_output_tokens, 4096)

    @patch("gemini_interiority.gemini_client.genai.Client")
    def test_developer_client_receives_timeout_in_milliseconds(self, client_class):
        config = GeminiConfig(api_key="test-key", request_timeout_seconds=60)

        GeminiRunnerClient(config)

        http_options = client_class.call_args.kwargs["http_options"]
        self.assertEqual(http_options.timeout, 60_000)

    @patch("gemini_interiority.gemini_client.genai.Client")
    def test_zero_disables_request_timeout(self, client_class):
        config = GeminiConfig(api_key="test-key", request_timeout_seconds=0)

        GeminiRunnerClient(config)

        self.assertIsNone(client_class.call_args.kwargs["http_options"])

    @patch("gemini_interiority.gemini_client.genai.Client")
    def test_generation_configs_omit_temperature(self, client_class):
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="LOW", thought=False)]
                    ),
                    finish_reason=SimpleNamespace(name="STOP"),
                )
            ],
            usage_metadata=None,
            text="LOW",
        )
        client_class.return_value.models.generate_content.return_value = response
        client = GeminiRunnerClient(GeminiConfig(api_key="test-key"))

        client.generate("gemini-3.5-flash", "system", "user")
        client.generate_label(
            "gemini-3.5-flash",
            "system",
            "user",
            labels=["HIGH", "LOW", "NONE"],
        )

        for call in client_class.return_value.models.generate_content.call_args_list:
            request_config = call.kwargs["config"]
            self.assertIsNone(request_config.temperature)
            self.assertNotIn("temperature", request_config.model_dump(exclude_none=True))


if __name__ == "__main__":
    unittest.main()
