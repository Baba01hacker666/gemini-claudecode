import unittest
from app.translator import anthropic_messages_to_gemini, gemini_response_to_anthropic, build_ollama_payload, ollama_response_to_anthropic

class TestTranslator(unittest.TestCase):
    def test_anthropic_messages_to_gemini_text(self):
        messages = [{"role": "user", "content": "Hello"}]
        id_to_name, id_to_sig = {}, {}
        gemini_messages = anthropic_messages_to_gemini(messages, id_to_name, id_to_sig)

        self.assertEqual(len(gemini_messages), 1)
        self.assertEqual(gemini_messages[0]["role"], "user")
        self.assertEqual(gemini_messages[0]["parts"][0]["text"], "Hello")

    def test_anthropic_messages_to_gemini_tool_use(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "London"}}]}
        ]
        id_to_name, id_to_sig = {}, {}
        gemini_messages = anthropic_messages_to_gemini(messages, id_to_name, id_to_sig)

        self.assertEqual(len(gemini_messages), 2)
        self.assertEqual(gemini_messages[1]["role"], "model")
        self.assertEqual(gemini_messages[1]["parts"][0]["functionCall"]["name"], "get_weather")
        self.assertEqual(gemini_messages[1]["parts"][0]["functionCall"]["args"], {"location": "London"})

    def test_gemini_response_to_anthropic_text(self):
        gemini_response = {
            "candidates": [{
                "content": {"parts": [{"text": "Hello back!"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
        }
        anthropic_response = gemini_response_to_anthropic(gemini_response, "claude-3-5-sonnet-20241022")

        self.assertEqual(anthropic_response["role"], "assistant")
        self.assertEqual(anthropic_response["content"][0]["text"], "Hello back!")
        self.assertEqual(anthropic_response["stop_reason"], "end_turn")
        self.assertEqual(anthropic_response["usage"]["input_tokens"], 10)
        self.assertEqual(anthropic_response["usage"]["output_tokens"], 5)

    def test_build_ollama_payload_text(self):
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello Ollama"}],
            "stream": True,
            "temperature": 0.2,
        }
        payload, original_model, _ = build_ollama_payload(body)
        self.assertEqual(original_model, "claude-3-5-sonnet-20241022")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "Hello Ollama")
        self.assertEqual(payload["options"]["temperature"], 0.2)

    def test_ollama_response_to_anthropic_text(self):
        resp = {"message": {"content": "Hi from Ollama"}}
        anthropic_response = ollama_response_to_anthropic(resp, "claude-sonnet-4-6")
        self.assertEqual(anthropic_response["role"], "assistant")
        self.assertEqual(anthropic_response["content"][0]["text"], "Hi from Ollama")

if __name__ == "__main__":
    unittest.main()
