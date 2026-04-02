import unittest
from app.schema import sanitise_schema, validate_anthropic_request

class TestSchema(unittest.TestCase):
    def test_sanitise_schema_nested(self):
        raw_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name", "title": "Location Title"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["location"],
            "$schema": "http://json-schema.org/draft-07/schema#"
        }
        sanitised = sanitise_schema(raw_schema)

        self.assertNotIn("$schema", sanitised)
        self.assertNotIn("title", sanitised["properties"]["location"])
        self.assertNotIn("default", sanitised["properties"]["unit"])
        self.assertEqual(sanitised["type"], "object")

    def test_validate_anthropic_request_valid(self):
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hello"}]
        }
        # Should not raise
        validate_anthropic_request(body)

    def test_validate_anthropic_request_invalid_missing_model(self):
        body = {
            "messages": [{"role": "user", "content": "hello"}]
        }
        with self.assertRaisesRegex(ValueError, "Missing 'model' field"):
            validate_anthropic_request(body)

    def test_validate_anthropic_request_invalid_role(self):
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "system", "content": "you are an assistant"}]
        }
        with self.assertRaisesRegex(ValueError, "Invalid role: system"):
            validate_anthropic_request(body)

if __name__ == "__main__":
    unittest.main()
