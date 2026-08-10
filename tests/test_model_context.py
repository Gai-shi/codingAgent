from __future__ import annotations

import unittest

from ai_job.provider_adapters import DEFAULT_CONTEXT_WINDOW, lookup_context_window, resolve_context_window


class ModelContextTest(unittest.TestCase):
    def test_lookup_context_window_uses_known_model_registry(self):
        self.assertEqual(lookup_context_window("gpt-4.1"), 1_047_576)
        self.assertEqual(lookup_context_window(" GPT-4O-MINI "), 128_000)
        self.assertEqual(lookup_context_window("unknown-model"), None)

    def test_resolve_context_window_prefers_override_then_registry_then_default(self):
        self.assertEqual(resolve_context_window("gpt-4.1", context_window_override=123), 123)
        self.assertEqual(resolve_context_window("gpt-4o"), 128_000)
        self.assertEqual(resolve_context_window("unknown-model"), DEFAULT_CONTEXT_WINDOW)


if __name__ == "__main__":
    unittest.main()
