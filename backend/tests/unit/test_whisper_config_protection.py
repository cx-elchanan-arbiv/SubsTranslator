#!/usr/bin/env python3
"""
🛡️ Protection Tests for Whisper Configuration
Tests to ensure critical Whisper settings don't regress to problematic values.
"""

import os
import sys

import pytest

# Add backend to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, backend_dir)

from services.whisper_smart import SmartWhisperManager


@pytest.mark.unit
class TestWhisperConfigProtection:
    """Tests to protect against regression of critical Whisper settings."""

    def test_compute_type_is_int8_for_all_models(self):
        """
        🚨 CRITICAL: Ensure compute_type is always int8, never float16

        This test prevents regression to float16 which causes:
        - "float16 compute type not supported" errors on CPU
        - Fallback to base model instead of large
        - Poor transcription quality
        """
        manager = SmartWhisperManager()

        # Test all model sizes
        test_models = ["tiny", "base", "small", "medium", "large"]

        for model_name in test_models:
            # Mock the WhisperModel to capture the compute_type parameter
            captured_params = {}

            def mock_whisper_model(*args, **kwargs):
                captured_params.update(kwargs)

                # Return a mock model object
                class MockModel:
                    pass

                return MockModel()

            # Temporarily replace WhisperModel
            import services.whisper_smart as whisper_smart

            original_whisper_model = whisper_smart.WhisperModel
            whisper_smart.WhisperModel = mock_whisper_model

            try:
                # This should trigger model loading with int8
                manager.load_model(model_name)

                # Verify critical parameters
                assert (
                    "compute_type" in captured_params
                ), f"compute_type not set for {model_name}"
                assert (
                    captured_params["compute_type"] == "int8"
                ), f"❌ REGRESSION DETECTED: {model_name} uses {captured_params['compute_type']} instead of int8!"

                assert (
                    captured_params["device"] == "cpu"
                ), f"Device should be CPU for {model_name}"

            finally:
                # Restore original WhisperModel
                whisper_smart.WhisperModel = original_whisper_model
                # Clean up loaded models
                manager.loaded_models.clear()

    def test_large_model_specifically_uses_int8(self):
        """
        🎯 SPECIFIC TEST: Large model must use int8

        The large model is most critical because:
        - It's the most accurate model
        - It was failing with float16
        - Users expect it to work
        """
        manager = SmartWhisperManager()

        captured_params = {}

        def mock_whisper_model(*args, **kwargs):
            captured_params.update(kwargs)

            class MockModel:
                pass

            return MockModel()

        import services.whisper_smart as whisper_smart

        original_whisper_model = whisper_smart.WhisperModel
        whisper_smart.WhisperModel = mock_whisper_model

        try:
            manager.load_model("large")

            # Critical assertions for large model
            assert (
                captured_params["compute_type"] == "int8"
            ), "❌ CRITICAL REGRESSION: Large model not using int8!"
            assert (
                captured_params["device"] == "cpu"
            ), "Large model should use CPU device"

        finally:
            whisper_smart.WhisperModel = original_whisper_model
            manager.loaded_models.clear()

    def test_no_float16_anywhere(self):
        """
        🚫 NEGATIVE TEST: Ensure float16 is never used

        float16 causes "compute type not supported" errors on CPU
        """
        manager = SmartWhisperManager()

        # Check all possible code paths
        test_models = ["tiny", "base", "small", "medium", "large"]

        for model_name in test_models:
            captured_params = {}

            def mock_whisper_model(*args, **kwargs):
                captured_params.update(kwargs)
                # Verify float16 is never used
                if "compute_type" in kwargs:
                    assert (
                        kwargs["compute_type"] != "float16"
                    ), f"❌ FORBIDDEN: {model_name} is trying to use float16!"

                class MockModel:
                    pass

                return MockModel()

            import services.whisper_smart as whisper_smart

            original_whisper_model = whisper_smart.WhisperModel
            whisper_smart.WhisperModel = mock_whisper_model

            try:
                manager.load_model(model_name)

                # Double-check the captured parameters
                if "compute_type" in captured_params:
                    assert (
                        captured_params["compute_type"] != "float16"
                    ), f"❌ REGRESSION: {model_name} captured float16!"

            finally:
                whisper_smart.WhisperModel = original_whisper_model
                manager.loaded_models.clear()

    def test_transcribe_smart_preserves_int8(self):
        """
        🔄 INTEGRATION TEST: Full transcribe_smart flow uses int8

        Tests the complete flow to ensure int8 is used end-to-end
        """
        manager = SmartWhisperManager()

        captured_params = {}

        def mock_whisper_model(*args, **kwargs):
            captured_params.update(kwargs)

            class MockModel:
                def transcribe(self, *args, **kwargs):
                    # Mock transcription result
                    class MockSegment:
                        def __init__(self, start, end, text):
                            self.start = start
                            self.end = end
                            self.text = text

                    class MockInfo:
                        language = "en"
                        language_probability = 0.95

                    segments = [MockSegment(0.0, 1.0, "Test")]
                    info = MockInfo()
                    return segments, info

            return MockModel()

        import services.whisper_smart as whisper_smart

        original_whisper_model = whisper_smart.WhisperModel
        whisper_smart.WhisperModel = mock_whisper_model

        try:
            # Test with large model preference (most critical)
            result = manager.transcribe_smart(
                "dummy_audio.wav", model_preference="large"
            )

            # Verify the result structure
            assert "text" in result
            assert "segments" in result
            assert "language" in result

            # Most importantly: verify int8 was used
            assert (
                captured_params.get("compute_type") == "int8"
            ), "❌ CRITICAL: transcribe_smart didn't use int8!"

        finally:
            whisper_smart.WhisperModel = original_whisper_model
            manager.loaded_models.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
