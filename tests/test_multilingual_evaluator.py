"""Tests for Multilingual Whisper evaluator."""

import pytest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd

from mergekit.evaluation.multilingual_evaluator import MultilingualWhisperEvaluator
from mergekit.evaluation.whisper_evaluator import WhisperEvaluator


class TestMultilingualWhisperEvaluator:
    """Test MultilingualWhisperEvaluator class."""
    
    @patch("mergekit.evaluation.multilingual_evaluator.WhisperEvaluator")
    def test_initialization(self, mock_whisper_evaluator):
        """Test initialization of MultilingualWhisperEvaluator."""
        # Setup mock
        mock_evaluator_instance = MagicMock()
        mock_whisper_evaluator.return_value = mock_evaluator_instance
        
        # Initialize evaluator
        evaluator = MultilingualWhisperEvaluator(
            model_path="test/model/path", 
            device="cpu",
            batch_size=16
        )
        
        # Check if WhisperEvaluator was initialized correctly
        mock_whisper_evaluator.assert_called_once_with(
            model_path="test/model/path",
            device="cpu",
            batch_size=16
        )
    
    @patch("mergekit.evaluation.whisper_evaluator.WhisperEvaluator.evaluate_dataset")
    def test_evaluate_multilingual(self, mock_evaluate_dataset):
        """Test evaluate_multilingual method."""
        # Setup mock
        mock_evaluate_dataset.side_effect = [
            {"wer": 0.2, "cer": 0.1, "bleu": 0.8},
            {"wer": 0.3, "cer": 0.15, "bleu": 0.7}
        ]
        
        # Initialize evaluator with patched WhisperEvaluator
        with patch("mergekit.evaluation.multilingual_evaluator.WhisperEvaluator"):
            evaluator = MultilingualWhisperEvaluator(model_path="test/model/path")
            
            # Call evaluate_multilingual
            result = evaluator.evaluate_multilingual(
                datasets={"en": "dataset1", "fr": "dataset2"},
                max_samples_per_language=50,
                split="test"
            )
            
            # Check if evaluate_dataset was called for each language
            assert mock_evaluate_dataset.call_count == 2
            mock_evaluate_dataset.assert_any_call(
                dataset_name="dataset1",
                split="test",
                language="en",
                max_samples=50
            )
            mock_evaluate_dataset.assert_any_call(
                dataset_name="dataset2",
                split="test",
                language="fr",
                max_samples=50
            )
            
            # Check the result
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "language" in result.columns
            assert "wer" in result.columns
            assert "cer" in result.columns
            assert "bleu" in result.columns
    
    @patch("mergekit.evaluation.whisper_evaluator.WhisperEvaluator.evaluate_dataset")
    def test_evaluate_common_dataset(self, mock_evaluate_dataset):
        """Test evaluate_common_dataset method."""
        # Setup mock
        mock_evaluate_dataset.side_effect = [
            {"wer": 0.2, "cer": 0.1, "bleu": 0.8},
            {"wer": 0.3, "cer": 0.15, "bleu": 0.7}
        ]
        
        # Initialize evaluator with patched WhisperEvaluator
        with patch("mergekit.evaluation.multilingual_evaluator.WhisperEvaluator"):
            evaluator = MultilingualWhisperEvaluator(model_path="test/model/path")
            
            # Call evaluate_common_dataset
            result = evaluator.evaluate_common_dataset(
                dataset_name="common_dataset",
                languages=["en", "fr"],
                max_samples_per_language=50,
                split="test"
            )
            
            # Check if evaluate_dataset was called for each language
            assert mock_evaluate_dataset.call_count == 2
            mock_evaluate_dataset.assert_any_call(
                dataset_name="common_dataset",
                split="test.en",
                language="en",
                max_samples=50
            )
            mock_evaluate_dataset.assert_any_call(
                dataset_name="common_dataset",
                split="test.fr",
                language="fr",
                max_samples=50
            )
            
            # Check the result
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "language" in result.columns
            assert "wer" in result.columns
            assert "cer" in result.columns
            assert "bleu" in result.columns 