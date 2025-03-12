"""Tests for Whisper evaluator."""

import pytest
from unittest.mock import patch, MagicMock, ANY

import torch
import numpy as np

from mergekit.evaluation.whisper_evaluator import WhisperEvaluator
from mergekit.evaluation.asr_metrics import ASRMetrics


class TestWhisperEvaluator:
    """Test WhisperEvaluator class."""
    
    @patch("mergekit.evaluation.whisper_evaluator.WhisperForConditionalGeneration")
    @patch("mergekit.evaluation.whisper_evaluator.WhisperProcessor")
    def test_initialization(self, mock_processor, mock_model):
        """Test initialization of WhisperEvaluator."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        # Initialize evaluator
        evaluator = WhisperEvaluator(model_path="test/model/path", device="cpu")
        
        # Check if model and processor were loaded correctly
        mock_model.from_pretrained.assert_called_once_with(
            "test/model/path", 
            torch_dtype=torch.float32
        )
        mock_processor.from_pretrained.assert_called_once_with("test/model/path")
        
        # Check if model was moved to the correct device
        mock_model_instance.to.assert_called_once_with("cpu")
        
        # Check if ASRMetrics was initialized
        assert isinstance(evaluator.metrics, ASRMetrics)
    
    @patch("mergekit.evaluation.whisper_evaluator.WhisperForConditionalGeneration")
    @patch("mergekit.evaluation.whisper_evaluator.WhisperProcessor")
    def test_transcribe(self, mock_processor, mock_model):
        """Test transcribe method."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        # Mock _load_audio method
        with patch.object(WhisperEvaluator, '_load_audio', return_value=np.zeros(1000)):
            # Mock processor to return input features
            mock_processor_instance.return_value = {"input_features": torch.zeros(1, 80, 3000)}
            
            # Mock model.generate to return token IDs
            mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])
            
            # Mock processor.batch_decode to return transcriptions
            mock_processor_instance.batch_decode.return_value = ["test transcription"]
            
            # Initialize evaluator
            evaluator = WhisperEvaluator(model_path="test/model/path", device="cpu")
            
            # Call transcribe
            result = evaluator.transcribe(audio_paths=["test/audio.wav"])
            
            # Check if processor was called with the audio
            mock_processor_instance.assert_called_once()
            
            # Check if model.generate was called
            mock_model_instance.generate.assert_called_once()
            
            # Check if processor.batch_decode was called
            mock_processor_instance.batch_decode.assert_called_once_with(ANY, skip_special_tokens=True)
            
            # Check the result
            assert result == ["test transcription"]
    
    @patch("mergekit.evaluation.whisper_evaluator.WhisperForConditionalGeneration")
    @patch("mergekit.evaluation.whisper_evaluator.WhisperProcessor")
    @patch("mergekit.evaluation.whisper_evaluator.load_dataset")
    def test_evaluate_dataset(self, mock_load_dataset, mock_processor, mock_model):
        """Test evaluate_dataset method."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        mock_dataset.__getitem__.side_effect = lambda idx: {
            "audio": {"array": np.zeros(1000)},
            "text": "test reference"
        }
        mock_dataset.__iter__.return_value = iter([
            {"audio": {"array": np.zeros(1000)}, "text": "test reference 1"},
            {"audio": {"array": np.zeros(1000)}, "text": "test reference 2"}
        ])
        mock_dataset.features = {}
        mock_dataset.__getitem__.return_value = ["test reference 1", "test reference 2"]
        mock_load_dataset.return_value = mock_dataset
        
        # Mock processor to return input features
        mock_inputs = {
            "input_features": torch.zeros(2, 80, 3000)
        }
        mock_processor_instance.return_value = mock_inputs
        
        # Create a mock for the to() method on tensors
        mock_tensor = MagicMock()
        mock_inputs["input_features"].to = MagicMock(return_value=mock_tensor)
        
        # Mock model.generate to return token IDs
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3], [4, 5, 6]])
        
        # Mock processor.batch_decode to return transcriptions
        mock_processor_instance.batch_decode.return_value = ["test prediction 1", "test prediction 2"]
        
        # Mock ASRMetrics.evaluate
        with patch.object(ASRMetrics, 'evaluate', return_value={"wer": 0.25, "cer": 0.1, "bleu": 0.8}) as mock_evaluate:
            # Initialize evaluator
            evaluator = WhisperEvaluator(model_path="test/model/path", device="cpu")
            
            # Call evaluate_dataset
            result = evaluator.evaluate_dataset(
                dataset_name="test/dataset",
                split="test",
                language="en"
            )
            
            # Check if dataset was loaded
            mock_load_dataset.assert_called_once_with("test/dataset", split="test")
            
            # Check if processor was called with the audio
            mock_processor_instance.assert_called_once()
            
            # Check if model.generate was called
            mock_model_instance.generate.assert_called_once()
            
            # Check if processor.batch_decode was called
            mock_processor_instance.batch_decode.assert_called_once_with(ANY, skip_special_tokens=True)
            
            # Check if ASRMetrics.evaluate was called
            mock_evaluate.assert_called_once()
            
            # Check the result
            assert result == {"wer": 0.25, "cer": 0.1, "bleu": 0.8} 