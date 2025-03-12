"""Tests for ASR evaluation tools."""

import pytest
from unittest.mock import patch, MagicMock

from mergekit.evaluation.asr_metrics import ASRMetrics


class TestASRMetrics:
    """Test ASR metrics calculation."""
    
    def test_wer_calculation(self):
        """Test Word Error Rate calculation."""
        metrics = ASRMetrics(compute_cer=False, compute_bleu=False)
        
        references = ["this is a test", "another test sentence"]
        predictions = ["this is a test", "other test sentence"]
        
        with patch("jiwer.wer", return_value=0.25) as mock_wer:
            results = metrics.evaluate(references, predictions)
            
            assert mock_wer.called
            assert "wer" in results
            assert results["wer"] == 0.25
            assert "cer" not in results
            assert "bleu" not in results
    
    def test_cer_calculation(self):
        """Test Character Error Rate calculation."""
        metrics = ASRMetrics(compute_wer=False, compute_bleu=False)
        
        references = ["this is a test", "another test sentence"]
        predictions = ["this is a test", "other test sentence"]
        
        with patch("jiwer.cer", return_value=0.1) as mock_cer:
            results = metrics.evaluate(references, predictions)
            
            assert mock_cer.called
            assert "cer" in results
            assert results["cer"] == 0.1
            assert "wer" not in results
            assert "bleu" not in results
    
    def test_bleu_calculation(self):
        """Test BLEU score calculation."""
        metrics = ASRMetrics(compute_wer=False, compute_cer=False)
        
        references = ["this is a test", "another test sentence"]
        predictions = ["this is a test", "other test sentence"]
        
        mock_bleu = MagicMock()
        mock_bleu.compute.return_value = {"bleu": 0.8}
        
        with patch("evaluate.load", return_value=mock_bleu) as mock_load:
            metrics = ASRMetrics(compute_wer=False, compute_cer=False)
            results = metrics.evaluate(references, predictions)
            
            assert mock_load.called
            assert mock_bleu.compute.called
            assert "bleu" in results
            assert results["bleu"] == 0.8
            assert "wer" not in results
            assert "cer" not in results
    
    def test_all_metrics(self):
        """Test all metrics calculated together."""
        mock_bleu = MagicMock()
        mock_bleu.compute.return_value = {"bleu": 0.8}
        
        with patch("jiwer.wer", return_value=0.25) as mock_wer, \
             patch("jiwer.cer", return_value=0.1) as mock_cer, \
             patch("evaluate.load", return_value=mock_bleu) as mock_load:
            
            metrics = ASRMetrics()
            references = ["this is a test", "another test sentence"]
            predictions = ["this is a test", "other test sentence"]
            
            results = metrics.evaluate(references, predictions)
            
            assert mock_wer.called
            assert mock_cer.called
            assert mock_load.called
            assert mock_bleu.compute.called
            
            assert "wer" in results
            assert "cer" in results
            assert "bleu" in results
            assert results["wer"] == 0.25
            assert results["cer"] == 0.1
            assert results["bleu"] == 0.8 