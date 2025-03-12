"""Evaluation metrics for Automatic Speech Recognition (ASR) models."""

import jiwer
import evaluate
import numpy as np
from typing import Dict, List, Union, Optional


class ASRMetrics:
    """Evaluation metrics for ASR models."""
    
    def __init__(self, 
                 compute_wer: bool = True, 
                 compute_cer: bool = True,
                 compute_bleu: bool = True):
        """Initialize ASR metrics calculator.
        
        Args:
            compute_wer: Whether to compute Word Error Rate
            compute_cer: Whether to compute Character Error Rate
            compute_bleu: Whether to compute BLEU score
        """
        self.compute_wer = compute_wer
        self.compute_cer = compute_cer
        self.compute_bleu = compute_bleu
        self.bleu_metric = evaluate.load("bleu") if compute_bleu else None
        
    def evaluate(self, 
                references: List[str], 
                predictions: List[str],
                language: Optional[str] = None) -> Dict[str, float]:
        """Evaluate ASR predictions against reference transcriptions.
        
        Args:
            references: List of reference transcriptions
            predictions: List of model predictions
            language: Optional language code for language-specific processing
            
        Returns:
            Dictionary of metrics
        """
        results = {}
        
        # Word Error Rate
        if self.compute_wer:
            transformation = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation()
            ])
            wer = jiwer.wer(
                references, 
                predictions,
                truth_transform=transformation,
                hypothesis_transform=transformation
            )
            results["wer"] = wer
        
        # Character Error Rate
        if self.compute_cer:
            cer = jiwer.cer(references, predictions)
            results["cer"] = cer
            
        # BLEU Score
        if self.compute_bleu:
            bleu_score = self.bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            results["bleu"] = bleu_score["bleu"]
            
        return results 