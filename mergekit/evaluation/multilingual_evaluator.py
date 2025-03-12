"""Multilingual evaluation for Whisper ASR models."""

import pandas as pd
import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

from mergekit.evaluation.whisper_evaluator import WhisperEvaluator


class MultilingualWhisperEvaluator:
    """Evaluator for multilingual Whisper ASR models."""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "cuda",
                 batch_size: int = 8):
        """Initialize a multilingual Whisper evaluator.
        
        Args:
            model_path: Path to the Whisper model
            device: Device to run inference on ('cuda' or 'cpu')
            batch_size: Batch size for inference
        """
        self.evaluator = WhisperEvaluator(
            model_path=model_path, 
            device=device,
            batch_size=batch_size
        )
        
    def evaluate_multilingual(self,
                             datasets: Dict[str, str],
                             max_samples_per_language: int = 100,
                             split: str = "test") -> pd.DataFrame:
        """Evaluate on multiple languages using different datasets.
        
        Args:
            datasets: Dictionary mapping language codes to dataset names
            max_samples_per_language: Maximum samples to evaluate per language
            split: Dataset split to use
            
        Returns:
            DataFrame with results per language
        """
        results = []
        
        for language, dataset in datasets.items():
            logging.info(f"Evaluating language: {language} using dataset: {dataset}")
            metrics = self.evaluator.evaluate_dataset(
                dataset_name=dataset,
                split=split,
                language=language,
                max_samples=max_samples_per_language
            )
            
            metrics["language"] = language
            results.append(metrics)
            
            # Log current results
            logging.info(f"Results for {language}:")
            for metric, value in metrics.items():
                if metric != "language":
                    logging.info(f"  {metric.upper()}: {value:.4f}")
            
        return pd.DataFrame(results)
    
    def evaluate_common_dataset(self,
                              dataset_name: str,
                              languages: List[str],
                              max_samples_per_language: int = 100,
                              split: str = "test") -> pd.DataFrame:
        """Evaluate on multiple languages using a common dataset.
        
        Args:
            dataset_name: Name of the multilingual dataset
            languages: List of language codes to evaluate
            max_samples_per_language: Maximum samples to evaluate per language
            split: Dataset split to use
            
        Returns:
            DataFrame with results per language
        """
        results = []
        
        for language in languages:
            logging.info(f"Evaluating language: {language} on dataset: {dataset_name}")
            metrics = self.evaluator.evaluate_dataset(
                dataset_name=dataset_name,
                split=f"{split}.{language}" if split else language,
                language=language,
                max_samples=max_samples_per_language
            )
            
            metrics["language"] = language
            results.append(metrics)
            
        return pd.DataFrame(results) 