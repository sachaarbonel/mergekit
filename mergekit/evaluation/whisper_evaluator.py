"""Evaluator for Whisper ASR models."""

import os
import torch
import pandas as pd
from typing import Dict, List, Optional, Union
import logging

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
from tqdm import tqdm

from mergekit.evaluation.asr_metrics import ASRMetrics


class WhisperEvaluator:
    """Evaluator for Whisper ASR models."""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8):
        """Initialize a Whisper model evaluator.
        
        Args:
            model_path: Path to the Whisper model
            device: Device to run inference on ('cuda' or 'cpu')
            batch_size: Batch size for inference
        """
        logging.info(f"Loading Whisper model from {model_path}")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.device = device
        self.batch_size = batch_size
        self.metrics = ASRMetrics()
        
    def transcribe(self, audio_paths: List[str], language: Optional[str] = None) -> List[str]:
        """Transcribe a list of audio files.
        
        Args:
            audio_paths: List of paths to audio files
            language: Optional language code for transcription
            
        Returns:
            List of transcriptions
        """
        results = []
        
        for i in range(0, len(audio_paths), self.batch_size):
            batch = audio_paths[i:i+self.batch_size]
            batch_audio = [{"array": self._load_audio(path), "sampling_rate": 16000} for path in batch]
            
            inputs = self.processor(
                batch_audio, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            forced_decoder_ids = None
            if language:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
                
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_features,
                    forced_decoder_ids=forced_decoder_ids
                )
                
            transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            results.extend(transcriptions)
            
        return results
    
    def _load_audio(self, path):
        """Load audio file.
        
        Args:
            path: Path to audio file
            
        Returns:
            Audio waveform as numpy array
        """
        import librosa
        waveform, _ = librosa.load(path, sr=16000)
        return waveform
        
    def evaluate_dataset(self, 
                        dataset_name: str, 
                        split: str = "test",
                        language: Optional[str] = None,
                        audio_column: str = "audio",
                        text_column: str = "text",
                        max_samples: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model on a huggingface dataset.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            split: Dataset split to use
            language: Optional language code for transcription
            audio_column: Column name containing audio data
            text_column: Column name containing reference text
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        logging.info(f"Loading dataset {dataset_name} (split: {split})")
        dataset = load_dataset(dataset_name, split=split)
        if "audio" in dataset.features and isinstance(dataset.features["audio"], Audio):
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16000))
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            
        references = dataset[text_column]
        
        # Process audio and get transcriptions
        logging.info(f"Processing {len(dataset)} audio samples")
        audios = [sample[audio_column]["array"] for sample in dataset]
        inputs = self.processor(
            audios, 
            sampling_rate=16000,
            return_tensors="pt", 
            padding=True
        )
        
        predictions = []
        for i in tqdm(range(0, len(dataset), self.batch_size)):
            batch_inputs = {
                k: v[i:i+self.batch_size].to(self.device) 
                for k, v in inputs.items()
            }
            
            forced_decoder_ids = None
            if language:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
                
            with torch.no_grad():
                generated_ids = self.model.generate(
                    batch_inputs.input_features,
                    forced_decoder_ids=forced_decoder_ids
                )
                
            batch_transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(batch_transcriptions)
        
        # Calculate metrics
        logging.info("Calculating evaluation metrics")
        return self.metrics.evaluate(references, predictions, language=language) 