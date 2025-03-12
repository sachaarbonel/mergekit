#!/usr/bin/env python3
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

"""Command-line tool for evaluating Whisper ASR models."""

import click
import json
import os
import logging
from typing import List, Optional

from mergekit.evaluation.whisper_evaluator import WhisperEvaluator
from mergekit.options import MergeOptions, add_merge_options


def run_evaluation(
    model: str,
    dataset: str,
    split: str = "test",
    language: Optional[str] = None,
    audio_column: str = "audio",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    output_file: Optional[str] = None,
    batch_size: int = 8,
    device: str = "cuda",
):
    """Run evaluation on a Whisper model.
    
    Args:
        model: Path to the Whisper model
        dataset: HuggingFace dataset name
        split: Dataset split to use
        language: Language code for transcription
        audio_column: Column name containing audio data
        text_column: Column name containing reference text
        max_samples: Maximum number of samples to evaluate
        output_file: Path to save evaluation results as JSON
        batch_size: Batch size for inference
        device: Device to run inference on ('cuda' or 'cpu')
    """
    # Initialize evaluator
    evaluator = WhisperEvaluator(
        model_path=model,
        device=device,
        batch_size=batch_size
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataset_name=dataset,
        split=split,
        language=language,
        audio_column=audio_column,
        text_column=text_column,
        max_samples=max_samples
    )
    
    # Print results
    print("\n--- Evaluation Results ---")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Save results if requested
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


@click.command("mergekit-evaluate-whisper", cls=click.core.Command)
@click.option(
    "--model",
    required=True,
    help="Path to the Whisper model to evaluate",
)
@click.option(
    "--dataset",
    required=True,
    help="HuggingFace dataset name for evaluation",
)
@click.option(
    "--split",
    default="test",
    help="Dataset split to use",
)
@click.option(
    "--language",
    help="Language code for transcription (e.g., 'en', 'fr')",
)
@click.option(
    "--audio-column",
    default="audio",
    help="Column name containing audio data",
)
@click.option(
    "--text-column",
    default="text",
    help="Column name containing reference text",
)
@click.option(
    "--max-samples",
    type=int,
    help="Maximum number of samples to evaluate",
)
@click.option(
    "--output-file",
    help="Path to save evaluation results as JSON",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for inference",
)
@add_merge_options
def main(
    model: str,
    dataset: str,
    split: str,
    language: Optional[str],
    audio_column: str,
    text_column: str,
    max_samples: Optional[int],
    output_file: Optional[str],
    batch_size: int,
    merge_options: MergeOptions,
):
    """Evaluate a Whisper model on ASR metrics.
    
    This tool runs a Whisper model on a speech dataset and calculates
    Word Error Rate (WER), Character Error Rate (CER), and BLEU score.
    """
    # Apply global options
    merge_options.apply_global_options()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run evaluation
    run_evaluation(
        model=model,
        dataset=dataset,
        split=split,
        language=language,
        audio_column=audio_column,
        text_column=text_column,
        max_samples=max_samples,
        output_file=output_file,
        batch_size=batch_size,
        device="cuda" if merge_options.cuda else "cpu",
    )


if __name__ == "__main__":
    main() 