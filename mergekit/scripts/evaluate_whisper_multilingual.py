#!/usr/bin/env python3
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

"""Command-line tool for evaluating Whisper models across multiple languages."""

import click
import json
import os
import pandas as pd
import logging
from typing import List, Dict, Optional

from mergekit.evaluation.multilingual_evaluator import MultilingualWhisperEvaluator
from mergekit.options import MergeOptions, add_merge_options


def run_multilingual_evaluation(
    model: str,
    languages: List[str],
    dataset: str,
    split: str = "test",
    max_samples: int = 100,
    output_file: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 8,
):
    """Evaluate a Whisper model across multiple languages.
    
    Args:
        model: Path to the Whisper model
        languages: List of language codes to evaluate
        dataset: HuggingFace dataset name (should support multiple languages)
        split: Dataset split to use
        max_samples: Maximum samples per language
        output_file: Path to save results as CSV
        device: Device to run inference on ('cuda' or 'cpu')
        batch_size: Batch size for inference
        
    Returns:
        DataFrame with results per language
    """
    evaluator = MultilingualWhisperEvaluator(
        model_path=model,
        device=device,
        batch_size=batch_size
    )
    
    results = evaluator.evaluate_common_dataset(
        dataset_name=dataset,
        languages=languages,
        max_samples_per_language=max_samples,
        split=split
    )
    
    # Print results
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n--- Multilingual Evaluation Results ---")
    print(results.set_index("language"))
    
    # Calculate average metrics
    avg_metrics = {}
    for col in results.columns:
        if col != "language":
            avg_metrics[f"avg_{col}"] = results[col].mean()
    
    print("\n--- Average Metrics ---")
    for metric, value in avg_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Save results if requested
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        results.to_csv(output_file, index=False)
        
        # Save summary to a separate file
        summary_file = os.path.splitext(output_file)[0] + "_summary.json"
        with open(summary_file, "w") as f:
            json.dump(avg_metrics, f, indent=2)
            
        print(f"\nResults saved to {output_file}")
        print(f"Summary saved to {summary_file}")
    
    return results


@click.command("mergekit-evaluate-whisper-multilingual", cls=click.core.Command)
@click.option(
    "--model",
    required=True,
    help="Path to the Whisper model to evaluate",
)
@click.option(
    "--languages",
    required=True,
    multiple=True,
    help="Language codes to evaluate (can specify multiple)",
)
@click.option(
    "--dataset",
    required=True,
    help="HuggingFace dataset name (should support multiple languages)",
)
@click.option(
    "--split",
    default="test",
    help="Dataset split to use",
)
@click.option(
    "--max-samples",
    type=int,
    default=100,
    help="Maximum samples per language",
)
@click.option(
    "--output-file",
    help="Path to save results as CSV",
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
    languages: List[str],
    dataset: str,
    split: str,
    max_samples: int,
    output_file: Optional[str],
    batch_size: int,
    merge_options: MergeOptions,
):
    """Evaluate a Whisper model across multiple languages.
    
    This tool evaluates a Whisper model on multiple languages using
    a common dataset and generates a comparative report of performance
    across languages.
    """
    # Apply global options
    merge_options.apply_global_options()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run evaluation
    run_multilingual_evaluation(
        model=model,
        languages=languages,
        dataset=dataset,
        split=split,
        max_samples=max_samples,
        output_file=output_file,
        device="cuda" if merge_options.cuda else "cpu",
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main() 