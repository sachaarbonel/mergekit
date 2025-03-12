#!/usr/bin/env python3
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

"""Command-line tool for comparing multiple Whisper ASR models."""

import click
import json
import os
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

from mergekit.evaluation.whisper_evaluator import WhisperEvaluator
from mergekit.options import MergeOptions, add_merge_options


def run_comparison(
    models: List[str],
    model_names: Optional[List[str]],
    dataset: str,
    split: str = "test",
    language: Optional[str] = None,
    max_samples: int = 100,
    output_file: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 8,
):
    """Compare multiple Whisper models on the same dataset.
    
    Args:
        models: List of paths to Whisper models
        model_names: Optional list of names for the models
        dataset: HuggingFace dataset name
        split: Dataset split to use
        language: Language code for transcription
        max_samples: Maximum number of samples to evaluate
        output_file: Path to save comparison results as CSV
        device: Device to run inference on ('cuda' or 'cpu')
        batch_size: Batch size for inference
        
    Returns:
        DataFrame with comparison results
    """
    # Use model paths as names if not provided
    if not model_names:
        model_names = [os.path.basename(model) for model in models]
    elif len(model_names) != len(models):
        raise ValueError("Number of model names must match number of models")
    
    results = []
    
    # Evaluate each model
    for i, (model, name) in enumerate(zip(models, model_names)):
        logging.info(f"Evaluating model {i+1}/{len(models)}: {name}")
        
        evaluator = WhisperEvaluator(
            model_path=model,
            device=device,
            batch_size=batch_size
        )
        
        model_results = evaluator.evaluate_dataset(
            dataset_name=dataset,
            split=split,
            language=language,
            max_samples=max_samples
        )
        
        model_results["model"] = name
        results.append(model_results)
        
        # Print current model results
        print(f"Results for {name}:")
        for metric, value in model_results.items():
            if metric != "model":
                print(f"  {metric.upper()}: {value:.4f}")
    
    # Create comparison dataframe
    df = pd.DataFrame(results)
    
    # Print comparison table
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n--- Model Comparison ---")
    print(df.set_index("model"))
    
    # Save results if requested
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nComparison saved to {output_file}")
        
    return df


@click.command("mergekit-compare-whisper", cls=click.core.Command)
@click.option(
    "--models",
    required=True,
    multiple=True,
    help="Paths to Whisper models to compare (can specify multiple)",
)
@click.option(
    "--model-names",
    multiple=True,
    help="Names for the models (order must match --models)",
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
    help="Language code for transcription",
)
@click.option(
    "--max-samples",
    type=int,
    default=100,
    help="Maximum number of samples to evaluate",
)
@click.option(
    "--output-file",
    help="Path to save comparison results as CSV",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for inference",
)
@add_merge_options
def main(
    models: List[str],
    model_names: List[str],
    dataset: str,
    split: str,
    language: Optional[str],
    max_samples: int,
    output_file: Optional[str],
    batch_size: int,
    merge_options: MergeOptions,
):
    """Compare multiple Whisper models on ASR metrics.
    
    This tool evaluates multiple Whisper models on the same dataset
    and generates a comparative report of their performance.
    """
    # Apply global options
    merge_options.apply_global_options()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run comparison
    run_comparison(
        models=models,
        model_names=model_names,
        dataset=dataset,
        split=split,
        language=language,
        max_samples=max_samples,
        output_file=output_file,
        device="cuda" if merge_options.cuda else "cpu",
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main() 