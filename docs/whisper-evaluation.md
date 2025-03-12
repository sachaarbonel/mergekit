# Whisper ASR Evaluation Tools

MergeKit provides a set of tools for evaluating Whisper Automatic Speech Recognition (ASR) models, particularly useful for assessing the quality of merged models.

## Installation

To use the ASR evaluation tools, you need to install the required dependencies:

```bash
# If using pip
pip install "mergekit[whisper]"

# If using Poetry
poetry install --with whisper
```

## Available Metrics

The evaluation tools calculate the following metrics:

- **Word Error Rate (WER)**: The percentage of words that were incorrectly predicted. Lower is better.
- **Character Error Rate (CER)**: The percentage of characters that were incorrectly predicted. Lower is better.
- **BLEU Score**: A measure of the similarity between the predicted and reference transcriptions. Higher is better.

## Command-Line Tools

### Basic Evaluation

Evaluate a single Whisper model on a dataset:

```bash
mergekit-evaluate-whisper \
  --model path/to/whisper/model \
  --dataset mozilla-foundation/common_voice_11_0 \
  --split test \
  --language en \
  --max-samples 100 \
  --output-file results.json
```

### Comparing Multiple Models

Compare multiple Whisper models on the same dataset:

```bash
mergekit-compare-whisper \
  --models path/to/model1 path/to/model2 path/to/merged_model \
  --model-names "Base Model" "Fine-tuned Model" "Merged Model" \
  --dataset mozilla-foundation/common_voice_11_0 \
  --split test \
  --language en \
  --max-samples 100 \
  --output-file comparison.csv
```

### Multilingual Evaluation

Evaluate a Whisper model across multiple languages:

```bash
mergekit-evaluate-whisper-multilingual \
  --model path/to/whisper/model \
  --languages en fr de es \
  --dataset mozilla-foundation/common_voice_11_0 \
  --split test \
  --max-samples 100 \
  --output-file multilingual_results.csv
```

## Programmatic Usage

You can also use the evaluation tools programmatically in your Python code:

```python
from mergekit.evaluation.whisper_evaluator import WhisperEvaluator
from mergekit.evaluation.multilingual_evaluator import MultilingualWhisperEvaluator

# Evaluate a single model
evaluator = WhisperEvaluator(model_path="path/to/whisper/model")
results = evaluator.evaluate_dataset(
    dataset_name="mozilla-foundation/common_voice_11_0",
    split="test",
    language="en",
    max_samples=100
)
print(results)

# Evaluate across multiple languages
multilingual_evaluator = MultilingualWhisperEvaluator(model_path="path/to/whisper/model")
results_df = multilingual_evaluator.evaluate_common_dataset(
    dataset_name="mozilla-foundation/common_voice_11_0",
    languages=["en", "fr", "de", "es"],
    max_samples_per_language=100
)
print(results_df)
```

## Recommended Datasets

For evaluating Whisper models, we recommend the following datasets:

- **Common Voice**: `mozilla-foundation/common_voice_11_0` - Available in many languages
- **LibriSpeech**: `librispeech_asr` - Clean and noisy English speech
- **VoxPopuli**: `facebook/voxpopuli` - Multilingual European Parliament speeches

## Tips for Effective Evaluation

1. **Use a representative dataset**: Choose a dataset that matches your target use case.
2. **Evaluate on multiple languages**: For multilingual models, test performance across different languages.
3. **Compare with baselines**: Always compare your merged model with the original models.
4. **Consider domain-specific metrics**: For specialized applications, you may want to track domain-specific metrics.
5. **Batch size tuning**: Adjust the batch size based on your GPU memory to optimize evaluation speed. 