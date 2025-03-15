# Self-Consistency Reproduction

This repository contains an implementation of the self-consistency method for improving chain-of-thought reasoning in language models, as described in the paper:

**"Self-Consistency Improves Chain of Thought Reasoning in Language Models"**  
by Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou.  
Published at ICLR 2023.

## Overview

Self-consistency is a decoding strategy that improves chain-of-thought (CoT) reasoning in language models. The method consists of three main steps:

1. Prompt a language model using chain-of-thought prompting
2. Sample multiple reasoning paths from the model (instead of just taking the greedy path)
3. Marginalize out the reasoning paths by selecting the most consistent answer among the generated answers

The key insight is that complex reasoning problems typically have multiple valid reasoning paths that lead to the same correct answer. By sampling diverse reasoning paths and taking the most common answer, we can improve the model's reasoning performance.

## Repository Structure

```
.
├── src/
│   ├── self_consistency.py     # Main implementation of self-consistency
│   ├── datasets.py             # Dataset loaders
│   ├── prompts.py              # Chain-of-thought prompts
│   ├── run_experiment.py       # Script to run experiments
│   └── visualize_results.py    # Script to visualize results
├── data/                       # Directory for datasets
├── results/                    # Directory for results
├── run_all.py                  # Script to run all experiments
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running a Single Experiment

To run an experiment on a single dataset:

```bash
python src/run_experiment.py --dataset gsm8k --model meta-llama/Llama-3.2-3B-Instruct --sample_size 5 --num_samples 5
```

Arguments:
- `--dataset`: Dataset to use (choices: gsm8k, svamp, aqua, strategyqa, arc_challenge, commonsenseqa)
- `--model`: Model to use (default: meta-llama/Llama-3.2-3B-Instruct)
- `--device`: Device to run the model on (default: cpu)
- `--sample_size`: Number of examples to sample from the dataset (default: 5)
- `--num_samples`: Number of reasoning paths to sample (default: 5)
- `--temperature`: Temperature for sampling (default: 0.7)
- `--top_k`: Top-k for sampling (default: 40)
- `--max_new_tokens`: Maximum number of tokens to generate (default: 512)
- `--output_dir`: Directory to save results (default: ../results)

### Running All Experiments

To run experiments on all datasets:

```bash
python run_all.py --model meta-llama/Llama-3.2-3B-Instruct --sample_size 5 --num_samples 5
```

### Visualizing Results

To visualize the results:

```bash
python src/visualize_results.py --results_file results/gsm8k_flan-t5-small_5samples.json --dataset gsm8k --model flan-t5-small
```

## Datasets

The implementation supports the following datasets:

- **GSM8K**: Grade School Math problems
- **SVAMP**: Challenge dataset for math word problems
- **AQuA**: Algebra word problems
- **StrategyQA**: Questions requiring multi-step reasoning
- **ARC-Challenge**: Science questions
- **CommonsenseQA**: Commonsense reasoning questions

## Results

The original paper reported significant improvements across all datasets:

- GSM8K: +17.9% absolute accuracy gains
- SVAMP: +11.0%
- AQuA: +12.2%
- StrategyQA: +6.4%
- ARC-challenge: +3.9%

Our reproduction aims to verify these results using the meta-llama/Llama-3.2-3B-Instruct model.

## Citation

```
@inproceedings{
    wang2023selfconsistency,
    title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
    author={Xuezhi Wang and Jason Wei and Dale Schuurmans and Quoc Le and Ed Chi and Sharan Narang and Aakanksha Chowdhery and Denny Zhou},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=1PL1NIMMrw}
}
```