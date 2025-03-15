"""
Script to run self-consistency experiments.
"""

import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

from self_consistency import SelfConsistency
from data_loaders import (
    load_gsm8k, 
    load_svamp, 
    load_aqua, 
    load_strategyqa, 
    load_arc_challenge,
    load_commonsenseqa
)
from prompts import (
    GSM8K_PROMPT, 
    SVAMP_PROMPT, 
    AQUA_PROMPT, 
    STRATEGYQA_PROMPT, 
    ARC_CHALLENGE_PROMPT,
    COMMONSENSEQA_PROMPT
)


def format_prompts(questions: List[str], prompt_template: str) -> List[str]:
    """
    Format questions using the prompt template.
    
    Args:
        questions: List of questions
        prompt_template: Prompt template with {question} placeholder
        
    Returns:
        List of formatted prompts
    """
    return [prompt_template.format(question=q) for q in questions]


def run_experiment(
    dataset_name: str,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    device: str = "cpu",
    sample_size: int = 5,
    num_samples: int = 5,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    max_new_tokens: int = 2048,
    output_dir: str = "../results",
    answer_pattern: Optional[str] = None
):
    """
    Run self-consistency experiment on a dataset.
    
    Args:
        dataset_name: Name of the dataset to use
        model_name: Name of the model to use
        device: Device to run the model on
        sample_size: Number of examples to sample from the dataset
        num_samples: Number of reasoning paths to sample
        temperature: Temperature for sampling
        top_k: Top-k for sampling
        max_new_tokens: Maximum number of tokens to generate
        output_dir: Directory to save results
        answer_pattern: Regex pattern to extract the answer
    """
    # Set default answer pattern based on dataset
    if answer_pattern is None:
        if dataset_name in ["strategyqa"]:
            answer_pattern = r"The answer is (yes|no)\.?"
        elif dataset_name in ["arc_challenge", "commonsenseqa"]:
            answer_pattern = r"The answer is \(([A-E])\)\.?"
        else:
            answer_pattern = r"The answer is (.*?)\.?"
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == "gsm8k":
        questions, answers = load_gsm8k(sample_size=sample_size)
        prompt_template = GSM8K_PROMPT
    elif dataset_name == "svamp":
        questions, answers = load_svamp(sample_size=sample_size)
        prompt_template = SVAMP_PROMPT
    elif dataset_name == "aqua":
        questions, answers = load_aqua(sample_size=sample_size)
        prompt_template = AQUA_PROMPT
    elif dataset_name == "strategyqa":
        questions, answers = load_strategyqa(sample_size=sample_size)
        prompt_template = STRATEGYQA_PROMPT
    elif dataset_name == "arc_challenge":
        questions, answers = load_arc_challenge(sample_size=sample_size)
        prompt_template = ARC_CHALLENGE_PROMPT
    elif dataset_name == "commonsenseqa":
        questions, answers = load_commonsenseqa(sample_size=sample_size)
        prompt_template = COMMONSENSEQA_PROMPT
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Format prompts
    prompts = format_prompts(questions, prompt_template)
    
    # Initialize self-consistency model
    print(f"Initializing model {model_name}...")
    sc = SelfConsistency(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        num_samples=num_samples,
        answer_pattern=answer_pattern
    )
    
    # Run evaluation
    print(f"Running evaluation with {num_samples} samples per question...")
    results = sc.evaluate_self_consistency(prompts, answers)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, 
        f"{dataset_name}_{model_name.split('/')[-1]}_{num_samples}samples.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nResults for {dataset_name} with {model_name}:")
    print(f"Greedy CoT Accuracy: {results['greedy_accuracy']:.4f}")
    print(f"Self-Consistency Accuracy: {results['sc_accuracy']:.4f}")
    print(f"Improvement: {results['improvement']:.4f}")
    
    # Print detailed results for the first few examples
    print("\nDetailed results for the first few examples:")
    for i, result in enumerate(results['results'][:3]):
        print(f"\nExample {i+1}:")
        print(f"Question: {result['prompt'].split('Q:')[-1].split('A:')[0].strip()}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Greedy Answer: {result['greedy_answer']}")
        print(f"Self-Consistency Answers: {result['sc_answers']}")
        print(f"Self-Consistency Majority: {result['sc_majority']}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run self-consistency experiments")
    parser.add_argument("--dataset", type=str, default="gsm8k", 
                        choices=["gsm8k", "svamp", "aqua", "strategyqa", "arc_challenge", "commonsenseqa"],
                        help="Dataset to use")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model to use")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run the model on")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of examples to sample from the dataset")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of reasoning paths to sample")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for sampling")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--output_dir", type=str, default="../results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    run_experiment(
        dataset_name=args.dataset,
        model_name=args.model,
        device=args.device,
        sample_size=args.sample_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir
    )