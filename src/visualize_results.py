"""
Script to visualize self-consistency experiment results.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional


def load_results(results_file: str) -> Dict[str, Any]:
    """
    Load results from a JSON file.
    
    Args:
        results_file: Path to the results file
        
    Returns:
        Dictionary with results
    """
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(results: Dict[str, Any], dataset_name: str, model_name: str, output_dir: str):
    """
    Plot accuracy comparison between greedy CoT and self-consistency.
    
    Args:
        results: Results dictionary
        dataset_name: Name of the dataset
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    greedy_acc = results["greedy_accuracy"]
    sc_acc = results["sc_accuracy"]
    improvement = results["improvement"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    methods = ["Greedy CoT", "Self-Consistency"]
    accuracies = [greedy_acc, sc_acc]
    bars = ax.bar(methods, accuracies, color=["#1f77b4", "#ff7f0e"])
    
    # Add labels and title
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy Comparison on {dataset_name} with {model_name}")
    
    # Add text on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}", ha="center", va="bottom")
    
    # Add improvement text
    ax.text(1.5, min(greedy_acc, sc_acc) / 2, 
            f"Improvement: {improvement:.4f}", 
            ha="center", va="center", 
            bbox=dict(facecolor="white", alpha=0.8))
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_{model_name}_accuracy.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved accuracy plot to {output_file}")


def plot_answer_distribution(results: Dict[str, Any], dataset_name: str, model_name: str, output_dir: str):
    """
    Plot distribution of answers for a few examples.
    
    Args:
        results: Results dictionary
        dataset_name: Name of the dataset
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    # Get a few examples (up to 5)
    examples = results["results"][:5]
    
    for i, example in enumerate(examples):
        # Get answer counts
        answer_counts = example["sc_counts"]
        if not answer_counts:
            continue
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame(
            {"Answer": list(answer_counts.keys()), "Count": list(answer_counts.values())}
        )
        df = df.sort_values("Count", ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars
        bars = ax.bar(df["Answer"], df["Count"], color="#1f77b4")
        
        # Add labels and title
        ax.set_xlabel("Answer")
        ax.set_ylabel("Count")
        ax.set_title(f"Answer Distribution for Example {i+1}")
        
        # Add ground truth and majority answer
        gt = example["ground_truth"]
        majority = example["sc_majority"]
        
        # Add text on plot
        ax.text(0.02, 0.95, f"Question: {example['prompt'].split('Q:')[-1].split('A:')[0].strip()}", 
                transform=ax.transAxes, va="top", ha="left", 
                bbox=dict(facecolor="white", alpha=0.8), wrap=True)
        ax.text(0.02, 0.85, f"Ground Truth: {gt}", 
                transform=ax.transAxes, va="top", ha="left", 
                bbox=dict(facecolor="white", alpha=0.8))
        ax.text(0.02, 0.78, f"Majority Answer: {majority}", 
                transform=ax.transAxes, va="top", ha="left", 
                bbox=dict(facecolor="white", alpha=0.8))
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{dataset_name}_{model_name}_example{i+1}_distribution.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Saved answer distribution plot for example {i+1} to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize self-consistency results")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to the results file")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name")
    parser.add_argument("--output_dir", type=str, default="../results/plots",
                        help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    
    # Plot accuracy comparison
    plot_accuracy_comparison(results, args.dataset, args.model, args.output_dir)
    
    # Plot answer distribution
    plot_answer_distribution(results, args.dataset, args.model, args.output_dir)