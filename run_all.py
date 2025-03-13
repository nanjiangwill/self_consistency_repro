"""
Script to run all self-consistency experiments.
"""

import os
import argparse
import subprocess
from tqdm import tqdm

# Define datasets and their configurations
DATASETS = [
    "gsm8k",
    "svamp",
    "aqua",
    "strategyqa",
    "arc_challenge",
    "commonsenseqa"
]

def run_all_experiments(
    model_name: str = "google/flan-t5-small",
    device: str = "cpu",
    sample_size: int = 5,
    num_samples: int = 5,
    output_dir: str = "results"
):
    """
    Run all self-consistency experiments.
    
    Args:
        model_name: Name of the model to use
        device: Device to run the model on
        sample_size: Number of examples to sample from each dataset
        num_samples: Number of reasoning paths to sample
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiments for each dataset
    for dataset in tqdm(DATASETS, desc="Datasets"):
        print(f"\n{'='*50}")
        print(f"Running experiment for {dataset}")
        print(f"{'='*50}\n")
        
        cmd = [
            "python", "src/run_experiment.py",
            "--dataset", dataset,
            "--model", model_name,
            "--device", device,
            "--sample_size", str(sample_size),
            "--num_samples", str(num_samples),
            "--output_dir", output_dir
        ]
        
        subprocess.run(cmd)
    
    print("\nAll experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all self-consistency experiments")
    parser.add_argument("--model", type=str, default="google/flan-t5-small",
                        help="Model to use")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run the model on")
    parser.add_argument("--sample_size", type=int, default=5,
                        help="Number of examples to sample from each dataset")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of reasoning paths to sample")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    run_all_experiments(
        model_name=args.model,
        device=args.device,
        sample_size=args.sample_size,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )