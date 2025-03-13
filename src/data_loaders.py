"""
Dataset loaders for self-consistency evaluation.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datasets import load_dataset


def load_gsm8k(split: str = "test", sample_size: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load the GSM8K dataset.
    
    Args:
        split: Dataset split to load
        sample_size: Number of examples to sample (None for all)
        
    Returns:
        Tuple of (questions, answers)
    """
    dataset = load_dataset("gsm8k", "main", split=split)
    
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    
    questions = [item["question"] for item in dataset]
    
    # Extract the answer from the answer field (format: "####: <answer>")
    answers = []
    for item in dataset:
        answer_text = item["answer"]
        # Extract the final answer after the last "####"
        parts = answer_text.split("####")
        if len(parts) > 1:
            answers.append(parts[-1].strip())
        else:
            answers.append(answer_text.strip())
    
    return questions, answers


def load_svamp(split: str = "test", sample_size: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load the SVAMP dataset.
    
    Args:
        split: Dataset split to load
        sample_size: Number of examples to sample (None for all)
        
    Returns:
        Tuple of (questions, answers)
    """
    dataset = load_dataset("svamp", split=split)
    
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    
    questions = [item["Body"] + " " + item["Question"] for item in dataset]
    answers = [str(item["Answer"]) for item in dataset]
    
    return questions, answers


def load_aqua(sample_size: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load the AQuA dataset.
    
    Args:
        sample_size: Number of examples to sample (None for all)
        
    Returns:
        Tuple of (questions, answers)
    """
    dataset = load_dataset("aqua_rat", "raw", split="test")
    
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    
    questions = [item["question"] for item in dataset]
    answers = [item["correct"] for item in dataset]
    
    return questions, answers


def load_strategyqa(sample_size: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load the StrategyQA dataset.
    
    Args:
        sample_size: Number of examples to sample (None for all)
        
    Returns:
        Tuple of (questions, answers)
    """
    dataset = load_dataset("metaeval/strategy-qa", split="train")
    
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    
    questions = [item["question"] for item in dataset]
    answers = [str(item["answer"]).lower() for item in dataset]
    
    return questions, answers


def load_arc_challenge(sample_size: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load the ARC-Challenge dataset.
    
    Args:
        sample_size: Number of examples to sample (None for all)
        
    Returns:
        Tuple of (questions, answers)
    """
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    
    questions = []
    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        # Format is different for ARC dataset
        choices_text = "\n".join([f"({choices['label'][i]}) {choices['text'][i]}" for i in range(len(choices['label']))])
        questions.append(f"{question}\n{choices_text}")
    
    answers = [item["answerKey"] for item in dataset]
    
    return questions, answers


def load_commonsenseqa(sample_size: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load the CommonsenseQA dataset.
    
    Args:
        sample_size: Number of examples to sample (None for all)
        
    Returns:
        Tuple of (questions, answers)
    """
    dataset = load_dataset("commonsense_qa", split="validation")
    
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    
    questions = []
    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        # Format is different for CommonsenseQA dataset
        choices_text = "\n".join([f"({choice['label']}) {choice['text']}" for choice in choices])
        questions.append(f"{question}\n{choices_text}")
    
    answers = [item["answerKey"] for item in dataset]
    
    return questions, answers