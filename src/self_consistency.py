"""
Implementation of Self-Consistency method for Chain-of-Thought reasoning.

Based on the paper:
"Self-Consistency Improves Chain of Thought Reasoning in Language Models"
by Xuezhi Wang et al.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re


class SelfConsistency:
    """
    Implementation of Self-Consistency method for Chain-of-Thought reasoning.
    """
    
    def __init__(
        self, 
        model_name: str = "google/flan-t5-small",
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 40,
        num_samples: int = 5,
        answer_pattern: str = r"The answer is (.*)\."
    ):
        """
        Initialize the Self-Consistency model.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_k: Top-k for sampling
            num_samples: Number of reasoning paths to sample
            answer_pattern: Regex pattern to extract the answer from the generated text
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.num_samples = num_samples
        self.answer_pattern = answer_pattern
        
        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def extract_answer(self, text: str) -> str:
        """
        Extract the answer from the generated text using the answer pattern.
        
        Args:
            text: Generated text
            
        Returns:
            Extracted answer or empty string if no answer found
        """
        match = re.search(self.answer_pattern, text)
        if match:
            return match.group(1).strip()
        return ""
    
    def generate_cot_sample(self, prompt: str) -> Tuple[str, str]:
        """
        Generate a single chain-of-thought reasoning sample.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (full_generation, extracted_answer)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self.extract_answer(generation)
        
        return generation, answer
    
    def generate_cot_greedy(self, prompt: str) -> Tuple[str, str]:
        """
        Generate a greedy chain-of-thought reasoning (baseline).
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (full_generation, extracted_answer)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self.extract_answer(generation)
        
        return generation, answer
    
    def generate_self_consistency(self, prompt: str) -> Dict[str, Any]:
        """
        Generate multiple reasoning paths and apply self-consistency.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dictionary with results including:
            - samples: List of generated reasoning paths
            - answers: List of extracted answers
            - majority_answer: Most common answer
            - answer_counts: Counter of answer frequencies
        """
        samples = []
        answers = []
        
        # Generate multiple samples
        for _ in range(self.num_samples):
            generation, answer = self.generate_cot_sample(prompt)
            samples.append(generation)
            answers.append(answer)
        
        # Count answer frequencies
        answer_counts = Counter(answers)
        
        # Get the majority answer (most consistent)
        if answer_counts:
            majority_answer = answer_counts.most_common(1)[0][0]
        else:
            majority_answer = ""
        
        return {
            "samples": samples,
            "answers": answers,
            "majority_answer": majority_answer,
            "answer_counts": answer_counts
        }
    
    def evaluate_self_consistency(
        self, 
        prompts: List[str], 
        ground_truth: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate self-consistency on a set of prompts.
        
        Args:
            prompts: List of input prompts
            ground_truth: List of ground truth answers
            
        Returns:
            Dictionary with evaluation results
        """
        results = []
        correct_greedy = 0
        correct_sc = 0
        
        for i, (prompt, gt) in enumerate(zip(prompts, ground_truth)):
            # Generate greedy baseline
            greedy_gen, greedy_answer = self.generate_cot_greedy(prompt)
            
            # Generate self-consistency result
            sc_result = self.generate_self_consistency(prompt)
            
            # Check if answers are correct
            greedy_correct = greedy_answer.strip() == gt.strip()
            sc_correct = sc_result["majority_answer"].strip() == gt.strip()
            
            if greedy_correct:
                correct_greedy += 1
            
            if sc_correct:
                correct_sc += 1
            
            results.append({
                "prompt": prompt,
                "ground_truth": gt,
                "greedy_generation": greedy_gen,
                "greedy_answer": greedy_answer,
                "greedy_correct": greedy_correct,
                "sc_samples": sc_result["samples"],
                "sc_answers": sc_result["answers"],
                "sc_majority": sc_result["majority_answer"],
                "sc_counts": sc_result["answer_counts"],
                "sc_correct": sc_correct
            })
        
        # Calculate accuracy
        greedy_accuracy = correct_greedy / len(prompts) if prompts else 0
        sc_accuracy = correct_sc / len(prompts) if prompts else 0
        
        return {
            "results": results,
            "greedy_accuracy": greedy_accuracy,
            "sc_accuracy": sc_accuracy,
            "improvement": sc_accuracy - greedy_accuracy
        }