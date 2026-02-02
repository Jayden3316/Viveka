"""
Uncertainty quantification methods for supervision signals.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import math
from scipy.stats import entropy


class SemanticEntropyCalculator:
    """
    Calculate semantic entropy from multiple generations.
    
    Based on Kuhn et al. (2023): "Semantic Uncertainty: Linguistic Invariances 
    for Uncertainty Estimation in Natural Language Generation"
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Args:
            similarity_threshold: Threshold for considering answers semantically equivalent
        """
        self.similarity_threshold = similarity_threshold
    
    def compute_semantic_entropy(
        self,
        generations: List[str],
        generation_probs: List[float],
        use_exact_match: bool = True
    ) -> float:
        """
        Compute semantic entropy from generations.
        
        Args:
            generations: List of generated answers
            generation_probs: Probability of each generation
            use_exact_match: If True, use exact string matching for semantic equivalence
        
        Returns:
            Semantic entropy score
        """
        if not generations:
            return 0.0
        
        # Cluster semantically equivalent answers
        clusters = self._cluster_semantically(generations, use_exact_match)
        
        # Compute cluster probabilities
        cluster_probs = []
        for cluster_indices in clusters:
            cluster_prob = sum(generation_probs[i] for i in cluster_indices)
            cluster_probs.append(cluster_prob)
        
        # Normalize
        cluster_probs = np.array(cluster_probs)
        cluster_probs = cluster_probs / cluster_probs.sum()
        
        # Compute entropy
        semantic_entropy = entropy(cluster_probs, base=2)
        
        return semantic_entropy
    
    def _cluster_semantically(
        self,
        generations: List[str],
        use_exact_match: bool
    ) -> List[List[int]]:
        """
        Cluster semantically equivalent generations.
        
        Returns:
            List of clusters, where each cluster is a list of generation indices
        """
        if use_exact_match:
            return self._cluster_exact_match(generations)
        else:
            return self._cluster_similarity(generations)
    
    def _cluster_exact_match(self, generations: List[str]) -> List[List[int]]:
        """Cluster by exact string match (normalized)."""
        normalized = [self._normalize_answer(g) for g in generations]
        
        clusters = {}
        for idx, norm_answer in enumerate(normalized):
            if norm_answer not in clusters:
                clusters[norm_answer] = []
            clusters[norm_answer].append(idx)
        
        return list(clusters.values())
    
    def _cluster_similarity(self, generations: List[str]) -> List[List[int]]:
        """Cluster by semantic similarity (can use embeddings)."""
        # Placeholder for similarity-based clustering
        # In practice, would use sentence embeddings and clustering algorithm
        raise NotImplementedError("Similarity-based clustering not yet implemented")
    
    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer for comparison."""
        return text.lower().strip().rstrip('.!?')


class UncertaintyMetrics:
    """Collection of uncertainty metrics."""
    
    @staticmethod
    def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive entropy from logits.
        
        Args:
            logits: [batch_size, vocab_size]
        
        Returns:
            entropy: [batch_size]
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy
    
    @staticmethod
    def token_probability_variance(
        token_probs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute variance in token probabilities across generation.
        
        Args:
            token_probs: List of probability tensors for each generated token
        
        Returns:
            variance: Scalar or per-sequence variance
        """
        # Stack token probabilities
        stacked = torch.stack(token_probs, dim=0)  # [seq_len, batch, vocab]
        
        # Compute variance across sequence
        variance = stacked.var(dim=0).mean(dim=-1)  # [batch]
        
        return variance
    
    @staticmethod
    def max_probability(logits: torch.Tensor) -> torch.Tensor:
        """
        Maximum softmax probability (confidence).
        
        Args:
            logits: [batch_size, vocab_size]
        
        Returns:
            max_prob: [batch_size]
        """
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1)[0]
        return max_prob
    
    @staticmethod
    def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute perplexity.
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len]
        
        Returns:
            perplexity: [batch_size]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs of target tokens
        target_log_probs = log_probs.gather(
            dim=-1,
            index=targets.unsqueeze(-1)
        ).squeeze(-1)
        
        # Average and exponentiate
        avg_nll = -target_log_probs.mean(dim=-1)
        perplexity = torch.exp(avg_nll)
        
        return perplexity


class SupervisionSignalGenerator:
    """Generate supervision signals for probe training."""
    
    def __init__(
        self,
        model,
        tokenizer,
        num_generations: int = 10,
        temperature: float = 1.0,
        max_new_tokens: int = 50
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_generations = num_generations
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        self.semantic_entropy_calc = SemanticEntropyCalculator()
        self.metrics = UncertaintyMetrics()
    
    @torch.no_grad()
    def generate_supervision_signals(
        self,
        input_ids: torch.Tensor,
        ground_truth_answers: Optional[List[List[str]]] = None,
        methods: List[str] = ['semantic_entropy', 'correctness']
    ) -> Dict[str, torch.Tensor]:
        """
        Generate supervision signals from model generations.
        
        Args:
            input_ids: [batch_size, seq_len]
            ground_truth_answers: List of acceptable answers per question
            methods: Which supervision signals to compute
        
        Returns:
            Dictionary of supervision signals:
                - semantic_entropy: [batch_size]
                - correctness: [batch_size] (binary)
                - confidence: [batch_size]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        signals = {}
        
        # Generate multiple outputs
        all_generations = []
        all_probs = []
        all_logits = []
        
        for _ in range(self.num_generations):
            outputs = self.model.base_model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode generations
            generations = self.tokenizer.batch_decode(
                outputs.sequences[:, input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Compute generation probabilities
            # Average token probability as proxy for sequence probability
            gen_probs = []
            for scores in outputs.scores:
                probs = F.softmax(scores, dim=-1)
                token_probs = probs.max(dim=-1)[0]
                gen_probs.append(token_probs)
            
            avg_probs = torch.stack(gen_probs).mean(dim=0).cpu().numpy()
            
            all_generations.append(generations)
            all_probs.append(avg_probs)
            all_logits.append(outputs.scores)
        
        # Transpose to get per-sample generations
        sample_generations = [
            [all_generations[i][j] for i in range(self.num_generations)]
            for j in range(batch_size)
        ]
        
        sample_probs = [
            [all_probs[i][j] for i in range(self.num_generations)]
            for j in range(batch_size)
        ]
        
        # Compute semantic entropy
        if 'semantic_entropy' in methods:
            semantic_entropies = []
            for gens, probs in zip(sample_generations, sample_probs):
                se = self.semantic_entropy_calc.compute_semantic_entropy(
                    gens, probs
                )
                semantic_entropies.append(se)
            
            signals['semantic_entropy'] = torch.tensor(
                semantic_entropies, device=device
            )
        
        # Compute correctness (if ground truth available)
        if 'correctness' in methods and ground_truth_answers is not None:
            correctness = []
            for gens, gt_answers in zip(sample_generations, ground_truth_answers):
                # Check if any generation matches ground truth
                is_correct = any(
                    self._check_answer_correctness(gen, gt_answers)
                    for gen in gens
                )
                correctness.append(1.0 if is_correct else 0.0)
            
            signals['correctness'] = torch.tensor(correctness, device=device)
        
        # Compute confidence (average max probability)
        if 'confidence' in methods:
            avg_confidence = [np.mean(probs) for probs in sample_probs]
            signals['confidence'] = torch.tensor(avg_confidence, device=device)
        
        return signals
    
    @staticmethod
    def _check_answer_correctness(
        generation: str,
        ground_truth: List[str]
    ) -> bool:
        """Check if generation matches any ground truth answer."""
        gen_normalized = generation.lower().strip()
        
        for gt in ground_truth:
            gt_normalized = gt.lower().strip()
            if gt_normalized in gen_normalized or gen_normalized in gt_normalized:
                return True
        
        return False
    
    def create_binary_labels(
        self,
        uncertainty_scores: torch.Tensor,
        threshold: Optional[float] = None,
        method: str = 'median'
    ) -> torch.Tensor:
        """
        Convert continuous uncertainty scores to binary labels.
        
        Args:
            uncertainty_scores: [batch_size]
            threshold: Manual threshold (if None, computed automatically)
            method: 'median', 'mean', or 'quantile'
        
        Returns:
            binary_labels: [batch_size] - 1 for uncertain, 0 for certain
        """
        if threshold is None:
            if method == 'median':
                threshold = uncertainty_scores.median().item()
            elif method == 'mean':
                threshold = uncertainty_scores.mean().item()
            elif method == 'quantile':
                threshold = uncertainty_scores.quantile(0.75).item()
        
        binary_labels = (uncertainty_scores > threshold).long()
        
        return binary_labels