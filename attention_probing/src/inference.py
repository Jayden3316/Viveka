"""
Inference and evaluation for trained probes.
"""

import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

from model import ModelWithProbe


class ProbeInference:
    """Run inference with trained probes."""
    
    def __init__(
        self,
        model: ModelWithProbe,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def predict_uncertainty(
        self,
        questions: List[str],
        batch_size: int = 8,
        return_attention: bool = False
    ) -> Dict[str, List]:
        """
        Predict uncertainty for a list of questions.
        
        Returns:
            Dictionary with:
                - uncertainty_scores: List of uncertainty predictions
                - generations: List of generated answers
                - attention_patterns: Optional attention weights
        """
        results = {
            'uncertainty_scores': [],
            'generations': [],
            'questions': questions
        }
        
        if return_attention:
            results['attention_patterns'] = []
        
        # Process in batches
        for i in tqdm(range(0, len(questions), batch_size)):
            batch_questions = questions[i:i + batch_size]
            
            # Tokenize with probe token
            inputs = self.model.tokenizer(
                [f"{q} [PROBE]" for q in batch_questions],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get probe predictions
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_probe_logits=True,
                output_attentions=return_attention
            )
            
            probe_logits = outputs['probe_logits']
            
            # Convert to uncertainty scores
            if self.model.probe_classifier.use_regression:
                uncertainty = probe_logits.squeeze(-1).cpu().tolist()
            else:
                # Use softmax probability of "uncertain" class
                probs = torch.softmax(probe_logits, dim=-1)
                uncertainty = probs[:, 1].cpu().tolist()  # Class 1 = uncertain
            
            results['uncertainty_scores'].extend(uncertainty)
            
            # Generate answers (without probe token)
            gen_inputs = self.model.tokenizer(
                batch_questions,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            generated = self.model.base_model.generate(
                **gen_inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.model.tokenizer.pad_token_id
            )
            
            decoded = self.model.tokenizer.batch_decode(
                generated[:, gen_inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            results['generations'].extend(decoded)
            
            if return_attention:
                # Extract attention to BOS for each layer
                attention_weights = outputs['attentions']
                batch_patterns = self._extract_attention_patterns(attention_weights)
                results['attention_patterns'].extend(batch_patterns)
        
        return results
    
    def _extract_attention_patterns(
        self,
        attention_weights: Tuple[torch.Tensor, ...]
    ) -> List[Dict]:
        """
        Extract attention patterns focusing on probe token.
        
        Returns:
            List of dictionaries with attention info per sample
        """
        patterns = []
        
        # attention_weights: tuple of [batch, num_heads, seq_len, seq_len] per layer
        batch_size = attention_weights[0].shape[0]
        
        for b in range(batch_size):
            sample_pattern = {
                'bos_attention_per_layer': [],  # Attention to BOS from probe
                'layer_importance': []  # 1 - BOS attention
            }
            
            for layer_attn in attention_weights:
                # Average across heads
                avg_attn = layer_attn[b].mean(dim=0)  # [seq_len, seq_len]
                
                # Get probe token attention (last position)
                probe_attn = avg_attn[-1, :]  # [seq_len]
                
                # Attention to BOS
                bos_attn = probe_attn[0].item()
                sample_pattern['bos_attention_per_layer'].append(bos_attn)
                
                # Layer importance (1 - BOS attention)
                importance = 1.0 - bos_attn
                sample_pattern['layer_importance'].append(importance)
            
            patterns.append(sample_pattern)
        
        return patterns
    
    def evaluate_on_dataset(
        self,
        questions: List[str],
        ground_truth_answers: List[List[str]],
        batch_size: int = 8,
        save_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Evaluate probe on a dataset with ground truth.
        
        Returns:
            Dictionary of evaluation metrics
        """
        results = self.predict_uncertainty(
            questions,
            batch_size=batch_size,
            return_attention=True
        )
        
        # Compute metrics
        uncertainty_scores = results['uncertainty_scores']
        generations = results['generations']
        
        # Correctness
        correct = []
        for gen, gt_answers in zip(generations, ground_truth_answers):
            is_correct = any(
                gt.lower() in gen.lower() for gt in gt_answers
            )
            correct.append(is_correct)
        
        # Correlation between uncertainty and incorrectness
        import numpy as np
        from scipy.stats import spearmanr, pearsonr
        
        incorrect = [1 - int(c) for c in correct]
        
        spearman_corr, spearman_p = spearmanr(uncertainty_scores, incorrect)
        pearson_corr, pearson_p = pearsonr(uncertainty_scores, incorrect)
        
        # Calibration: high uncertainty should correlate with incorrectness
        metrics = {
            'accuracy': sum(correct) / len(correct),
            'avg_uncertainty': np.mean(uncertainty_scores),
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p
        }
        
        # Save results
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'results': results,
                    'ground_truth': ground_truth_answers
                }, f, indent=2)
            
            print(f"Saved evaluation results to {save_path}")
        
        return metrics
    
    def analyze_layer_importance(
        self,
        questions: List[str],
        batch_size: int = 8,
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Analyze which layers are most important for uncertainty detection.
        
        Returns:
            Dictionary with layer importance statistics
        """
        results = self.predict_uncertainty(
            questions,
            batch_size=batch_size,
            return_attention=True
        )
        
        attention_patterns = results['attention_patterns']
        
        # Aggregate layer importance across all samples
        import numpy as np
        
        num_layers = len(attention_patterns[0]['layer_importance'])
        
        layer_importances = np.array([
            pattern['layer_importance']
            for pattern in attention_patterns
        ])  # [num_samples, num_layers]
        
        analysis = {
            'mean_importance_per_layer': layer_importances.mean(axis=0).tolist(),
            'std_importance_per_layer': layer_importances.std(axis=0).tolist(),
            'median_importance_per_layer': np.median(layer_importances, axis=0).tolist(),
            'num_layers': num_layers
        }
        
        # Identify most/least important layers
        mean_importance = analysis['mean_importance_per_layer']
        analysis['most_important_layers'] = sorted(
            range(len(mean_importance)),
            key=lambda i: mean_importance[i],
            reverse=True
        )[:5]
        
        analysis['least_important_layers'] = sorted(
            range(len(mean_importance)),
            key=lambda i: mean_importance[i]
        )[:5]
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            print(f"Saved layer importance analysis to {save_path}")
        
        return analysis


def load_checkpoint_for_inference(
    checkpoint_path: Path,
    device: str = 'cuda'
) -> ProbeInference:
    """Load trained model from checkpoint for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Reconstruct model
    from model import ModelWithProbe
    
    model = ModelWithProbe(
        model_name=config['model_name'],
        probe_config=config['probe_config']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return ProbeInference(model, device=device)