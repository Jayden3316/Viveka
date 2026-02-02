"""
Data processing for TriviaQA, PopQA, and HaluEval datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import pickle


class QADataset(Dataset):
    """Base dataset for Q&A with uncertainty annotations."""
    
    def __init__(
        self,
        questions: List[str],
        answers: List[List[str]],  # Multiple acceptable answers
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        probe_token: str = '[PROBE]'
    ):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.probe_token = probe_token
        
        # Cache for generated outputs and uncertainty scores
        self.generations_cache = {}
        self.uncertainty_cache = {}
        
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        question = self.questions[idx]
        answers = self.answers[idx]
        
        # Tokenize with probe token appended
        text = f"{question} {self.probe_token}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'question': question,
            'answers': answers,
            'idx': idx
        }
    
    def add_uncertainty_labels(
        self,
        uncertainty_scores: Dict[int, float],
        uncertainty_binary: Optional[Dict[int, int]] = None
    ):
        """Add precomputed uncertainty labels."""
        self.uncertainty_cache = uncertainty_scores
        if uncertainty_binary:
            self.binary_labels = uncertainty_binary
    
    def get_with_labels(self, idx: int) -> Dict:
        """Get item with uncertainty labels if available."""
        item = self[idx]
        
        if idx in self.uncertainty_cache:
            item['uncertainty_score'] = self.uncertainty_cache[idx]
        
        if hasattr(self, 'binary_labels') and idx in self.binary_labels:
            item['uncertainty_label'] = self.binary_labels[idx]
        
        return item


class TriviaQAProcessor:
    """Process TriviaQA dataset."""
    
    @staticmethod
    def load_data(split: str = 'train', num_samples: Optional[int] = None) -> Tuple[List[str], List[List[str]]]:
        """
        Load TriviaQA data.
        
        Returns:
            questions: List of questions
            answers: List of acceptable answer lists
        """
        dataset = load_dataset('trivia_qa', 'unfiltered.nocontext', split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        questions = []
        answers = []
        
        for item in dataset:
            questions.append(item['question'])
            # TriviaQA has aliases - all acceptable
            answer_list = [item['answer']['value']] + item['answer']['aliases']
            answers.append(answer_list)
        
        return questions, answers


class PopQAProcessor:
    """Process PopQA dataset."""
    
    @staticmethod
    def load_data(split: str = 'test', num_samples: Optional[int] = None) -> Tuple[List[str], List[List[str]]]:
        """Load PopQA data."""
        dataset = load_dataset('akariasai/PopQA', split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        questions = []
        answers = []
        
        for item in dataset:
            questions.append(item['question'])
            # PopQA format
            possible_answers = item.get('possible_answers', [item['answer']])
            answers.append(possible_answers if isinstance(possible_answers, list) else [possible_answers])
        
        return questions, answers


class HaluEvalProcessor:
    """Process HaluEval dataset (specifically QA subset)."""
    
    @staticmethod
    def load_data(subset: str = 'qa_samples', num_samples: Optional[int] = None) -> Tuple[List[str], List[List[str]]]:
        """
        Load HaluEval data.
        
        HaluEval contains hallucinated and non-hallucinated samples.
        """
        dataset = load_dataset('pminervini/HaluEval', subset, split='data')
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        questions = []
        answers = []
        
        for item in dataset:
            # Extract question and correct answer
            questions.append(item['question'])
            
            # HaluEval structure varies by subset
            if 'right_answer' in item:
                answers.append([item['right_answer']])
            elif 'answer' in item:
                answers.append([item['answer']])
            else:
                answers.append([item.get('knowledge', '')])
        
        return questions, answers


def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    train_config: Dict,
    data_dir: Path = Path('./data')
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        tokenizer: Tokenizer with [PROBE] token
        train_config: Training configuration
        data_dir: Directory for caching processed data
    
    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / 'processed_datasets.pkl'
    
    # Check cache
    if cache_file.exists() and not train_config.get('reload_data', False):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        train_dataset = QADataset(
            cached_data['train_questions'],
            cached_data['train_answers'],
            tokenizer,
            max_length=train_config.get('max_length', 512)
        )
        
        val_datasets = {}
        for name in ['triviaqa', 'popqa', 'halueval']:
            val_datasets[name] = QADataset(
                cached_data[f'{name}_questions'],
                cached_data[f'{name}_answers'],
                tokenizer,
                max_length=train_config.get('max_length', 512)
            )
    else:
        print("Processing datasets from scratch...")
        
        # Load training data (TriviaQA)
        train_questions, train_answers = TriviaQAProcessor.load_data(
            split='train',
            num_samples=train_config.get('num_train_samples', 20000)
        )
        
        # Load validation data
        val_triviaqa_q, val_triviaqa_a = TriviaQAProcessor.load_data(
            split='validation',
            num_samples=train_config.get('num_val_samples', 2000)
        )
        
        val_popqa_q, val_popqa_a = PopQAProcessor.load_data(
            num_samples=train_config.get('num_val_samples', 2000)
        )
        
        val_halueval_q, val_halueval_a = HaluEvalProcessor.load_data(
            num_samples=train_config.get('num_val_samples', 2000)
        )
        
        # Cache processed data
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'train_questions': train_questions,
                'train_answers': train_answers,
                'triviaqa_questions': val_triviaqa_q,
                'triviaqa_answers': val_triviaqa_a,
                'popqa_questions': val_popqa_q,
                'popqa_answers': val_popqa_a,
                'halueval_questions': val_halueval_q,
                'halueval_answers': val_halueval_a,
            }, f)
        
        # Create datasets
        train_dataset = QADataset(train_questions, train_answers, tokenizer)
        val_datasets = {
            'triviaqa': QADataset(val_triviaqa_q, val_triviaqa_a, tokenizer),
            'popqa': QADataset(val_popqa_q, val_popqa_a, tokenizer),
            'halueval': QADataset(val_halueval_q, val_halueval_a, tokenizer),
        }
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 8),
        shuffle=True,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loaders = {}
    for name, dataset in val_datasets.items():
        val_loaders[name] = DataLoader(
            dataset,
            batch_size=train_config.get('val_batch_size', 16),
            shuffle=False,
            num_workers=train_config.get('num_workers', 4),
            pin_memory=True
        )
    
    return {'train': train_loader, **val_loaders}