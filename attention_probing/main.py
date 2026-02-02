"""
Main CLI entry point for attention probing experiments.
"""

import argparse
import yaml
from pathlib import Path
import torch

from model import ModelWithProbe
from data import create_dataloaders
from train import ProbeTrainer
from inference import ProbeInference, load_checkpoint_for_inference


def train_command(args):
    """Train probe model."""
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line args
    if args.model_name:
        config['model_name'] = args.model_name
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    
    # Setup model
    print(f"Loading model: {config['model_name']}")
    model = ModelWithProbe(
        model_name=config['model_name'],
        probe_config=config['probe_config'],
        freeze_base=config.get('freeze_base', True)
    )
    
    # Setup data
    print("Preparing datasets...")
    dataloaders = create_dataloaders(
        tokenizer=model.tokenizer,
        train_config=config,
        data_dir=Path(args.data_dir)
    )
    
    # Setup trainer
    trainer = ProbeTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loaders={k: v for k, v in dataloaders.items() if k != 'train'},
        config=config,
        device=args.device
    )
    
    # Train
    trainer.train()


def inference_command(args):
    """Run inference with trained probe."""
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    inference = load_checkpoint_for_inference(
        Path(args.checkpoint),
        device=args.device
    )
    
    if args.questions_file:
        # Load questions from file
        with open(args.questions_file, 'r') as f:
            import json
            data = json.load(f)
            questions = data['questions']
            ground_truth = data.get('answers', None)
    else:
        # Interactive mode
        questions = [args.question]
        ground_truth = None
    
    # Run inference
    print("Running inference...")
    results = inference.predict_uncertainty(
        questions,
        batch_size=args.batch_size,
        return_attention=True
    )
    
    # Print results
    for q, u, g in zip(results['questions'], results['uncertainty_scores'], results['generations']):
        print(f"\nQuestion: {q}")
        print(f"Generated: {g}")
        print(f"Uncertainty: {u:.4f}")
    
    # Save if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")


def evaluate_command(args):
    """Evaluate trained probe."""
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    inference = load_checkpoint_for_inference(
        Path(args.checkpoint),
        device=args.device
    )
    
    # Load evaluation data
    with open(args.eval_data, 'r') as f:
        import json
        data = json.load(f)
        questions = data['questions']
        answers = data['answers']
    
    # Run evaluation
    print("Evaluating...")
    metrics = inference.evaluate_on_dataset(
        questions=questions,
        ground_truth_answers=answers,
        batch_size=args.batch_size,
        save_path=Path(args.output) if args.output else None
    )
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def analyze_command(args):
    """Analyze layer importance."""
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    inference = load_checkpoint_for_inference(
        Path(args.checkpoint),
        device=args.device
    )
    
    # Load questions
    with open(args.questions_file, 'r') as f:
        import json
        data = json.load(f)
        questions = data['questions']
    
    # Analyze
    print("Analyzing layer importance...")
    analysis = inference.analyze_layer_importance(
        questions=questions,
        batch_size=args.batch_size,
        save_path=Path(args.output) if args.output else None
    )
    
    # Print results
    print("\nLayer Importance Analysis:")
    print(f"Most important layers: {analysis['most_important_layers']}")
    print(f"Least important layers: {analysis['least_important_layers']}")


def main():
    parser = argparse.ArgumentParser(description='Attention Probing for Uncertainty Detection')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train probe model')
    train_parser.add_argument('--config', type=str, required=True, help='Path to config file')
    train_parser.add_argument('--model-name', type=str, help='Model name (overrides config)')
    train_parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    train_parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    inference_parser.add_argument('--question', type=str, help='Single question to evaluate')
    inference_parser.add_argument('--questions-file', type=str, help='JSON file with questions')
    inference_parser.add_argument('--output', type=str, help='Output file path')
    inference_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    inference_parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate probe')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    eval_parser.add_argument('--eval-data', type=str, required=True, help='Evaluation data JSON')
    eval_parser.add_argument('--output', type=str, help='Output file path')
    eval_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    eval_parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze layer importance')
    analyze_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    analyze_parser.add_argument('--questions-file', type=str, required=True, help='Questions JSON')
    analyze_parser.add_argument('--output', type=str, help='Output file path')
    analyze_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    analyze_parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'inference':
        inference_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'analyze':
        analyze_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()