import yaml
import argparse
import torch
import json
import re
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# Import data processors
from data import TriviaQAProcessor, PopQAProcessor, HaluEvalProcessor

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename."""
    return re.sub(r'[^\w\-_]', '_', name)

class GenerationDataset(Dataset):
    def __init__(self, questions: List[str], tokenizer, system_prompt: str, template_kwargs: Dict = None):
        self.questions = questions
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.template_kwargs = template_kwargs or {}
        
    def __len__(self):
        return len(self.questions)
        
    def __getitem__(self, idx):
        question = self.questions[idx]
        
        # Apply chat template if available, otherwise manual formatting
        if self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": f"{self.system_prompt}\n{question}"}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                **self.template_kwargs
            )
        else:
            # Fallback for models without chat template
            prompt = f"{self.system_prompt}\n\nQuestion: {question}\nAnswer:"
            
        return prompt

def main():
    parser = argparse.ArgumentParser(description="Generate answers using LLMs")
    parser.add_argument("--config", type=str, default="configs/generation_config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, help="Specific model key from config to run")
    parser.add_argument("--all", action="store_true", help="Run all models in config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Determine which models to run
    available_models = config['models']
    models_to_run = []
    
    if args.all:
        models_to_run = list(available_models.keys())
    elif args.model:
        if args.model not in available_models:
            raise ValueError(f"Model '{args.model}' not found in config. Available: {list(available_models.keys())}")
        models_to_run = [args.model]
    else:
        # Default behavior? require explicit selection to avoid accidents
        print(f"Available models: {list(available_models.keys())}")
        print("Please specify --model <name> or --all")
        return

    output_dir = Path(config.get('output_dir', './generations'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load generation params
    gen_params = config.get('generation_params', {})
    batch_size = config.get('batch_size', 4)
    system_prompt = config.get('system_prompt', "")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for model_key in models_to_run:
        model_name = available_models[model_key]
        print(f"\nLoading model: {model_key} ({model_name})")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = 'left' # Important for batched generation
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            if device == "cpu":
                model.to(device)
            model.eval()
            
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        for dataset_conf in config['datasets']:
            dataset_name = dataset_conf['name']
            num_samples = dataset_conf.get('num_samples', 10)
            
            print(f"Processing dataset: {dataset_name} (samples: {num_samples})")
            
            # Load data
            questions, answers = [], []
            try:
                if dataset_name == 'triviaqa':
                    questions, answers = TriviaQAProcessor.load_data(split='train', num_samples=num_samples)
                elif dataset_name == 'popqa':
                    questions, answers = PopQAProcessor.load_data(num_samples=num_samples)
                elif dataset_name == 'halueval':
                    questions, answers = HaluEvalProcessor.load_data(num_samples=num_samples)
                else:
                    print(f"Unknown dataset: {dataset_name}")
                    continue
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                continue

            # Check if tokenizer supports 'thinking' argument (for reasoning models)
            template_kwargs = {}
            if tokenizer.chat_template:
                try:
                    # quick check with dummy input
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": "test"}], 
                        tokenize=False, 
                        enable_thinking=config.get(enable_thinking)
                    )
                    template_kwargs["enable_thinking"] = False
                    print(f"Enabled 'enable_thinking=False' for {model_key}")
                except Exception:
                    # Does not support 'thinking' argument
                    pass

            # Create dataloader
            dataset = GenerationDataset(questions, tokenizer, system_prompt, template_kwargs)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            results = []
            global_idx = 0
            
            for batch_prompts in tqdm(dataloader, desc=f"Generating {dataset_name}"):
                batch_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
                
                with torch.no_grad():
                    # Generate multiple outputs if requested
                    num_return_sequences = gen_params.get('num_generations', 1)
                    
                    outputs = model.generate(
                        **batch_inputs,
                        max_new_tokens=gen_params.get('max_new_tokens', 50),
                        temperature=gen_params.get('temperature', 0.7),
                        top_p=gen_params.get('top_p', 0.9),
                        top_k=gen_params.get('top_k', 50),
                        num_return_sequences=num_return_sequences,
                        do_sample=gen_params.get('do_sample', True),
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                # Decode output
                # outputs shape: [batch_size * num_return_sequences, seq_len]
                input_len = batch_inputs['input_ids'].shape[1]
                generated_tokens = outputs[:, input_len:]
                decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
                # Reshape decoded outputs to [batch_size, num_return_sequences]
                # Each input sample corresponds to num_return_sequences consecutive outputs
                current_batch_size = len(batch_prompts)
                
                for i in range(current_batch_size):
                    start_idx = i * num_return_sequences
                    sample_generations = decoded_batch[start_idx : start_idx + num_return_sequences]
                    
                    results.append({
                        "question_number": global_idx,
                        "data": {
                            "question": questions[global_idx],
                            "generated_answers": sample_generations,
                            "ground_truth_answers": answers[global_idx]
                        }
                    })
                    global_idx += 1

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_model = sanitize_filename(model_key)
            out_file = output_dir / f"{dataset_name}_{sanitized_model}_{timestamp}.json"
            
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
