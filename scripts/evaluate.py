import argparse
import yaml
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import json

from models import get_model
from data import RLDataset, RLCollator
from utils.metrics import calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RL model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Path to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    model = get_model(config['model']['model_type'], **config['model'])

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(args.device)
    model.eval()

    # Create test dataset and loader
    test_dataset = RLDataset(
        data_path=config['data']['test_data_path'],
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )

    collator = RLCollator(pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collator
    )

    # Evaluate
    all_metrics = []
    generated_outputs = []

    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)

            # Get model outputs
            outputs, _ = model(input_ids, attention_mask)

            # Calculate metrics
            metrics = calculate_metrics(
                outputs=outputs,
                targets=batch.get('target_ids'),
                rewards=None  # No rewards in evaluation
            )
            all_metrics.append(metrics)

            # Decode outputs
            decoded_outputs = tokenizer.batch_decode(
                outputs.argmax(-1),
                skip_special_tokens=True
            )
            generated_outputs.extend(decoded_outputs)

    # Aggregate metrics
    final_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        final_metrics[key] = sum(values) / len(values)

    # Save results
    if args.output:
        results = {
            'metrics': final_metrics,
            'generated_outputs': generated_outputs
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    # Print metrics
    print("\nEvaluation Results:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    main()
