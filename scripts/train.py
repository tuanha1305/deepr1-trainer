import argparse
import yaml
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from models import get_model
from data import RLDataset, RLCollator
from trainers import RLTrainer
from utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Train RL model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config['training'].get('exp_name', 'train'))

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])

    # Create datasets
    train_dataset = RLDataset(
        data_path=config['data']['train_data_path'],
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )

    val_dataset = RLDataset(
        data_path=config['data']['eval_data_path'],
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )

    # Create dataloaders
    collator = RLCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collator
    )

    # Initialize model
    model = get_model(
        config['model']['model_type'],
        **config['model']
    )

    # Initialize optimizer
    optimizer_name = config['training']['optimizer']
    if optimizer_name == 'RiemannianAdam':
        from geoopt.optim import RiemannianAdam
        optimizer = RiemannianAdam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
    else:
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(
            model.parameters(),
            lr=config['training']['learning_rate']
        )

    # Initialize trainer
    trainer = RLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=args.device,
        config=config
    )

    # Load checkpoint if specified
    if args.checkpoint:
        trainer._load_checkpoint(args.checkpoint)

    # Start training
    trainer.train(config['training']['epochs'])


if __name__ == '__main__':
    main()
