import argparse
import yaml
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from deepseek_trainer.models import get_model
from deepseek_trainer.data import RLDataset, RLCollator
from deepseek_trainer.trainers import DistillationTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Distill RL model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--teacher-checkpoint', type=str, required=True, help='Path to teacher model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save student model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

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

    # Initialize teacher model
    teacher_model = get_model(config['model']['model_type'], **config['model'])
    checkpoint = torch.load(args.teacher_checkpoint, map_location=args.device)
    teacher_model.load_state_dict(checkpoint['model_state'])

    # Initialize student model (smaller architecture)
    student_config = config['model'].copy()
    student_config['hidden_dim'] = student_config['hidden_dim'] // 2
    student_model = get_model('SmallerRLModel', **student_config)

    # Initialize optimizer for student
    optimizer = torch.optim.Adam(
        student_model.parameters(),
        lr=config['training']['learning_rate']
    )

    # Initialize trainer
    trainer = DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=args.device,
        config=config,
        temperature=config.get('distillation', {}).get('temperature', 2.0)
    )

    # Start distillation
    trainer.train(config['training']['epochs'])

    # Save final student model
    torch.save({
        'model_state': student_model.state_dict(),
        'config': student_config
    }, f"{args.output_dir}/student_final.pt")


if __name__ == '__main__':
    main()
