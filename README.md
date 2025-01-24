# DeepR1 Trainer

A robust implementation for training and fine-tuning language models using Reinforcement Learning in hyperbolic space.

## 🌟 Features

- **Hyperbolic Embeddings**: Train models in hyperbolic space for better representation of hierarchical data
- **Reinforcement Learning**: Custom RL training loop with configurable rewards
- **Knowledge Distillation**: Compress models while maintaining performance
- **Modular Architecture**: Easily extensible for different model architectures and training strategies
- **Comprehensive Logging**: Built-in support for experiment tracking and visualization
- **Efficient Data Processing**: Optimized data loading and preprocessing pipeline

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
CUDA >= 11.6 (for GPU support)
```

### Installation

```bash
git clone https://github.com/tuanha1305/deepr1-trainer.git
cd deepr1-trainer
pip install -r requirements.txt
```

### Quick Start

1. Configure your training settings in `configs/`:
```python
# configs/training_config.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    max_seq_length: int = 512
```

2. Run training:
```bash
python scripts/train.py --config configs/default_config.yaml
```

3. Run distillation:
```bash
python scripts/distill.py --teacher-model path/to/model --config configs/distill_config.yaml
```

## 📊 Project Structure

```
deepr1_trainer/
├── configs/          # Configuration files
├── data/            # Dataset and data processing
├── models/          # Model architectures
├── trainers/        # Training implementations
├── utils/           # Utility functions
├── rewards/         # Reward functions
├── scripts/         # Training scripts
└── tests/           # Unit tests
```

## 💡 Key Components

### Models

The project includes two main model architectures:

1. **SmallRLModel**: 
   - Uses hyperbolic embeddings
   - Suitable for hierarchical data
   - Configurable architecture

2. **SmallerRLModel**: 
   - Distilled version
   - Faster inference
   - Maintains core capabilities

### Reward Functions

Customizable reward functions include:

- Accuracy rewards based on cosine similarity
- Format adherence rewards
- Combined reward strategies

## 🔧 Configuration

All hyperparameters and training settings can be configured through YAML files:

```yaml
# configs/default_config.yaml
model:
  input_dim: 128
  hidden_dim: 256
  output_dim: 128

training:
  learning_rate: 1e-4
  batch_size: 32
  epochs: 10
  max_seq_length: 512

rewards:
  accuracy_weight: 1.0
  format_weight: 0.5
```

## 📈 Monitoring

Training progress can be monitored through:

- Tensorboard integration
- Custom metric logging
- Progress visualization tools

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📫 Contact

Lucas Ha - tuanictu97@gmail.com
Project Link: https://github.com/tuanha1305/deepr1-trainer

## 🙏 Acknowledgments

- [GeoOpt](https://github.com/geoopt/geoopt) for hyperbolic optimization
- [PyTorch](https://pytorch.org/) framework
- [Transformers](https://github.com/huggingface/transformers) library