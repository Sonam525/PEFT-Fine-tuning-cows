# Efficiency at Scale: Fine-Tuning Billion-Parameter Models for Dairy Cattle Monitoring

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/Sonam5)

> **Leveraging Foundation Vision Models with QLoRA and DoRA for Dairy Cattle Behavior Recognition**

This repository contains the official implementation of our paper exploring Parameter-Efficient Fine-Tuning (PEFT) methods for adapting billion-parameter vision models to livestock behavior classification.

![Infographic](./assets/Infographics.png)

## 🎯 Key Findings

- **83.16% test accuracy** using QLoRA with only **2.72% trainable parameters**
- **65% reduction** in training time compared to training from scratch
- **1:98 training-to-test ratio** - 2,160 verified images generalize to 211,800+ real-world samples
- **Underfitting, not overfitting**, is the primary challenge when adapting foundation models to agricultural imagery

## 📊 Results Summary

| Method | Target Modules | Rank | Trainable Params | Training Time | Test Accuracy |
|--------|---------------|------|------------------|---------------|---------------|
| ResNet-18 (scratch) | — | — | 11.2M (100%) | 16h 45m | 72.87% |
| ViT-Small (scratch) | — | — | 21.7M (100%) | 18h 39m | 61.91% |
| DINOv3 (frozen) | — | — | 4.7M (0.07%) | 17h 27m | 76.56% |
| QLoRA | q_proj | 8 | 2.6M (0.04%) | 6h 32m | 77.17% |
| QLoRA | q_proj | 16 | 5.2M (0.08%) | 7h 16m | 78.38% |
| QLoRA | all-linear | 16 | 46.8M (0.70%) | 4h 43m | 80.40% |
| **QLoRA** | **all-linear** | **64** | **183.0M (2.72%)** | **5h 46m** | **83.16%** |
| DoRA | q_proj | 8 | 2.8M (0.04%) | 11h 31m | 81.53% |
| DoRA | q_proj | 16 | 5.4M (0.08%) | 10h 27m | 81.03% |
| DoRA | all-linear | 16 | 48.4M (0.72%) | 11h 51m | 81.23% |
| **DoRA** | **all-linear** | **64** | **184.5M (2.75%)** | **10h 59m** | **83.14%** |

## 🏗️ Repository Structure

```
PEFT-Fine-tuning-cows/
├── assets/
│   └── infographics.png
├── notebooks/
│   ├── DoRA Fewer Layers R=8.ipynb
│   ├── DoRA Fewer Layers R=16.ipynb
│   ├── DoRA More Layers R=16.ipynb
│   ├── DoRA More Layers R=64.ipynb
│   ├── Q-lora Fewer Layers R = 8.ipynb
│   ├── Q-lora Fewer Layers R = 16.ipynb
│   ├── Q-lora More Layers Rank = 16.ipynb
│   ├── Q-lora More Layers Rank = 64.ipynb
│   ├── TrainFromScatch_Preprocesing Pipeline Val = 1.ipynb
│   └── UsingPretrainedModel_DinoV3 Embeddings Val = 1.ipynb
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

## 📓 Notebooks

### Baseline Approaches
| Notebook | Description |
|----------|-------------|
| [TrainFromScatch_Preprocesing Pipeline Val = 1](notebooks/TrainFromScatch_Preprocesing%20Pipeline%20Val%20=%201.ipynb) | Training ResNet-18/Vit-Small from scratch with data preprocessing pipeline |
| [UsingPretrainedModel_DinoV3 Embeddings Val = 1](notebooks/UsingPretrainedModel_DinoV3%20Embeddings%20Val%20=%201.ipynb) | Frozen DINOv3 feature extraction with classification head |

### QLoRA Experiments
| Notebook | Target Modules | Rank | Test Accuracy |
|----------|---------------|------|---------------|
| [Q-lora Fewer Layers R = 8](notebooks/Q-lora%20Fewer%20Layers%20R%20=%208.ipynb) | q_proj | 8 | 77.17% |
| [Q-lora Fewer Layers R = 16](notebooks/Q-lora%20Fewer%20Layers%20R%20=%2016.ipynb) | q_proj | 16 | 78.38% |
| [Q-lora More Layers Rank = 16](notebooks/Q-lora%20More%20Layers%20Rank%20=%2016.ipynb) | all-linear | 16 | 80.40% |
| [Q-lora More Layers Rank = 64](notebooks/Q-lora%20More%20Layers%20Rank%20=%2064.ipynb) | all-linear | 64 | **83.16%** |

### DoRA Experiments
| Notebook | Target Modules | Rank | Test Accuracy |
|----------|---------------|------|---------------|
| [DoRA Fewer Layers R=8](notebooks/DoRA%20Fewer%20Layers%20R=8.ipynb) | q_proj | 8 | 81.53% |
| [DoRA Fewer Layers R=16](notebooks/DoRA%20Fewer%20Layers%20R=16.ipynb) | q_proj | 16 | 81.03% |
| [DoRA More Layers R=16](notebooks/DoRA%20More%20Layers%20R=16.ipynb) | all-linear | 16 | 81.23% |
| [DoRA More Layers R=64](notebooks/DoRA%20More%20Layers%20R=64.ipynb) | all-linear | 64 | **83.14%** |

## 🤗 Pre-trained Models

Model checkpoints are available on Hugging Face:

| Model | Description | Test Accuracy | Link |
|-------|-------------|---------------|------|
| DINOv3 + QLoRA (r=64) | Best performing model | 83.16% | [🤗 cow-behavior-dinov3-qlora-r64](https://huggingface.co/Sonam5/cow-behavior-dinov3-qlora-r64) |
| DINOv3 + DoRA (r=64) | Best DoRA configuration | 83.14% | [🤗 cow-behavior-dinov3-dora-r64](https://huggingface.co/Sonam5/cow-behavior-dinov3-dora-r64) |
| DINOv3 Frozen | Frozen feature extraction | 76.56% | [🤗 cow-behavior-dinov3-Frozen](https://huggingface.co/Sonam5/cow-behavior-dinov3-Frozen) |
| ResNet-18 (scratch) | Baseline trained from scratch | 72.87% | [🤗 cow-behavior-from-scratch](https://huggingface.co/Sonam5/cow-behavior-from-scratch) |

### Loading Pre-trained Models

```python
from transformers import AutoModel
from peft import PeftModel

# Load QLoRA model (best performing)
base_model = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
model = PeftModel.from_pretrained(base_model, "Sonam5/cow-behavior-dinov3-qlora-r64")
```

## 📁 Dataset

Our dataset consists of **nine dairy cow behaviors**:

| Behavior | Training Samples | Test Samples |
|----------|------------------|--------------|
| Drinking | 240 | 3,011 |
| Feeding head down | 240 | 30,952 |
| Feeding head up | 240 | 18,783 |
| Lying | 240 | 83,509 |
| Standing | 240 | 69,807 |
| Walking | 240 | 3,819 |
| Frontal pushing | 240 | 600 |
| Gallop | 240 | 575 |
| Leap | 240 | 744 |

**Total**: 2,160 training images → 211,800 test samples (1:98 ratio)

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- CUDA compatible GPU with 16GB+ VRAM (Tesla V100 or equivalent)

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PEFT-Fine-tuning-cows.git
cd PEFT-Fine-tuning-cows

# Create conda environment
conda env create -f environment.yml
conda activate cow-behavior-analysis
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PEFT-Fine-tuning-cows.git
cd PEFT-Fine-tuning-cows

# Install dependencies
pip install -r requirements.txt
```

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{yang2025peft_cattle,
  title={When Billion-Parameter Foundation Models Meet Limited Data: Parameter-Efficient Fine-Tuning with QLoRA and DoRA for Generalizable Image Classification},
  author={Yang, Haiyu and Sharma, Sumit and Liu, Enhong and Hostens, Miel},
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DINOv3](https://github.com/facebookresearch/dinov3) - Foundation model
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning library
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - Quantization library

## 📧 Contact

**Haiyu Yang ** - Cornell University

---
