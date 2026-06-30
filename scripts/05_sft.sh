#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 5: Supervised Fine-Tuning (SFT) ==="

PROJECT_DIR="$(dirname "$0")/.."
DATA_DIR="$PROJECT_DIR/data"
CKPT_DIR="$PROJECT_DIR/checkpoints"

mkdir -p "$CKPT_DIR/sft" "$DATA_DIR/instruction"

# Create instruction dataset if it doesn't exist
if [ ! -f "$DATA_DIR/instruction/instructions.jsonl" ]; then
    echo "Creating instruction dataset..."
    python3 -c "
import json
import os

data_dir = '$DATA_DIR/instruction'
os.makedirs(data_dir, exist_ok=True)

# Instruction dataset for alignment
instructions = [
    {'instruction': 'Explain what a transformer is in machine learning.', 'input': '', 'output': 'A transformer is a neural network architecture that uses self-attention mechanisms to process sequential data. It was introduced in the paper \"Attention Is All You Need\" (Vaswani et al., 2017). The key innovation is the self-attention mechanism, which allows the model to weigh the importance of different parts of the input when producing each part of the output. Transformers have become the foundation of modern NLP, powering models like GPT, BERT, and LLaMA.'},
    {'instruction': 'What is the difference between DPO and RLHF?', 'input': '', 'output': 'RLHF (Reinforcement Learning from Human Feedback) trains a separate reward model on human preferences, then uses PPO to optimize the language model against that reward. DPO (Direct Preference Optimization) skips the reward model entirely — it directly optimizes the language model using a loss function derived from the Bradley-Terry preference model. DPO is simpler and more stable, while RLHF can achieve higher peaks with more careful tuning.'},
    {'instruction': 'What does RoPE do?', 'input': '', 'output': 'RoPE (Rotary Position Embedding) encodes positional information by rotating the query and key vectors in attention. Instead of adding position embeddings, it applies a rotation matrix that encodes relative positions. This means the attention score between two tokens depends only on their relative distance, not absolute positions. RoPE enables better length generalization and is used in LLaMA, Mistral, and most modern LLMs.'},
    {'instruction': 'Why use RMSNorm instead of LayerNorm?', 'input': '', 'output': 'RMSNorm (Root Mean Square Layer Normalization) simplifies LayerNorm by removing the mean centering step. It only normalizes by the root mean square of activations. This makes it ~10-15% faster while maintaining equivalent performance. RMSNorm also eliminates the bias term, reducing parameters. It has become the standard in modern LLMs (LLaMA, Mistral, Qwen).'},
    {'instruction': 'What is SwiGLU?', 'input': '', 'output': 'SwiGLU is an activation function used in the feed-forward network of transformers. It combines a Swish activation with a Gated Linear Unit. The FFN has three weight matrices instead of two: one for the gate, one for the value, and one for the output projection. The gate controls which information passes through. SwiGLU consistently outperforms ReLU and GELU in language models.'},
    {'instruction': 'How does GQA work?', 'input': '', 'output': 'GQA (Grouped Query Attention) is a middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). In MHA, each head has its own Q, K, V. In MQA, all heads share one K, V. GQA groups heads together, sharing K, V within each group. For example, with 8 query heads and 4 KV heads, every 2 query heads share one K, V pair. This reduces KV cache memory by 2x while maintaining most of MHA quality.'},
    {'instruction': 'What is the WSD learning rate schedule?', 'input': '', 'output': 'WSD (Warmup-Stable-Decay) is a learning rate schedule with three phases: warmup (linearly increasing LR), stable (constant LR), and decay (exponentially decreasing LR). It was found to outperform pure cosine decay for LLM training, especially for continual pretraining. The key insight is that the stable phase allows consistent learning, while the decay phase refines the model.'},
    {'instruction': 'Explain tokenization in LLMs.', 'input': '', 'output': 'Tokenization converts raw text into numerical tokens that the model processes. Most modern LLMs use BPE (Byte Pair Encoding) or its variants. BPE starts with individual bytes/characters and iteratively merges the most frequent pairs. This creates a vocabulary that balances between character-level (too long sequences) and word-level (too large vocabulary, no generalization). SentencePiece is a popular implementation that handles unnormalized text.'},
]

with open(os.path.join(data_dir, 'instructions.jsonl'), 'w') as f:
    for item in instructions:
        f.write(json.dumps(item) + '\n')

print(f'Created {len(instructions)} instruction examples')
"
fi

# Run SFT
echo "Starting supervised fine-tuning..."
python3 -c "
import sys
import os
import torch

sys.path.insert(0, '$PROJECT_DIR')
from config.model_config import MewtwoConfig
from src.tokenizer.tokenizer import MewtwoTokenizer
from src.alignment.sft import SFTTrainer

config = MewtwoConfig()
tokenizer = MewtwoTokenizer.from_pretrained('$DATA_DIR/tokenizer')

# Load latest pretrain checkpoint
ckpt_dir = '$CKPT_DIR/pretrain'
best_ckpt = os.path.join(ckpt_dir, 'best_model.pt')
if os.path.exists(best_ckpt):
    checkpoint = torch.load(best_ckpt, map_location='cpu', weights_only=False)
    print(f'Loaded checkpoint: {best_ckpt}')
else:
    print('WARNING: No pretrain checkpoint found. Training from scratch.')
    checkpoint = None

sft = SFTTrainer(
    config=config,
    tokenizer=tokenizer,
    output_dir='$CKPT_DIR/sft',
    lr=5e-5,
    max_steps=500,
    batch_size=4,
    gradient_accumulation_steps=4,
)

sft.train('$DATA_DIR/instruction/instructions.jsonl')
print('SFT complete!')
"
