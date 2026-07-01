import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gradio as gr
from src.model.pocketllm import MewtwoLLM
from src.tokenizer.tokenizer import load_tokenizer
from src.inference.generate import generate
from config.model_config import MewtwoConfig

# Load model and tokenizer
config = MewtwoConfig()
tokenizer = load_tokenizer("data/tokenizer/mewtwo.model")
model = MewtwoLLM(config)

# Load checkpoint (try different naming conventions)
for stage in ["rlhf", "dpo", "sft", "pretrain"]:
    for name in ["mewtwo_final.pt", "mewtwo_best.pt", "model.pt"]:
        ckpt = f"checkpoints/{stage}/{name}"
        if os.path.exists(ckpt):
            checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint: {ckpt}")
            break
    else:
        continue
    break

model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {n_params:,} params")


def generate_text(prompt, max_tokens, temperature, top_k, top_p):
    if not prompt.strip():
        return "Please enter a prompt."
    
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )
    return output


demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="What is the meaning of life?", lines=3),
        gr.Slider(minimum=16, maximum=512, value=200, step=16, label="Max Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=0, maximum=100, value=50, step=5, label="Top-K"),
        gr.Slider(minimum=0.5, maximum=1.0, value=0.95, step=0.05, label="Top-P"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title=f"MewtwoLLM — {n_params/1e6:.0f}M Parameter LLM",
    description="A language model built from scratch with RoPE + RMSNorm + SwiGLU + GQA.",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
