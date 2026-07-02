"""
MewtwoLLM — Local CPU Training
Optimized for limited RAM. Run in phases.
"""
import os, sys, gc, time

sys.path.insert(0, os.path.dirname(__file__))

from config.model_config import MewtwoConfig
from src.training.pretrain import train

print("=" * 60)
print("MewtwoLLM — Local CPU Training (Phase Mode)")
print("=" * 60)

config = MewtwoConfig()
config.total_steps = 200        # Phase 1: 200 steps
config.batch_size = 1
config.gradient_accumulation_steps = 16
config.context_length = 512
config.device = "cpu"
config.use_compile = False
config.num_workers = 0
config.log_interval = 25
config.save_interval = 50

gc.collect()

data_path = "data/raw/openwebtext_sample.txt"
if not os.path.exists(data_path):
    print("ERROR: No training data. Run download_data.py first.")
    sys.exit(1)

print(f"Data: {data_path}")
print(f"Steps: {config.total_steps}")
print(f"Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
print(f"Context: {config.context_length}")
print(f"Checkpoint dir: checkpoints/pretrain_local")
print()

start = time.time()
model = train(config, data_path, checkpoint_dir="checkpoints/pretrain_local", resume=True)
elapsed = time.time() - start
print(f"\nPhase complete in {elapsed/60:.1f} minutes")
print("To continue, change config.total_steps and run again (auto-resumes).")
