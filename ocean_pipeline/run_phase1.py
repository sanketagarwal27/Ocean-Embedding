import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from ocean_pipeline.training.train_phase1 import main as train_main
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline Model Pipeline")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval", action="store_true", help="Only run evaluation against ARGO")
    args = parser.parse_args()

    print("="*64)
    print("  Phase 1: Baseline CNN Encoder + MLP Decoder")
    print("="*64)

    if not args.eval:
        print("\n--- Starting Training ---")
        # Call training
        # Pass args to train_main using sys.argv override or by importing
        sys.argv = [sys.argv[0], "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--lr", str(args.lr)]
        train_main()
        
    print("\n--- Starting Evaluation against ARGO ---")
    eval_script = Path(__file__).parent / "evaluation" / "eval_phase1.py"
    subprocess.run([sys.executable, str(eval_script)])
    
    print("\n" + "="*64)
    print("  Phase 1 Complete")
    print("="*64)

if __name__ == "__main__":
    main()
