"""
Training script for InfiniteDance
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch.multiprocessing as mp
from models.arguments import get_parser
from models.train_utils import train_llama_music2dance
from models.motion import load_vqvae_model


def main():
    parser = get_parser()  
    args = parser.parse_args()
    
    print(f"Output directory: {args.out_dir}")
    print(f"Dance directory: {args.dance_dir}")
    print(f"Batch size: {args.batch_size}")

    print("Loading VQVAE model...")
    net, codebooks, opt = load_vqvae_model(checkpoint_path=args.vqvae_checkpoint_path)
    print("VQVAE model loaded successfully.")

    mp.spawn(
        train_llama_music2dance,
        args=(args.world_size, args, codebooks, args.MASTER_PORT),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
