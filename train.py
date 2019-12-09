import torch
import os
import numpy as np
import argparse
import random
from Processor import Processor

parser = argparse.ArgumentParser()
# Training parameters
parser.add_argument("--batch_size", "-bs", type=int, default=16)
parser.add_argument("--bert_path", "-bp", type=str, default="../chinese_L-12_H-768_A-12/")
parser.add_argument("--text_size", "-ts", type=float, default=0.05)
parser.add_argument("--save_dir", "-sd", type=str, default="checkpoints/")
parser.add_argument("--random_seed", "-rs", type=int, default=0)
parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
parser.add_argument("--dropout_rate", "-dr", type=float, default=0.4)
parser.add_argument("--output_dir", "-od", type=str, default="outputs/result")
# Model parameters
parser.add_argument("--max_len", "-ml", type=int, default=48)
parser.add_argument("--max_len_sent", "-mls", type=int, default=8)
parser.add_argument("--bert_embedding_size", "-bes", type=int, default=768)
parser.add_argument("--pola_class", "-pc", type=int, default=3)
parser.add_argument("--cate_class", "-cc", type=int, default=13)

if __name__ == "__main__":
    args = parser.parse_args()

    # Fix the random seed of package random
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Fix the random seed of Pytorch when using CPU
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_state)

    # Fix the random seed of Pytorch when using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)

    process = Processor(args)
    process.train()