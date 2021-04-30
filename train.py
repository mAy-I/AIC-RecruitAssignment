import argparse

from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--csv_path", type=str, default="train.csv")
    parser.add_argument("--val_csv_path", type=str, default="test.csv")
    parser.add_argument("--pretrained_weight", type=str)
    parser.add_argument("--memo", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
