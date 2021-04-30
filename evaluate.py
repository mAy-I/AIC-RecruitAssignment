import argparse

from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_csv_path", type=str, default="test.csv")
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--pretrained_weight", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.evaluate()
