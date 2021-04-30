import os
import time

import numpy as np
import torch
import torch.nn as nn

from .model import resnet50
from .data import get_train_loader, get_test_loader


class Trainer(object):
    def __init__(self, args):
        self._args = args
        self._timestamp = str(time.time()).split(".")[0]

        self._LOG_DIR = "results/logs"
        self._LOG_PATH = os.path.join(self._LOG_DIR, f"{self._timestamp}.log")
        self._WEIGHT_DIR = "results/weights"
        self._WEIGHT_PATH = os.path.join(self._WEIGHT_DIR, f"{self._timestamp}.pt")

        self._logging = False
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._train_loader = None
        self._test_loader = None

        self._init_model()

    def _init_model(self):
        """Initialize model."""
        self._model = resnet50(pretrained=False, num_classes=101)
        
        if self._args.pretrained_weight is not None:
            self._model.load_state_dict(torch.load(self._args.pretrained_weight))
            self._write_log(f"Loaded pretrained weight {self._args.pretrained_weight}")
        else:
            self._write_log(f"Initialized model weights.")
        self._model.cuda()

    def _init_optimizer(self):
        """Initialize optimizer."""
        self._optimizer = torch.optim.SGD(self._model.parameters(),
                                            lr=self._args.lr,
                                            weight_decay=self._args.weight_decay)

    def _init_train_loader(self, csv_path, batch_size, num_workers):
        """Initialize train data loader."""
        _, self._train_loader = get_train_loader(csv_path, batch_size, num_workers)

    def _init_test_loader(self, csv_path, batch_size, num_workers):
        """Initialize test data loader."""
        _, self._test_loader = get_test_loader(csv_path, batch_size, num_workers)

    def _train_epoch(self):
        """Train model for an epoch."""
        self._model.train()
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()

        for batch_idx, (_, imgs, labels) in enumerate(self._train_loader):
            props = self._model(imgs.cuda())
            loss = criterion(props, labels.cuda())

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                self._write_log(f"Batch: {batch_idx + 1:04d}/{len(self._train_loader):04d} | " + 
                                f"Loss: {loss:.2f} | Time: {(time.time() - start_time)/60:.2f} mins.")
                start_time = time.time()

    def _write_log(self, msg=""):
        """Write message to a log file and print."""
        if not self._logging:
            print(msg)
            return 0
        if not os.path.exists(self._LOG_DIR):
            os.makedirs(self._LOG_DIR)
        mode = "w" if not os.path.exists(self._LOG_PATH) else "a"
        with open(self._LOG_PATH, mode) as log:
            log.write(f"{msg}\n")
            print(msg)
        return 1

    def _save_model(self):
        """Save model state dictionary."""
        if not os.path.exists(self._WEIGHT_DIR):
            os.makedirs(self._WEIGHT_DIR)
        torch.save(self._model.state_dict(), self._WEIGHT_PATH)
        self._write_log(f"Saved {self._WEIGHT_PATH}.")

    def train(self):
        """Main training loop."""
        self._logging = True
        self._write_log(f"Training log for {self._timestamp}.\n")

        for key, val in self._args.__dict__.items():
            self._write_log(f"{key}: {val}")

        self._init_train_loader(self._args.csv_path, self._args.batch_size, self._args.num_workers)
        self._init_test_loader(self._args.val_csv_path, self._args.batch_size*2, self._args.num_workers)
        self._init_optimizer()

        for epoch in range(self._args.num_epochs):
            self._write_log(f"\nEpoch {epoch + 1}")
            self._train_epoch()
            self.evaluate()
            self._save_model()

    def evaluate(self):
        """Evaluate model performance."""
        self._model.eval()
        if self._test_loader is None:
            self._init_test_loader(self._args.val_csv_path, self._args.batch_size, self._args.num_workers)
        
        correct = list()
        for _, imgs, labels in self._test_loader:
            with torch.no_grad():
                props = self._model(imgs.cuda()).cpu().numpy()
            preds = np.argmax(props, axis=1)
            correct.extend(preds == labels.numpy())
        
        accuracy = np.mean(correct)
        self._write_log(f"Test set accuracy: {accuracy:.4f}")
