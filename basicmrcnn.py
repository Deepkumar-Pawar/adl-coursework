#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=2e-4, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=0, #cpu_count(), #note make this 0 if not using lab machine
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument("--sgd-momentum", default=0, type=float)
parser.add_argument("--data-aug-hflip", action="store_true")



class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    print(args)
    
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    

    if False:
        train_dataset = None #torchvision.datasets.CIFAR10(args.dataset_root, train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
    else:
        train_dataset = train_data # torchvision.datasets.CIFAR10(args.dataset_root, train=True, download=True, transform=transforms.ToTensor())

    test_dataset = test_data #torchvision.datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    model = CNN(height=42, width=42, channels=3, class_count=2)

    ## TASK 8: Redefine the criterion to be softmax cross entropy
    criterion = nn.CrossEntropyLoss()

    ## TASK 11: Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.sgd_momentum)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    print("About to train...")
    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )
    print("Finsished training.")
    summary_writer.close()


class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        # Define 1st Conv, Pool, and BatchNorm
        
        # Conv 1
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=96,
            kernel_size=(7, 7),
            padding= (0,0), #(2, 2),
        )
        self.initialise_layer(self.conv1)
        
        # Pool 1
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # BatchNorm 1
        self.batchnorm1 = nn.BatchNorm2d(self.conv1.out_channels)

        # Define 2nd Conv, Pool, and BatchNorm
        
        # Conv 2
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=160,
            kernel_size=(3, 3),
            padding= (0,0) #(2, 2)
        )

        self.initialise_layer(self.conv2)
        
        # Pool 2
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # BatchNorm 2
        self.batchnorm2 = nn.BatchNorm2d(num_features=self.conv2.out_channels)
        
        # Define 3rd Conv, Pool, and BatchNorm
        
        # Conv 3
        self.conv3 = nn.Conv2d(
            in_channels=160,
            out_channels=288,
            kernel_size=(3, 3),
            padding= (0,0) #(2, 2)
        )

        self.initialise_layer(self.conv2)
        
        # Pool 3
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # BatchNorm 3
        self.batchnorm3 = nn.BatchNorm2d(num_features=self.conv3.out_channels)

        
        # Define FC 1

        self.fc1 = nn.Linear(2592, 512)
        self.initialise_layer(self.fc1)

        # BatchNorm layer after first FC layer
        # import pdb; pdb.set_trace()
        self.batchnorm4 = nn.BatchNorm1d(num_features=self.fc1.out_features)

        
        # Define FC 2

        self.fc2 = nn.Linear(1536, 512)
        self.initialise_layer(self.fc2)

        # BatchNorm layer after second FC layer

        self.batchnorm5 = nn.BatchNorm1d(num_features=self.fc2.out_features)
        
        # Define FC 3

        self.fc3 = nn.Linear(512, 1)
        self.initialise_layer(self.fc3)

        # BatchNorm layer after third FC layer
        self.batchnorm6 = nn.BatchNorm1d(num_features=self.fc3.out_features)

    #FORWARD FOR combining first 2 dimensions
#     def forward(self, images: torch.Tensor) -> torch.Tensor:
#         images = images.reshape(-1, 3, 42, 42)
#         #images = images[:, 0, :, :, :]
#         print(images.shape)
#         x = F.relu(self.conv1(images))
#         x = self.pool1(x)
#         x = self.batchnorm1(x)

#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = self.batchnorm2(x)
        
#         x = F.relu(self.conv3(x))
#         x = self.pool3(x)
#         x = self.batchnorm3(x)

#         ## TASK 4: Flatten the output of the pooling layer so it is of shape
#         ##         (batch_size, 4096)
        
#         x = torch.flatten(x, start_dim=1)
        
        
#         ## TASK 5-2: Pass x through the first fully connected layer
        
#         x = F.relu(self.fc1(x))
#         x = self.batchnorm4(x)
        
#         print("fc1 layer shape: ", x.shape)
        
#         ## TASK 6-2: Pass x through the last fully connected layer
#         #IMPORTANT: concatenate fc layers before next steps

#         x = self.fc2(x)
#         x = self.batchnorm5(x)
        
#         x = self.fc3(x)
#         x = self.batchnorm6(x)

#         return x
    
    # FORWARD METHOD for running 3 parallel CNNs
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        #images = images.reshape(-1, 3, 42, 42)
        images1 = images[:, 0, :, :, :]
        
        #pass first resolutions through the network
        x1 = F.relu(self.conv1(images1))
        x1 = self.pool1(x1)
        x1 = self.batchnorm1(x1)

        x1 = F.relu(self.conv2(x1))
        x1 = self.pool2(x1)
        x1 = self.batchnorm2(x1)
        
        x1 = F.relu(self.conv3(x1))
        x1 = self.pool3(x1)
        x1 = self.batchnorm3(x1)

        # Flatten the output of the pooling layer so it is of shape (batch_size, 4096)
        x1 = torch.flatten(x1, start_dim=1)
        
        # Pass x through the first fully connected layer
        x1 = F.relu(self.fc1(x1))
        x1 = self.batchnorm4(x1)
        
        images2 = images[:, 1, :, :, :]
        
        #pass second resolutions through the network
        x2 = F.relu(self.conv1(images2))
        x2 = self.pool1(x2)
        x2 = self.batchnorm1(x2)

        x2 = F.relu(self.conv2(x2))
        x2 = self.pool2(x2)
        x2 = self.batchnorm2(x2)
        
        x2 = F.relu(self.conv3(x2))
        x2 = self.pool3(x2)
        x2 = self.batchnorm3(x2)

        # Flatten the output of the pooling layer so it is of shape (batch_size, 4096)
        x2 = torch.flatten(x2, start_dim=1)
        
        # Pass x through the first fully connected layer
        x2 = F.relu(self.fc1(x2))
        x2 = self.batchnorm4(x2)
        
        
        images3 = images[:, 2, :, :, :]
        
        #pass third resolutions through the network
        x3 = F.relu(self.conv1(images3))
        x3 = self.pool1(x3)
        x3 = self.batchnorm1(x3)

        x3 = F.relu(self.conv2(x3))
        x3 = self.pool2(x3)
        x3 = self.batchnorm2(x3)
        
        x3 = F.relu(self.conv3(x3))
        x3 = self.pool3(x3)
        x3 = self.batchnorm3(x3)

        # Flatten the output of the pooling layer so it is of shape (batch_size, 4096)
        x3 = torch.flatten(x3, start_dim=1)
        
        # Pass x through the first fully connected layer
        x3 = F.relu(self.fc1(x3))
        x3 = self.batchnorm4(x3)
        
        
        print("fc1 layer shape: ", x3.shape)
        
        ## TASK 6-2: Pass x through the last fully connected layer
        #IMPORTANT: concatenate fc layers before next steps
        xCat = torch.cat((x1, x2, x3), dim=1)
        x = self.fc2(xCat)
        x = self.batchnorm5(x)
        
        x = self.fc3(x)
        x = self.batchnorm6(x)

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        

        
        #import sys; sys.exit(1)
        for epoch in range(start_epoch, epochs):
            self.model.train()
            
            data_load_start_time = time.time()
             #stuck after this
            for batch, labels in self.train_loader:
                
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()


                ## TASK 1: Compute the forward pass of the model, print the output shape
                ##         and quit the program
                logits = self.model.forward(batch)
                                        
                import sys; sys.exit(1)

                ## TASK 7: Rename `output` to `logits`, remove the output shape printing
                ##         and get rid of the `import sys; sys.exit(1)`

                # Task 7 done above

                ## TASK 9: Compute the loss using self.criterion and
                ##         store it in a variable called `loss`

                loss = self.criterion(logits, labels)

                ## TASK 10: Compute the backward pass

                loss.backward()

                ## TASK 12: Step the optimizer and then zero out the gradient buffers.
                
                self.optimizer.step()

                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
      f"CNN_bn_"
      f"bs={args.batch_size}_"
      f"lr={args.learning_rate}_"
      f"momentum=0.9_" +
      ("hflip_" if args.data_aug_hflip else "") +
      f"run_"
  )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    main(args)
