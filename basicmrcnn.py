#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

from torch.utils import data
from torch import Tensor
from typing import Tuple
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
from torch.utils.data import Dataset, DataLoader

import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True


def crop_to_region(coords: Tuple[int], img: Tensor, crop_size: int=42) -> Tensor:
    """ 
    Given coordinates in the form Tuple[int](y, x), return a cropped
    sample of the input imaged centred at (y, x), matching the input size.
    Args:
        coords (Tuple[int]): The input coordinates (y, x) where the crop will be
        centred.
        img (Tensor): The input image, either 3x400x400, 3x250x250, 3x150x150
        crop_size (int, optional): The size of the returned crop. Defaults to 42.

    Returns:
        Tensor: The image cropped with central coordinates at (y, x) of size 
        (3 x size x size) # is size here referring to 42?
    """
    _, H, W = img.shape
    y, x = coords
    y_min, x_min = max(0, y-crop_size//2), max(0, x-crop_size//2)
    y_max, x_max = min(H, y+crop_size//2), min(W, x+crop_size//2)
    region = img[:, y_min:y_max, x_min:x_max]
    if region.shape[1] < crop_size:
        to_pad = crop_size - region.shape[1]
        padding = (0, 0, to_pad, 0) if (y-crop_size//2) < 0 else (0, 0, 0, to_pad)
        region = F.pad(region, padding, mode='replicate')

    if region.shape[2] < crop_size:
        to_pad = crop_size - region.shape[2]
        padding = (to_pad, 0, 0, 0) if (x-crop_size//2) < 0 else (0, to_pad, 0, 0)
        region = F.pad(region, padding, mode='replicate')
    return region

class MIT(data.Dataset):
    def __init__(self, dataset_path: str):
        """
        Given the dataset path, create the MIT dataset. Creates the
        variable self.dataset which is a list of dictionaries with three keys:
            1) X: For train the crop of image. This is of shape [3, 3, 42, 42]. The 
                first dim represents the crop across each different scale
                (400x400, 250x250, 150x150), the second dim is the colour
                channels C, followed by H and W (42x42). For inference, this is 
                the full size image of shape [3, H, W].
            2) y: The label for the crop. 1 = a fixation point, 0 = a
                non-fixation point. -1 = Unlabelled i.e. val and test
            3) file: The file name the crops were extracted from.
            
        If the dataset belongs to val or test, there are 4 additional keys:
            1) X_400: The image resized to 400x400
            2) X_250: The image resized to 250x250
            3) X_150: The image resized to 150x150
            4) spatial_coords: The centre coordinates of all 50x50 (2500) crops
            
        These additional keys help to load the different scales within the
        dataloader itself in a timely manner. Precomputing all crops requires too
        much storage for the lab machines, and resizing/cropping on the fly
        slows down the dataloader, so this is a happy balance.
        Args:
            dataset_path (str): Path to train/val/test.pth.tar
        """
        self.dataset = torch.load(dataset_path, weights_only=True)
        self.mode = 'train' if 'train' in dataset_path else 'inference'
        self.num_crops = 2500 if self.mode == 'inference' else 1

    def __getitem__(self, index) -> Tuple[Tensor, int]:
        """
        Given the index from the DataLoader, return the image crop(s) and label
        Args:
            index (int): the dataset index provided by the PyTorch DataLoader.
        Returns:
            Tuple[Tensor, int]: A two-element tuple consisting of: 
                1) img (Tensor): The image crop of shape [3, 3, 42, 42]. The 
                first dim represents the crop across each different scale
                (400x400, 250x250, 150x150), the second dim is the colour
                channels C, followed by H and W (42x42).
                2) label (int): The label for this crop. 1 = a fixation point, 
                0 = a non-fixation point. -1 = Unlabelled i.e. val and test.
        """
        sample_index = index // self.num_crops
        
        img = self.dataset[sample_index]['X']
        
        # Inference crops are not precomputed due to file size, do here instead
        if self.mode == 'inference': 
            _, H, W = img.shape
            crop_index = index % self.num_crops
            crop_y, crop_x = self.dataset[sample_index]['spatial_coords'][crop_index]
            scales = []
            for size in ['X_400', 'X_250', 'X_150']:
                scaled_img = self.dataset[sample_index][size]
                y_ratio, x_ratio = scaled_img.shape[1] / H, scaled_img.shape[2] / W
                
                # Need to rescale the crops central coordinate.
                scaled_coords = (int(y_ratio * crop_y), int(x_ratio * crop_x))
                crops = crop_to_region(scaled_coords, scaled_img)
                scales.append(crops)
            img = torch.stack(scales, axis=1)
            
        label = self.dataset[sample_index]['y']

        return img, label

    def __len__(self):
        """
        Returns the length of the dataset (length of the list of dictionaries * number
        of crops). 
        __len()__ always needs to be defined so that the DataLoader
            can create the batches
        Returns:
            len(self.dataset) (int): the length of the list of dictionaries * number of
            crops.
        """
        return len(self.dataset) * self.num_crops


trainingdata = MIT("data/train_data.pth.tar")
testingdata = MIT("data/test_data.pth.tar")


# each element in self.dataset dictionary which has three components so just get X and y component (so X component 3x3x42x42 and y is label) -> inputs to the CNN


print(trainingdata.dataset[0]['y'])
print(trainingdata.dataset[0]['X'])

class NormalisedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------------------------------------------------------------------------------------------------------------------------------------------


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
    default=cpu_count(), #note make this 0 if not using lab machine
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
    
    
    # Stack X tensors and extract y values
    X_train = torch.stack([sample['X'] for sample in trainingdata.dataset])
    y_train = torch.tensor([sample['y'] for sample in trainingdata.dataset])

    # Calculate mean and std for normalization
    mean_train = X_train.view(X_train.size(0), X_train.size(1), -1).mean(dim=(0, 2))
    std_train = X_train.view(X_train.size(0), X_train.size(1), -1).std(dim=(0, 2))

    # Normalize X_train
    normalised_X_train = (X_train - mean_train[None, :, None, None]) / std_train[None, :, None, None]

    # Create the custom dataset
    normalised_trainingdata = NormalisedDataset(normalised_X_train, y_train)

    # Pass the dataset into DataLoader
    train_loader = DataLoader(normalised_trainingdata, batch_size=128, shuffle=True)

#     X_train = torch.stack([sample['X'] for sample in trainingdata.dataset]) # stack X tensors to get tensor of shape num_samples, 3, 3, 42, 42 

#     mean_train = X_train.view(X_train.size(0), X_train.size(1), -1).mean(dim=(0,2)) # mean and std across all samples for each channel reshaped to samples, channels=3, height*width 
#     std_train = X_train.view(X_train.size(0), X_train.size(1), -1).std(dim=(0,2))

#     normalised_X_train = (X_train - mean_train[None, :, None, None]) / std_train[None, :, None, None]

#     print(normalised_X_train.shape)

#     normalised_trainingdata = []
#     for i in range(len(trainingdata.dataset)):
#         normalised_sample = {'X':normalised_X_train[i], 'y':trainingdata.dataset[i]['y']}
#         normalised_trainingdata.append(normalised_sample)
    
    
    print(args)
    
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    

    if False:
        train_dataset = None #torchvision.datasets.CIFAR10(args.dataset_root, train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
    else:
        train_dataset = normalised_trainingdata # torchvision.datasets.CIFAR10(args.dataset_root, train=True, download=True, transform=transforms.ToTensor())

    test_dataset = testingdata.dataset #torchvision.datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    
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

        self.initialise_layer(self.conv3)
        
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
                print(logits.shape)
                                        
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
