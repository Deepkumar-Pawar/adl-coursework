###################################### IMPORTS

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
from torch.utils.data import Dataset

import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import numpy as np
import statistics

torch.backends.cudnn.benchmark = True

###################################### DATA PROCESSING

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
        # print(sample_index)
        
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
valdata = MIT("data/val_data.pth.tar")
testingdata = MIT("data/test_data.pth.tar")

###################################### DATA AUGMENTATION

training_data_augmented = trainingdata.dataset * 2
for i in range(len(trainingdata.dataset), len(training_data_augmented)):
    training_data_augmented[i]['X'] = torch.flip(training_data_augmented[i]['X'], dims=[3])
    
print('done')

red_channel = ([sample['X'][:, 0, :, :] for sample in training_data_augmented])
blue_channel = ([sample['X'][:, 1, :, :] for sample in training_data_augmented])
green_channel = ([sample['X'][:, 2, :, :] for sample in training_data_augmented])

normalised_training_data = training_data_augmented

def channel_mean(x):
    return (torch.stack(x).mean().item())

def channel_std(x):
    return (torch.stack(x).std().item())

red_mean = channel_mean(red_channel)
blue_mean = channel_mean(blue_channel)
green_mean = channel_mean(green_channel)

red_std = channel_std(red_channel)
blue_std = channel_std(blue_channel)
green_std = channel_std(green_channel)

print(red_mean, blue_mean, green_mean, red_std, blue_std, green_std)

for i in range(0, len(normalised_training_data)): # goes through the current list
    training_data_X = normalised_training_data[i]['X'] # gets the 3x3x42x42 X
    training_data_X[:,0,:,:] = (training_data_X[:,0,:,:] - red_mean) / red_std # gets red channel values and normalises them
    training_data_X[:, 1, :,:] = (training_data_X[:,1,:,:] - blue_mean) / blue_std
    training_data_X[:, 2, :,:] = (training_data_X[:,2,:,:] - green_mean) / green_std
    
    normalised_training_data[i]['X'] = training_data_X
    #print('in loop')

final_data = trainingdata
final_data.dataset = normalised_training_data

###################################### FORMAT FILES
# this code generates the ground truth dictionary for inference datasets indexed by file name (formatted so they end with _fixMap.jpg)
# ensure the predictions dictionary matches this format for file name

def format_file_name_fixMap_jpg(filename):
    return f"{filename[:-5]}_fixMap.jpg"

def get_filenames_fixMap_jpg(dataset: Dataset):
    filenames = [format_file_name_fixMap_jpg(sample['file']) for sample in dataset]

    return filenames

def get_ground_truth_dict(dataset: Dataset):

    filenames = get_filenames_fixMap_jpg(dataset)

    ground_truth_dict = {}
    transform = transforms.ToTensor()

    for filename in filenames:
        ground_truth_dict[filename] = transform(Image.open(f"data/ALLFIXATIONMAPS/ALLFIXATIONMAPS/{filename}")).squeeze() * 255
    
    return ground_truth_dict

val_ground_truth_dict = get_ground_truth_dict(valdata.dataset)
test_ground_truth_dict = get_ground_truth_dict(testingdata.dataset)


###################################### DEFINE AUC
from scipy.integrate import simpson


def roc_auc(pred, target, n_points=20, include_prior=False):
        """
        Calculates the Reciever-Operating-Characteristic (ROC) area under
        the curve (AUC) by numerical integration.
        """

        target = np.array(target)/255
                
        # generated = pred
        # changed above comment line to below one as above was throwing error
        generated = np.array(pred)
        # min max normalisation
        generated = (generated - generated.min())/(generated.max() - generated.min())

        def roc(p=0.1):
            x = generated.reshape(-1) > p
            t = target.reshape(-1) > p

            return np.sum(x==t)/len(t)

        calculate_roc = np.vectorize(roc)

        x = np.linspace(0, 1, n_points)
        auc = simpson(calculate_roc(x))/n_points

        return auc

def calculate_auc(preds, targets):
	"""
	inputs -- 2 dictionary with prediction and target images. The 2 dictionaries have the  same number of keys, where each key identifies a unique image. 
	The predictions have the predicted fixation maps while the targets have the ground truth fixation maps which are available from "https://people.csail.mit.edu/tjudd/WherePeopleLook/" 
	"""
	assert preds.keys() == targets.keys()
	mean_auc = 0
	for key in preds.keys():
		mean_auc += roc_auc(preds[key], targets[key])
	mean_auc /= len(preds.keys())
	return mean_auc


###################################### DEFINE SHUFFLED AUC
import random
#take the ground truth dictionary of 2d maps and build the set of fixation points

def get_fixation_set(gt_dict, threshold):
    fixation_set = []
    for saliency_map in gt_dict.values():
        #print("hello")
        np_map = saliency_map.numpy()
        coords = np.argwhere(np_map >= threshold).astype(float)
        coords[:, 0] /= saliency_map.shape[0]
        coords[:, 1] /= saliency_map.shape[1]
        fixation_set.extend(map(tuple, coords))
    return np.array(fixation_set)

# get the positive set from a single gt map
def get_positive_set(gt_map, threshold):
    positive_set = []
    np_map = gt_map.numpy()
    coords = np.argwhere(np_map >= threshold).astype(float)
    #print(coords)
    positive_set.extend(map(tuple, coords))
    return np.array(positive_set, dtype=int)

# continuously sample the negative set until we get a sufficient amount.
# IMPORTANT: ensure that no samples from the negative set belong to the positive set.
def sample_negative_set(pos, neg):
    pos_set = pos.astype(set)
    pos_len = len(pos)
    neg_len = len(neg)

    samples = []
    #print(pos_len)
    while len(samples) < pos_len:
        sample_index = np.random.choice(neg_len)
        sample_point = neg[sample_index]
        
        if sample_point not in pos_set:
            samples.append(sample_point)

    return np.array(samples)
    #print([228,458] in pos_set)

#return the saliency values given a set of coords
def get_saliency_values(coords, saliency_map):
    return saliency_map[coords[:, 0], coords[:, 1]]

# Shuffled AUC metric: accounts for center bias by sampling the negative set from all other ground truth images
def calculate_shuffled_auc(preds, targets):
    # get GT map and create binary fixation point set
    fixation_set = get_fixation_set(targets, 200)

    s_auc = 0

    for image_name, map in targets.items():

        #rescale normalised fixation points and convert to int to get negative set
        negative_set = np.copy(fixation_set)
        negative_set[:, 0] *= map.shape[0]
        negative_set[:, 1] *= map.shape[1]
        negative_set = negative_set.astype(int)
        
        #get fixation points for positive set
        positive_set = get_positive_set(map, 200)
        #print(positive_set)

        # get the associated prediction map
        s_map = preds[image_name].numpy()
        # s_map = np.random.rand(x.shape[0], x.shape[1])

        #sample the size of positive set from the negative set, as the negative set very large so this reduces computation
        samples = sample_negative_set(positive_set, negative_set)

        # return the saliency values from the coords defined by the pos and neg set
        pos_saliencies = get_saliency_values(positive_set, s_map)
        neg_saliencies = get_saliency_values(samples, s_map)

        #plot the predicted saliencies
        y_true = np.concatenate([np.ones(len(pos_saliencies)), np.zeros(len(neg_saliencies))])
        y_scores = np.concatenate([pos_saliencies, neg_saliencies])

        # Sort by the predicted scores
        sorted_indices = np.argsort(y_scores)[::-1]  # Sorting in descending order
        sorted_scores = y_scores[sorted_indices]
        sorted_labels = y_true[sorted_indices]
        
        # Compute TPR and FPR based on sorted scores
        tp_cumsum = np.cumsum(sorted_labels)  
        fp_cumsum = np.cumsum(1 - sorted_labels)  
        
        # Calculate TPR and FPR
        tpr_values = tp_cumsum / tp_cumsum[-1]  
        fpr_values = fp_cumsum / fp_cumsum[-1]  
        
        # Compute the AUC using the trapezoidal rule
        auc_score = np.trapz(tpr_values, fpr_values)  # Trapezoidal rule

        s_auc += auc_score

    s_auc /= len(targets)
        
    return s_auc
        
# comparing a dict to itself should give a metric of 1
#calculate_shuffled_auc(val_ground_truth_dict, val_ground_truth_dict)

###################################### MODEL CHECKPOINTING
# checkpointing for base model

val_base_model_path = "./model/base/val/base_model.pth"
val_base_auc_path = "./model/base/val/base_auc_score"
test_base_model_path = "./model/base/test/base_model.pth"
test_base_auc_path = "./model/base/test/base_auc_score"

def checkpoint_base_model(model, auc, is_test):
    model_path = ""
    auc_path = ""

    if is_test:
        model_path = test_base_model_path
        auc_path = test_base_auc_path
    else:
        model_path = val_base_model_path
        auc_path = val_base_auc_path

    # scoreboard type checkpointing
    path = Path(auc_path)  # Path to the file you want to check

    if path.exists():
        with open(auc_path, 'r+', encoding='utf-8') as file:
            
            old_auc = float(file.read().strip())

            if auc > old_auc:
                torch.save(model, model_path)
                
                file.truncate(0)
                file.seek(0)
                file.write(str(auc))
    else:
        torch.save(model, model_path)
    
        with open(auc_path, 'w', encoding='utf-8') as file:
            file.write(str(auc))

val_model_path = "./model/val/model.pth"
val_auc_path = "./model/val/auc_score"
test_model_path = "./model/test/model.pth"
test_auc_path = "./model/test/auc_score"

all_models_path = "./data/models/"

def checkpoint_model(model, auc, is_test):
    model_path = ""
    auc_path = ""

    if is_test:
        model_path = test_model_path
        auc_path = test_auc_path

        torch.save(model, f"{all_models_path}model_test_{str(auc)[:5]}.pth")
    else:
        model_path = val_model_path
        auc_path = val_auc_path

        torch.save(model, f"{all_models_path}model_val_{str(auc)[:5]}.pth")

    # scoreboard type checkpointing
    path = Path(auc_path)  # Path to the file you want to check

    if path.exists():
        with open(auc_path, 'r+', encoding='utf-8') as file:
            
            old_auc = float(file.read().strip())

            if auc > old_auc:
                torch.save(model, model_path)
                
                file.truncate(0)
                file.seek(0)
                file.write(str(auc))
    else:
        torch.save(model, model_path)
    
        with open(auc_path, 'w', encoding='utf-8') as file:
            file.write(str(auc))

###################################### TRAIN CNN

from torchviz import make_dot

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=2e-3, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=256,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=50,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=5,
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
parser.add_argument("--sgd-momentum", default=0.9, type=float)
parser.add_argument("--data-aug-hflip", action="store_true")
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--max_norm", default=0.1, type=float)


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):

    for i in range(5):

        train_loader = torch.utils.data.DataLoader(
            final_data,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.worker_count,
        )
        
        print(args)
        
        args.dataset_root.mkdir(parents=True, exist_ok=True)
        
        test_dataset = testingdata.dataset
        
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.worker_count,
            pin_memory=True,
        )

        model = CNN(height=42, width=42, channels=3, class_count=2, dropout=args.dropout)

        # for torchviz visualisation of architecture
        # dummy_input = torch.randn(1, 3, 3, 42, 42)

        # output = model(dummy_input)

        # graph = make_dot(output, params=dict(model.named_parameters()))

        # # Save or render the graph
        # graph.render("cnn_graph_merged", format="png")  # Save as PNG

        # define criterion as Binary Cross Entropy Loss - it does sigmoid as built in
        criterion = nn.BCEWithLogitsLoss()

        # define optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.sgd_momentum, weight_decay=2e-4)

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
        print("Finished training.")
        summary_writer.close()

        print("\nTESTING MODEL")
        test_auc, test_s_auc = test_model(model, testingdata, test_ground_truth_dict, DEVICE)
        print("AUC:", test_auc, "on test dataset\n")
        print("Shuffled AUC:", test_s_auc, "on test dataset\n")
        # save model locally in notebook for manual evaluation
        checkpoint_base_model(model, float(test_auc), True)

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout:float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.dropout = nn.Dropout(p=dropout)
        self.dropout2d = nn.Dropout2d(p=dropout)

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

        # self.fc2 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(1536, 512)

        self.initialise_layer(self.fc2)

        # BatchNorm layer after second FC layer

        self.batchnorm5 = nn.BatchNorm1d(num_features=self.fc2.out_features)
        
        # Define FC 3

        self.fc3 = nn.Linear(512, 1)
        self.initialise_layer(self.fc3)

        # BatchNorm layer after third FC layer
        self.batchnorm6 = nn.BatchNorm1d(num_features=self.fc3.out_features)
    
    # FORWARD METHOD for running 3 parallel CNNs
    def forward(self, images: torch.Tensor) -> torch.Tensor:

        batch_size = images.shape[0]

        images = images.reshape(-1, 3, 42, 42)
        
        x = F.relu(self.conv1(images))
        x = self.pool1(x)
        x = self.batchnorm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.batchnorm2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout2d(x)
        x = self.batchnorm3(x)

        # x = x.reshape(-1, 3, 42, 42)

        # Flatten the output of the pooling layer so it is of shape (batch_size, 4096)
        x = torch.flatten(x, start_dim=1)

        # print(x.shape)
        
        # Pass x through the first fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.batchnorm4(x)


        x = x.reshape(batch_size, -1)


        # x = self.fc2(xCat)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.batchnorm5(x)
        
        x = self.fc3(x)
        #x = self.batchnorm6(x)

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
        
        for epoch in range(start_epoch, epochs):

            self.model.train()
            
            data_load_start_time = time.time()

            epoch_accuracy = 0
            epoch_loss = 0

            for batch, labels in self.train_loader:
                
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                # forward pass
                logits = self.model.forward(batch).squeeze()

                # compute loss using criterion
                loss = self.criterion(logits, labels.float())

                # backward pass
                loss.backward()

                # step optimizer and zero out gradient buffer
                self.optimizer.step()
                self.optimizer.zero_grad()


                
                with torch.no_grad():
                    preds = (logits > 0.5).float()
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

                epoch_accuracy += accuracy
                epoch_loss += loss

            #update the momentum in the optimizer so that it linearly increases from 0.9 to 0.99
            update_momentum(self.model, self.optimizer, epoch, epochs, start_momentum=0.9, end_momentum=0.99)

            # apply weight constraints to limit the magnitude of all weights (should we apply per batch or per epoch?)
            apply_max_norm(self.model, 0.1)

            # log epoch accuracy and epoch loss
            epoch_accuracy /= self.train_loader.batch_size
            epoch_loss /= self.train_loader.batch_size
            self.summary_writer.add_scalar("epoch_accuracy", epoch_accuracy, self.step)
            self.summary_writer.add_scalar("epoch_loss", epoch_loss, self.step)

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                val_auc, val_s_auc = self.validate()
                checkpoint_base_model(self.model, float(val_auc), False)
                # log val auc
                self.summary_writer.add_scalar("val_auc", val_auc, self.step)
                self.summary_writer.add_scalar("val_s_auc", val_s_auc, self.step)
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

        print("\nValidating model...")

        auc, s_auc= test_model(self.model, valdata, val_ground_truth_dict, self.device)

        print("AUC:", auc, "on validation dataset\n")
        print("Shuffled AUC:", s_auc, "on validation dataset\n")
        return auc, s_auc

def update_momentum(self, optimizer, epoch, max_epochs, start_momentum=0.9, end_momentum=0.99):
    momentum = start_momentum + (end_momentum - start_momentum) * (epoch / max_epochs)
    for param_group in optimizer.param_groups:
        param_group['momentum'] = momentum

# Apply weight constraints
def apply_max_norm(model, max_norm):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.clamp_(-max_norm, max_norm)

def test_model(model, dataMIT, ground_truth_dict, device):
    
    model.eval()

    dataset_size = len(dataMIT.dataset)

    batch_size = 50
    num_batches = int(2500 / batch_size)

    preds_dict = {}

    sigmoid = nn.Sigmoid()

    for i in range (dataset_size):

        # potential bug: getting height and width mixed up, so if results look bad, make sure to check this is right

        formatted_filename = format_file_name_fixMap_jpg(dataMIT.dataset[i]['file'])
        height = dataMIT.dataset[i]['X'].shape[1]
        width = dataMIT.dataset[i]['X'].shape[2]

        rescale_to_img = transforms.Resize((height, width))

        img_crops = torch.stack([dataMIT.__getitem__(i * 2500 + j)[0] for j in range(2500)])

        preds_rows = []

        for j in range(num_batches):
            
            batch = img_crops[j * batch_size:(j * batch_size) + batch_size,:,:,:,:]

            batch = batch.to(device)

            preds = sigmoid(model(batch).detach().cpu().reshape(1, 50))
            # preds = (preds > 0.5).float()
            preds_rows.append(preds)


        sal_map = torch.cat(preds_rows, 0)

        sal_map = rescale_to_img(sal_map.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)    # unsqueezing and squeezing done because transforms.Resize requires (., ., H, W) shape tensor
        sal_map = rescale_to_img(sal_map) * 255


        preds_dict[formatted_filename] = sal_map

    auc = calculate_auc(preds_dict, ground_truth_dict)
    s_auc = calculate_shuffled_auc(preds_dict, ground_truth_dict)
    return auc, s_auc


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
    args = parser.parse_args()
    main(args)


######################################
######################################
######################################
######################################
