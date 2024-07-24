import argparse
from PIL import Image
from torchvision import transforms
from tomato_dataset import DatasetFolder
from torch.utils.data import DataLoader
import logging
import warnings
from typing import  Union
from pathlib import Path






def init_logger(log_file: Union[Path, str]) -> None:
    
    warnings.filterwarnings("ignore")
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    # create file handler
    fh = logging.FileHandler(log_file)
    # create console handler
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # clear handlers
    LOGGER.handlers = []
    # add the handlers to the logger
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)
    return LOGGER


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
    

# arguments that can be defined upon execution of the script
def parse_args():
    options = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    options.add_argument('--dataset', default='laboro_tomato', help='folder | laboro_tomato ')
    options.add_argument('--dataroot', default='laboro_tomato/', help='root directory of the dataset')
    options.add_argument('--batchsize', type=int, default=1, help='input batch size')
    options.add_argument('--imagesize', type=int, default=(512,384), help='size of the image (height, width)')
    options.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    opt = options.parse_args()
    return opt


def prepr_data(opt, LOGGER):
    transform=transforms.Compose([transforms.Resize(opt.imagesize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    target_transform = transforms.Compose([transforms.ToTensor()])

    LOGGER.info("Loading data...")

    train_set = DatasetFolder(root= opt.dataroot + "train", loader=pil_loader, extensions=(".png"), transform = transform, target_transform=target_transform)
    test_set = DatasetFolder(root= opt.dataroot + "test", loader=pil_loader, extensions=(".png"), transform = transform, target_transform=target_transform)

    train_data = DataLoader(dataset=train_set, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchsize, shuffle=True, drop_last=True)

    LOGGER.info(f"Training model on {len(train_set)} images.")
    LOGGER.info(f"Testing model on  {len(test_set)} images.")
    LOGGER.info(f"Train/Test ratio: {len(train_set) / len(test_set)}")

    return train_data, test_data