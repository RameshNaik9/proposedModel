from dataset.mnist import MNIST
from dataset.CUB200 import CUB_200
from dataset.ConText import ConText, MakeList, MakeListImage
from dataset.skin_dataset import (
    SkinCancerDataset,
    get_train_val_split,
)
from dataset.transform_func import make_transform
import os
import pandas as pd


def select_dataset(args):
    if args.dataset == "MNIST":
        dataset_train = MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=make_transform(args, "train"),
        )
        dataset_val = MNIST(
            "./data/mnist", train=False, transform=make_transform(args, "val")
        )
        return dataset_train, dataset_val

    if args.dataset == "CUB200":
        dataset_train = CUB_200(
            args, train=True, transform=make_transform(args, "train")
        )
        dataset_val = CUB_200(args, train=False, transform=make_transform(args, "val"))
        return dataset_train, dataset_val

    if args.dataset == "ConText":
        train, val = MakeList(args).get_data()
        dataset_train = ConText(train, transform=make_transform(args, "train"))
        dataset_val = ConText(val, transform=make_transform(args, "val"))
        return dataset_train, dataset_val

    if args.dataset == "SkinCancer":
        # Define paths to image folder and CSV file
        args.dataset_dir = "../data/skin-cancer-mnist-ham10000"
        csv_file = os.path.join(args.dataset_dir, "HAM10000_metadata.csv")
        img_dir = os.path.join(args.dataset_dir, "HAM10000_images_part_*")

        # Read the dataset and perform train-val split
        df_original = pd.read_csv(csv_file)
        df_train, df_val = get_train_val_split(df_original)

        dataset_train = SkinCancerDataset(
            df_train, img_dir, transform=make_transform(args, "train")
        )
        dataset_val = SkinCancerDataset(
            df_val, img_dir, transform=make_transform(args, "val")
        )

        return dataset_train, dataset_val

    if args.dataset == "ImageNet":
        train, val = MakeListImage(args).get_data()
        dataset_train = ConText(train, transform=make_transform(args, "train"))
        dataset_val = ConText(val, transform=make_transform(args, "val"))
        return dataset_train, dataset_val

    raise ValueError(f"unknown {args.dataset}")
