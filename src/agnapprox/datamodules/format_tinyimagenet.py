#!/usr/bin/env python3

import argparse
import os


def format_imagenet_val_set(home_folder):
    val_folder = os.path.join(home_folder, "val")
    val_img_dict = {}
    with open(os.path.join(val_folder, "val_annotations.txt"), "r") as f:
        for line in f:
            words = line.split("\t")
            val_img_dict[words[0]] = words[1]

    for img, folder in val_img_dict.items():
        newpath = os.path.join(val_folder, folder)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_folder, "images", img)):
            os.rename(
                os.path.join(val_folder, "images", img), os.path.join(newpath, img)
            )

    os.rmdir(os.path.join(val_folder, "images"))
    os.remove(os.path.join(val_folder, "val_annotations.txt"))


if __name__ == "__main__":

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

    parser = argparse.ArgumentParser(
        description="Move TinyImageNet Validation Set to folder structure"
    )
    parser.add_argument("--path", type=dir_path, help="TinyImageNet Home Directory")
    args = parser.parse_args()
    format_imagenet_val_set(args.path)
