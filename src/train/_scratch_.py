import os
import shutil
import random
import pandas as pd


def train_test_val_split(dataset_dir,
                         train_fraction=0.7, val_fraction=0.15, test_fraction=0.15):
    """Function to split the dataset in train/validation/test datasets

    Args:
        dataset_dir (string): Relative path to the dataset containing /cropped_raw and /density subdirectories
        train_fraction (float): Fraction for training set
        val_fraction (float): Fraction for validation set
        test_fraction (float): Fraction for test set
    """
    X_PATH = os.path.join(dataset_dir, 'cropped_raw')
    Y_PATH = os.path.join(dataset_dir, 'cropped_bw')

    def __bw_exists(f, _dir):
        name_without_ext = f.split('.')[0]
        for _f in _dir:
            if name_without_ext == _f.split('.')[0]:
                return True
        return False
    
    dirs = [
        'train_x', 'val_x', 'test_x',
        'train_y_bw', 'val_y_bw', 'test_y_bw'
    ]

    # Get the list of X's of all datapoints
    all_x = os.listdir(X_PATH)
    all_y_bw = os.listdir(Y_PATH)
    
    # Shuffle them
    random.shuffle(all_x)

    # Generate train, val and test sets for inputs
    train_split = int(train_fraction * len(all_x))
    val_split = int((train_fraction + val_fraction) * len(all_x))

    train_x_imgs = all_x[:train_split]
    val_x_imgs = all_x[train_split:val_split]
    test_x_imgs = all_x[val_split:]

    train_y_bw_imgs = [f for f in all_y_bw if __bw_exists(f, train_x_imgs)]
    val_y_bw_imgs = [f for f in all_y_bw if __bw_exists(f, val_x_imgs)]
    test_y_bw_imgs = [f for f in all_y_bw if __bw_exists(f, test_x_imgs)]

    print("DEBUG:: Training datapoints - n(X)={}, n(Y)={}".format(len(train_x_imgs), len(train_y_bw_imgs)))
    print("DEBUG:: Validation datapoints - n(X)={}, n(Y)={}".format(len(val_x_imgs), len(val_y_bw_imgs)))
    print("DEBUG:: Testing datapoints - n(X)={}, n(Y)={}".format(len(test_x_imgs), len(test_y_bw_imgs)))

    # Move the images to currosponding folders
    imgs = [
        train_x_imgs, val_x_imgs, test_x_imgs,
        train_y_bw_imgs, val_y_bw_imgs, test_y_bw_imgs
    ]

    for i, d in zip(imgs[0:3], dirs[0:3]):
        print("DEBUG:: Copying the images to {}".format(d))
        # Move all images in i to d
        for f in i:
            shutil.copy(os.path.join(X_PATH, f), os.path.join(dataset_dir, d))

    for i, d in zip(imgs[3:], dirs[3:]):
        print("DEBUG:: Copying the images to {}".format(d))
        # Move all images in i to d
        for f in i:
            shutil.copy(os.path.join(Y_PATH, f), os.path.join(dataset_dir, d))

def main():
    train_test_val_split(dataset_dir='dataset/preprocessed')

if __name__ == "__main__":
    main()