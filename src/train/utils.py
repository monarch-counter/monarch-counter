from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
from PIL import Image

import os
from os import path
import shutil
import random

def log_experiment(ts, args):
    """Utility function for logging the experiment

    Args:
        ts (String): Timestamp string that identifies the experiment run
        args (Object): Arguments passed while executing the file
    Returns:
        (Bool): Returns True if a new experiment is being performed, False if otherwise
    """
    EXPERIMENTS_LOG_CSV_PATH = path.join(os.getcwd(), 'src', 'train', 'experiments.csv')

    experiment_csv_df = pd.read_csv(EXPERIMENTS_LOG_CSV_PATH).drop('timestamp', axis='columns')

    experiment_present = experiment_csv_df.isin({
        'batch_size': [args.batch_size],
        'epochs': [args.epochs],
        'learning_rate': [args.lr],
        'dropout_rate': [args.dropout_rate],
        'n_filters': [args.n_filters],
        'unet_block_type': [args.unet_block_type],
        'unet_skip_conn_type': [args.unet_skip_conn_type]
    }).all(axis=1).any()

    if experiment_present:
        return False

    # Create a DataFrame for the new row 
    new_row = [
        [ts, args.batch_size, args.epochs, args.lr, args.dropout_rate, args.n_filters, args.unet_block_type, args.unet_skip_conn_type]
    ]
    new_row_df = pd.DataFrame(new_row)
    
    # Append the newly created DataFrame to the file
    new_row_df.to_csv(path_or_buf=EXPERIMENTS_LOG_CSV_PATH, mode='a', index=False, header=False)

    return True

def train_test_val_split(timestamp, dataset_dir, 
        train_fraction=0.7, val_fraction=0.15, test_fraction=0.15):
    """Function to split the dataset in train/validation/test datasets

    Args:
        dataset_dir (string): Relative path to the dataset containing /cropped_raw and /density subdirectories
        train_fraction (float): Fraction for training set
        val_fraction (float): Fraction for validation set
        test_fraction (float): Fraction for test set
    """
    X_PATH = path.join(dataset_dir, 'cropped_raw')
    Y_PATH = path.join(dataset_dir, 'density')
    
    def __check_args():
        """Helper for sanity check of the arguments

        Raises:
            ValueError: Train, Validation and Test fractions do not add up to 1.0
            ValueError: Dataset directory does not contain /cropped_raw or /density subdirectories
        """
        for d in ['density', 'cropped_raw']:
            if d not in os.listdir(dataset_dir):
                raise ValueError("Dataset path does not contain either /cropped_raw (for input) \
                    or /density (for output) subdirectory")

        if train_fraction + val_fraction + test_fraction != 1.0:
            raise ValueError("Train, Validation and Test fraction must add up to 1")

    def __tiff_exists(f, _dir):
        name_without_tiff = f.split('.')[0]
        for _f in _dir:
            if name_without_tiff == _f.split('.')[0]:
                return True
        return False
    
    def __log_dataset_splits(train, val, test):
        SPLITS_LOG_CSV_PATH = path.join(os.getcwd(), 'src', 'train', 'dataset_splits.csv')
        splits_log = pd.read_csv(SPLITS_LOG_CSV_PATH)

        # Create an empty column for this experiment run 
        splits_log[timestamp] = None

        # Assign the train/val/test encodings to the currosponding rows 
        # train = 0
        # val = 1
        # test = 2
        for f in train:
            splits_log.loc[splits_log['filenames'] == f, timestamp] = 0
        for f in val:
            splits_log.loc[splits_log['filenames'] == f, timestamp] = 1
        for f in test:
            splits_log.loc[splits_log['filenames'] == f, timestamp] = 2

        # Write the modified dataframe back to the csv
        splits_log.to_csv(SPLITS_LOG_CSV_PATH, index=False)

    try:
        __check_args()
    except Exception as e:
        print("ERROR:: Problem with arguments to train_test_val_split")
        raise e

    dirs = [ 
        'train_x', 'val_x', 'test_x',
        'train_y', 'val_y', 'test_y'
    ]

    # Remove directories made earlier along with their contents 
    for d in dirs:
        try:
            shutil.rmtree(path.join(dataset_dir, d))
        except FileNotFoundError as e:
            pass
    
    # Create directories newly
    for d in dirs:
        os.makedirs(path.join(dataset_dir, d))
    

    # Get the list of X's of all datapoints
    all_x = os.listdir(X_PATH)
    all_y = os.listdir(Y_PATH)
    
    # Shuffle them
    random.shuffle(all_x)
    
    # Generate train, val and test sets for inputs
    train_split = int(train_fraction * len(all_x))
    val_split = int((train_fraction + val_fraction) * len(all_x))

    train_x_imgs = all_x[:train_split]
    val_x_imgs = all_x[train_split:val_split]
    test_x_imgs = all_x[val_split:]

    train_y_imgs = [f for f in all_y if __tiff_exists(f, train_x_imgs)]
    val_y_imgs = [f for f in all_y if __tiff_exists(f, val_x_imgs)]
    test_y_imgs = [f for f in all_y if __tiff_exists(f, test_x_imgs)]

    # Log the current train/val/test split in a csv
    __log_dataset_splits(
        train=list(map(lambda x: x.split('.')[0], train_x_imgs)),
        val=list(map(lambda x: x.split('.')[0], val_x_imgs)),
        test=list(map(lambda x: x.split('.')[0], test_x_imgs)),
    )

    print("DEBUG:: Training datapoints - n(X)={}, n(Y)={}".format(len(train_x_imgs), len(train_y_imgs)))
    print("DEBUG:: Validation datapoints - n(X)={}, n(Y)={}".format(len(val_x_imgs), len(val_y_imgs)))
    print("DEBUG:: Testing datapoints - n(X)={}, n(Y)={}".format(len(test_x_imgs), len(test_y_imgs)))

    # Move the images to currosponding folders 
    imgs = [
        train_x_imgs, val_x_imgs, test_x_imgs,
        train_y_imgs, val_y_imgs, test_y_imgs
    ]

    for i, d in zip(imgs[0:3], dirs[0:3]):
        print("DEBUG:: Copying the images to {}".format(d))
        # Move all images in i to d 
        for f in i:
            shutil.copy(path.join(X_PATH, f), path.join(dataset_dir, d))
    
    for i, d in zip(imgs[3:], dirs[3:]):
        print("DEBUG:: Copying the images to {}".format(d))
        # Move all images in i to d 
        for f in i:
            shutil.copy(path.join(Y_PATH, f), path.join(dataset_dir, d))

def get_image_data_generators(dataset_dir="dtaset/preprocessed/512_cropped", img_size=512, seed=69, batch_size=4):
    """Get ImageDataGenerator instances for training and testing data

    Args:
        dataset_dir (str, optional): Path to the parent folder of preprocessed and split data. Defaults to "./dataset/preprocessed/512_cropped".
        seed (int, optional): Random seed to be used in batch sampling. Defaults to 69.
        batch_size (int, optional): Batch size. Defaults to 4.

    Returns:
        [type]: [description]
    """
    # Path variables
    TRAIN_X_PATH = path.join(dataset_dir, 'train_x')
    TRAIN_Y_PATH = path.join(dataset_dir, 'train_y')
    VAL_X_PATH = path.join(dataset_dir, 'val_x')
    VAL_Y_PATH = path.join(dataset_dir, 'val_y')

    input_dims = (img_size, img_size, 3)
    
    # Load a random (representative) sample of training dataset
    train_x_dir = os.listdir(TRAIN_X_PATH)
    random.seed(None)
    random.shuffle(train_x_dir)
    subset_len = int(len(train_x_dir) / 5)
    train_x_subset = train_x_dir[0:subset_len]
    X_subset = np.zeros((subset_len, *input_dims), dtype=float)
    for i, f in enumerate(train_x_subset):
        loaded_img = load_img(path.join(TRAIN_X_PATH, f))
        numpy_img = np.array(loaded_img)
        X_subset[i, ] = numpy_img

    #
    # Data generators for training
    #

    train_datagen_args = dict(
        featurewise_center = True,
        featurewise_std_normalization = True,
        rotation_range = 20,
        width_shift_range = 0.05,
        height_shift_range = 0.05,
        zoom_range = 0.1,
        horizontal_flip = True,
        rescale = 1./255
    )
    __train_x_gen = ImageDataGenerator(**train_datagen_args)
    __train_y_gen = ImageDataGenerator(**train_datagen_args)
    
    __train_x_gen.fit(X_subset, seed=seed)
    
    train_x_gen = __train_x_gen.flow_from_directory(
        TRAIN_X_PATH,
        class_mode=None,
        batch_size=batch_size,
        target_size=(img_size, img_size),
        seed=seed
    )
    train_y_gen = __train_y_gen.flow_from_directory(
        TRAIN_Y_PATH,
        class_mode=None,
        batch_size=batch_size,
        target_size=(img_size, img_size),
        color_mode="grayscale",
        seed=seed
    )
    
    print("DEBUG:: No. of files in train_x generator - {}".format(
        len(train_x_gen.filenames)))
    print("DEBUG:: No. of files in train_y generator - {} (color mode - {})".format(
        len(train_y_gen.filenames), 
        train_y_gen.color_mode))
    train_generator = zip(train_x_gen, train_y_gen)
    
    #
    # Data generators for validation 
    #
    
    val_datagen_args = dict(
        featurewise_center = True,
        featurewise_std_normalization = True,
        rescale = 1./255
    )
    __val_x_gen = ImageDataGenerator(**val_datagen_args)
    __val_y_gen = ImageDataGenerator(**val_datagen_args)

    # Use the same training subset for normalizing validation set
    __val_x_gen.fit(X_subset, seed=seed)

    val_x_gen = __val_x_gen.flow_from_directory(
        VAL_X_PATH,
        class_mode=None,
        target_size=(img_size, img_size),
        seed=seed
    )
    val_y_gen = __val_y_gen.flow_from_directory(
        VAL_Y_PATH,
        class_mode=None,
        target_size=(img_size, img_size),
        color_mode="grayscale",
        seed=seed
    )

    print("DEBUG:: No. of files in val_x generator - {}".format(
        len(val_x_gen.filenames)))
    print("DEBUG:: No. of files in val_y generator - {} (color mode - {})".format(
        len(val_y_gen.filenames), 
        val_y_gen.color_mode))
    val_generator = zip(val_x_gen, val_y_gen)

    return train_generator, val_generator

class DataGenerator(Sequence):
    """Data generator class for training and validation data generation

    Args:
        Sequence (Class): Base class in tf.keras.utils that provides the basic functionality
    """
    def __init__(self, img_size, x_images_path, y_images_path, batch_size):
        """Initialization"""
        self.indexes = None
        self.input_dim = (img_size, img_size)
        self.output_dim = img_size * img_size
        self.batch_size = batch_size
        self.image_path_out = y_images_path
        self.file_out = os.listdir(y_images_path)
        self.image_path_in = x_images_path
        self.file_in = os.listdir(x_images_path)
        self.n_channels = 3
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.file_in) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of input images
        file_names_temp = [self.file_in[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(file_names_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.file_in))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_names_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        y = np.empty((self.batch_size, self.output_dim))

        # Generate data
        for i, name in enumerate(file_names_temp):
            # Store input
            X[i, ] = np.array(Image.open(path.join(self.image_path_in, name))) * (1./255)

            # Store output
            y[i] = np.array(Image.open(path.join(self.image_path_out, name.split('.')[0] + '.tif'))).reshape(-1,
                                                                                                              1).squeeze()

        return X, y