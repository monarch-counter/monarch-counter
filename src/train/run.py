import argparse
from datetime import datetime
import os
from os import path

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from .utils import train_test_val_split, log_experiment, DataGenerator
from .unet import get_unet

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="dataset/preprocessed/512_cropped",
    help="Relative path to directory that contains the preprocessed dataset. \
        Defaults to dataset/preprocessed/512_cropped")
parser.add_argument("--image_size", default="512", type=int,
    help="Image size (height and width) of input and output images in terms of pixel counts. \
        E.g. if set to 128, model expects input and output images of 128x128 pixels. Defaults to 512.")

parser.add_argument("--batch_size", default="8", type=int,
    help="Batch size for training")
parser.add_argument("--epochs", default="50", type=int,
    help="Number of epochs for training")
parser.add_argument("--lr", default="5e-5", type=float,
    help="Learning rate of the optimizer")
parser.add_argument("--dropout_rate", default="0.0", type=float,
    help="Dropout rate for the Conv blocks. Starts off with this rate and keeps on increasing for the deeper blocks.")

parser.add_argument("--n_filters", default="16", type=int,
    help="No of filters (kernels) the model starts with in the first Conv layer")
parser.add_argument("--unet_block_type", default="default",
    help="Type of encoder and decoder blocks in the UNet. Can be 'default', 'multires' or 'dilated_multires'.")
parser.add_argument("--unet_skip_conn_type", default="default",
    help="Type of residual path used in the UNet. Can be 'default' or 'resnet'.")

def pretty_print_args(_args):
    print(' '.join(f'{k}={v}\n' for k, v in vars(_args).items()))


if __name__ == "__main__":
    _args = parser.parse_args() 
 
    # Create a unique timestamp
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Log this run of the experiment
    new_experiment = log_experiment(ts, _args)

    if new_experiment:
        print("New experiment with params -")
        pretty_print_args(_args)

        # Split the data 
        train_test_val_split(timestamp=ts, dataset_dir=_args.dataset_dir)

        # Create data generator instances to pass data to the model
        training_generator = DataGenerator(
            img_size=_args.image_size,
            x_images_path=path.join(os.getcwd(), _args.dataset_dir, 'train_x'),
            y_images_path=path.join(os.getcwd(), _args.dataset_dir, 'train_y'),
            batch_size=_args.batch_size
        )
        validation_generator = DataGenerator(
            img_size=_args.image_size,
            x_images_path=path.join(os.getcwd(), _args.dataset_dir, 'val_x'),
            y_images_path=path.join(os.getcwd(), _args.dataset_dir, 'val_y'),
            batch_size=_args.batch_size
        )

        # Create checkpoints to save the best performing models
        val_error_checkpoint = ModelCheckpoint(
            filepath=path.join(os.getcwd(), 'src', 'train', 'out', ts + "__least_val_error.hdf5"),
            monitor='val_mean_percent_count_err',
            save_best_only=True,
            mode='min'
        )

        val_loss_checkpoint = ModelCheckpoint(
            filepath=path.join(os.getcwd(), 'src', 'train', 'out', ts + "__least_val_loss.hdf5"),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )

        early_stopping_checkpoint = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            restore_best_weights=True
        )

        # Setup tensorboard 
        log_dir = "logs/fit/" + ts
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            update_freq=5,
            histogram_freq=5
        )

        # Create a model instance and train
        _model = get_unet(
            n_filters=_args.n_filters, 
            lr=_args.lr, 
            dropout_prob=_args.dropout_rate,
            block_type=_args.unet_block_type,
            skip_connection_type=_args.unet_skip_conn_type
        )
        _model.fit(
            x=training_generator,
            callbacks=[
                val_error_checkpoint,
                val_loss_checkpoint,
                # early_stopping_checkpoint, 
                tensorboard_callback
            ],
            epochs=_args.epochs,
            validation_data=validation_generator,
            shuffle=True,
            validation_freq=1,
            max_queue_size=10,
            workers=4,
            use_multiprocessing=True
        )
    else:
        print("Experiment already performed with params -")
        pretty_print_args(_args)
