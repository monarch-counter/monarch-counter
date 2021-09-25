import os
from os import path

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from ..train.unet import mean_percent_count_err

def __create_dirs(m_name):
    OUT_DIR = path.join(os.getcwd(), 'src', 'predict', 'out')
    out_dirs = os.listdir(OUT_DIR)

    # Parent directory name 
    m_name_dir = m_name.split('.')[0]
    
    # Check if the parent directory exists
    # If not, create it and return True. If it does, return False. 
    if m_name_dir not in out_dirs:
        os.makedirs(path.join(OUT_DIR, m_name_dir, 'pred_imgs'))
        os.makedirs(path.join(OUT_DIR, m_name_dir, 'results'))
        return True
    return False

def __predict(dataset_dir, img_size, m_name, m_path, save_imgs):
    predictions = {
        'images': list(),
        'pred_counts': list(),
        'actual_counts': list()
    }

    input_dims = (img_size, img_size, 3)
    output_dims = (img_size, img_size)
    
    OUT_DIR = path.join(os.getcwd(), 'src', 'predict', 'out')
    SPLITS_LOG_CSV_PATH = path.join(os.getcwd(), 'src', 'train', 'dataset_splits.csv')
    X_IMGS_DIR = path.join(os.getcwd(), dataset_dir, 'cropped_raw')
    Y_IMGS_DIR = path.join(os.getcwd(), dataset_dir, 'density')
    
    m_name_dir = m_name.split('.')[0]
    m_name_ts = m_name.split('__')[0]

    # Get the test set of this model 
    dataset_splits = pd.read_csv(SPLITS_LOG_CSV_PATH)
    entire_dataset = dataset_splits['filenames']
    model_testset = entire_dataset[dataset_splits[m_name_ts] == 2]

    # Forward pass the entire test set 
    custom_metric = {'mean_percent_count_err': mean_percent_count_err}
    with tf.keras.utils.custom_object_scope(custom_metric):
        # Load the model
        model = tf.keras.models.load_model(m_path)

        # For every image in the test set..
        for img in tqdm(model_testset):
            x_img_path = path.join(X_IMGS_DIR, img + '.jpg')
            y_img_path = path.join(Y_IMGS_DIR, img + '.tif')
            
            # Load the input image
            x = np.array(Image.open(x_img_path)) * (1./255)

            # Forward-pass the input image (Run prediction)
            yhat = model(x.reshape(1, *input_dims), training=False)
            yhat_np = yhat.numpy()

            # Save images if necessary 
            if save_imgs:
                yhat_np_reshaped = yhat_np.reshape(output_dims)
                yhat_img = Image.fromarray(yhat_np_reshaped, 'F')

                yhat_img.save(fp=path.join(OUT_DIR, m_name_dir, 'pred_imgs', img + '.tif'))
            
            # Save the prediction result 
            pred_count = yhat_np.sum() / 1000
            
            y = np.array(Image.open(y_img_path))
            actual_count = y.sum() / 1000

            predictions['images'].append(img) 
            predictions['pred_counts'].append(pred_count)
            predictions['actual_counts'].append(actual_count)

    return predictions 

def __generate_results(m_name, m_preds):
    m_name_dir = m_name.split('.')[0]
    OUT_DIR = path.join(os.getcwd(), 'src', 'predict', 'out')
    RESULTS_DIR = path.join(OUT_DIR, m_name_dir, 'results')

    # Create a DataFrame from the dictionary containing predictions
    preds_df = pd.DataFrame(data=m_preds, index=None, columns=m_preds.keys())
    
    # Generate graphs and save them 
    # plt.scatter(list(preds_df.index), preds_df['pred_counts'], label='Predictions')
    # plt.scatter(list(preds_df.index), preds_df['actual_counts'], label='Actual counts')
    # plt.legend()
    # plt.title('Actual and predicted Monarch counts')
    # plt.savefig(fname=path.join(RESULTS_DIR, m_name_dir + '__preds-and-actual.png'))
    # plt.close()

    plt.scatter(preds_df['actual_counts'], preds_df['pred_counts'], s=0.8)
    plt.xlim((0.0, preds_df['actual_counts'].max() + 10.0))
    plt.ylim((0.0, preds_df['actual_counts'].max() + 10.0))
    plt.axline(
        xy1=(0.0, 0.0), 
        xy2=(1.0, 1.0), 
        linestyle='--', color='r', linewidth=0.5)
    
    plt.xlabel('Actual counts')
    plt.ylabel('Predicted counts')
    plt.title('Actual vs. Predicted counts of Monarchs')
    plt.savefig(fname=path.join(RESULTS_DIR, m_name_dir + '__preds-vs-actual.png'))
    plt.close()

    preds_df['percent_error'] = (
        (preds_df['pred_counts'] - preds_df['actual_counts']) 
        / preds_df['actual_counts'].replace(to_replace=0.0, value=1.0)
    ) * 100
    plt.hist(x=preds_df['percent_error'], bins=200, range=(-50, 50))
    plt.title('Percent prediction errors for the test set')
    plt.savefig(fname=path.join(RESULTS_DIR, m_name_dir + '__percent-errors.png'))
    plt.close()

    # Save the prediction results in a csv
    preds_df.to_csv(path.join(RESULTS_DIR, 'counts.csv'), index=False)


def run_predictions(dataset_dir, models_dir, image_size=512, predict_all=False, save_images=False):
    DATASET_DIR = path.join(os.getcwd(), dataset_dir)
    SAVED_MODELS_DIR = path.join(os.getcwd(), models_dir)

    # Get only file names, ignore subdirectories 
    saved_models_list = [
        f for f in os.listdir(SAVED_MODELS_DIR) 
        if path.isfile(path.join(SAVED_MODELS_DIR, f))
    ]

    if predict_all:
        for model_name in saved_models_list:
            # Creates directories to store predictions and results
            # Returns True if newly created, False otherwise.  
            new_model = __create_dirs(m_name=model_name)
            
            # If the results for this model already exist, skip everything  
            if not new_model:
                continue

            # Run predictions for this model
            preds = __predict(
                dataset_dir=DATASET_DIR,
                img_size=image_size,
                m_name=model_name,
                m_path=path.join(SAVED_MODELS_DIR, model_name), 
                save_imgs=save_images
            )

            # Generate results for this model
            __generate_results(m_name=model_name, m_preds=preds)

    else:
        # Find the most recently generated file 
        max_mtime = 0
        most_recent_model = ''
        for model_name in saved_models_list:
            model_full_path = path.join(SAVED_MODELS_DIR, model_name)
            model_mtime = os.stat(model_full_path).st_mtime
            if model_mtime > max_mtime:
                most_recent_model = model_name
                max_mtime = model_mtime
        
        most_recent_model_ts = most_recent_model.split('__')[0]

        for model_name in saved_models_list:
            if most_recent_model_ts in model_name:
                # Creates directories to store predictions and results
                # Returns True if newly created, False otherwise.  
                new_model = __create_dirs(m_name=model_name)
                
                # If the results for this model already exist, skip everything  
                if not new_model:
                    continue
                
                # Run predictions for this model
                preds = __predict(
                    dataset_dir=DATASET_DIR,
                    img_size=image_size,
                    m_name=model_name,
                    m_path=path.join(SAVED_MODELS_DIR, model_name),
                    save_imgs=save_images
                )

                # Generate results for this model
                __generate_results(m_name=model_name, m_preds=preds)
