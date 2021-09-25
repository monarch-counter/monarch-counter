import argparse

from .utils import run_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default='dataset/preprocessed/512_cropped',
    help='Relative path to the directory containing the preprocessed dataset')
parser.add_argument("--models_dir", default='src/train/out',
    help='Relative path to the directory containing trained models')
parser.add_argument("--save_images", default='False', type=lambda x: (str(x).lower() == 'true'),
    help='Set True if the test run(s) should save the output images on disk. False by default.')
parser.add_argument("--predict_all", default='False', type=lambda x: (str(x).lower() == 'true'),
    help='Set True if you want to generate results for all images. \
        If False, the results for only the latest run are generated. False by default.')

def pretty_print_args(_args):
    print(' '.join(f'{k}={v}\n' for k, v in vars(_args).items()))

if __name__ == "__main__":
    _args = parser.parse_args() 

    pretty_print_args(_args)

    run_predictions(
        dataset_dir=_args.dataset_dir,
        models_dir=_args.models_dir, 
        predict_all=_args.predict_all, 
        save_images=_args.save_images
    )