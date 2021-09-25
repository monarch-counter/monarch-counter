"""generate black image with white dots based on csv labels"""
import argparse
import os
import pandas as pd
from PIL import Image, ImageOps, ExifTags
import sys
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import argparse
from cv2 import imread, imencode, imwrite
from os import walk, path
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--algo', help='one of bw,pad,dens,test_int,crop,rotate')
parser.add_argument('--labels_path', help='location of label csv-s')
parser.add_argument('--dest', help='location of black images with white dots')
parser.add_argument('--images_path', help='location of images')
parser.add_argument('--sigma', help='sigma for gaussian filter')
parser.add_argument('--unprocessed_path', help='location of unprocessed images that need padding or sorting')
parser.add_argument('--processed_path', help='')
parser.add_argument('--bw_path', help='black images with white dots')
parser.add_argument('--height', help='height of processed images')
parser.add_argument('--width', help='width of processed images')
parser.add_argument('--min', help='minimum height or width needed')
parser.add_argument('--window', help='sliding window dimension for cropping')
parser.add_argument('--slide', help='sliding window slide dimension from cropping')
parser.add_argument('--channels', help='number of channels in an image')
parser.add_argument('--source_format', help='format in source folder')
parser.add_argument('--dest_format', help='format in dest folder')
parser.add_argument('--compression_factor', help='Factor by which the input images will be compressed', type=int)
parser.add_argument('--image_size', help='Original size of the images', type=int)

args = parser.parse_args()

'''define required arguments for tasks'''
required = {
    'bw': [args.labels_path, args.dest, args.images_path],
    'pad': [args.unprocessed_path, args.processed_path, args.height, args.width, args.min],
    'dens': [args.images_path, args.dest],
    'test_int': [args.images_path, args.bw_path],
    'crop': [args.images_path, args.dest, args.window, args.slide, args.channels],
    'rotate': [args.images_path, args.bw_path],
    'check': [args.unprocessed_path, args.processed_path, args.source_format, args.dest_format],
    'compress': [args.compression_factor, args.images_path, args.dest, args.image_size]
}


def check_args(args):
    """check if required arguments are present"""
    if args.algo == None:
        print('Argument algo must be present')
        sys.exit()

    for arg in required[args.algo]:
        if arg is None:
            print('Required argument missing for algo -- {}. Check necessary arguments'.format(args.algo))
            sys.exit()


def call_algo(args):
    """call method based on task"""
    if args.algo == 'bw':
        generate_black_white(args)
    elif args.algo == 'pad':
        pad_sort(args)
    elif args.algo == 'dens':
        generate_density(args)
    elif args.algo == 'test_int':
        test_integrate(args)
    elif args.algo == 'crop':
        slide_crop(args)
    elif args.algo == 'rotate':
        rotate(args)
    elif args.algo == 'check':
        check_match_files(args)
    elif args.algo == 'compress':
        compress(args)


def generate_black_white(args):
    """generate black images with white dots based on csv labels"""
    label_files = [f for f in os.listdir(args.labels_path) if 'csv' in f]

    # create empty dataframe and append
    df_labels = pd.DataFrame(columns=['label', 'x', 'y', 'file', 'a', 'b'])
    for fi in label_files:
        with open(args.labels_path + '/' + fi) as f:
            df_temp = pd.read_csv(f, header=None, names=['label', 'x', 'y', 'file', 'a', 'b'])
            df_labels = df_labels.append(df_temp)

    # print("DEBUG:: No. of files: {}".format(df_labels['file'].unique().shape[0]))

    # loop through each image, create a white image of the same dimension and mark black dots for labels
    images = df_labels['file'].unique()
    for f in images:
        try:
            im = Image.open(args.images_path + '/' + f)

            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            try:
                exif = dict(im._getexif().items())
            except AttributeError as err:
                # print("ERROR:: {} with file {}".format(err, f))
                pass

            x, y = im.size
            labels = df_labels.loc[df_labels['file'] == f]
            im_new = Image.new(size=(x, y), mode='P')
            for _, row in labels.iterrows():
                im_new.putpixel(xy=(row['x'], row['y']), value=255)
            
            print("DEBUG:: No. of white pixel in {} - {}".format(
                f,
                np.count_nonzero(np.asarray(im_new.getdata())) 
            ))

            try:
                rotate_problem = False
                if exif[orientation] == 3:
                    rotate_problem = True
                    im_new = im_new.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    rotate_problem = True
                    im_new = im_new.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    rotate_problem = True
                    im_new = im_new.rotate(90, expand=True)

                if rotate_problem:
                    print('DEBUG:: Image {} has rotation issues. Fixed'.format(f))

            except (AttributeError, KeyError, IndexError, UnboundLocalError) as err:
                # print("ERROR:: {} with file {}".format(err, f))
                pass

            im_new.save(args.dest + '/' + f.split('.')[0] + '.png')

        except FileNotFoundError:
            continue
        except IndexError:
            continue

        # print("\n")


def pad_sort(args):
    """pad images to required dimension if smaller"""
    raw_images = os.listdir(args.unprocessed_path)
    for fi in raw_images:
        try:
            im = Image.open(args.unprocessed_path + '/' + fi)
        except FileNotFoundError:
            continue
        except OSError:
            print('OS Error with file {}'.format(fi))
            continue

        x, y = im.size
        if x < float(args.min) or y < float(args.min):
            continue

        delta_h = (int(args.height) - y) if int(args.height) > y else 0
        delta_w = (int(args.width) - x) if int(args.width) > x else 0
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(im, padding)
        img.save(args.processed_path + '/' + fi)


def generate_density(args):
    """generate tiff density maps by gaussian convolution from bw images"""
    bw_images = os.listdir(args.images_path)
    for fi in bw_images:
        try:
            im = Image.open(args.images_path + '/' + fi)
        except OSError:
            print('OS Error with file {}'.format(fi))
            continue

        pix = np.array(im)
        x, y = im.size
        im_dens = np.zeros((y, x))

        for i in range(y):
            for j in range(x):
                if pix[i, j] > 0:
                    im_dens[i, j] = 1000

        s = float(args.sigma)
        result = gaussian_filter(im_dens, sigma=(s, s), order=0)
        i = Image.fromarray(result)
        i.save(args.dest + '/' + fi.split('.')[0] + '.tif')


def test_integrate(args):
    """test if density maps integrate to actual counts"""
    dens_images = os.listdir(args.images_path)
    count_d, count_b = [], []
    for fi in dens_images:
        try:
            im_d = Image.open(args.images_path + '/' + fi)
            im_bw = Image.open(args.bw_path + '/' + fi.split('.')[0] + '.png')
        except OSError:
            print('OS Error with file {}'.format(fi))
            continue

        pix_d = np.array(im_d)
        count_d.append(np.sum(pix_d) / 1000)

        pix_b = np.array(im_bw)
        count_b.append(np.count_nonzero(pix_b))

        plt.figure()
        plt.scatter(count_b, count_d)
        plt.savefig('sanity_dens_int.jpg')

def __sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def slide_crop(args):
    """crop images based on a sliding window"""
    img_names = []
    imgs_colored = []

    # Window size for cropping
    (win_w, win_h) = (int(args.window), int(args.window))
    # Step size for cropping
    step_size = int(args.slide)

    # Image preproc

    # Read the names of images in the input dir
    for (dirpath, dirnames, filenames) in walk(args.images_path):
        img_names.extend(filenames)

    # Read the images using opencv (in colored and greyscaled formats)
    # and save them in a list
    for img_name in img_names:
        in_img_full_path = args.images_path + '/' + img_name
        try:
            if int(args.channels) == 1:
                imgs_colored.append((img_name, np.array(Image.open(in_img_full_path))))
            else:
                imgs_colored.append((img_name, imread(in_img_full_path)))
        except OSError:
            continue

    # Crop the colored images and save them
    for img_name, img in imgs_colored:
        try:
            for (x, y, window) in __sliding_window(img, stepSize=step_size, windowSize=(win_w, win_h)):
                if window.shape[0] != win_h or window.shape[1] != win_w:
                    continue
                img_str = imencode('.jpg', window)[1].tobytes()

                # imshow("window", window)
                # waitKey(1)
                if int(args.channels) == 1:
                    imwrite(args.dest + "/" + img_name.split('.')[0] + '__' + str(x) + '_' + str(y) + '__' + '.png',
                            window)
                else:
                    imwrite(args.dest + "/" + img_name.split('.')[0] + '__' + str(x) + '_' + str(y) + '__' + '.jpg',
                            window)

        except AttributeError:
            print('Error with filename {}'.format(img_name))
            pass


def rotate(args):
    """generate rotated images, and corresponding bw images"""
    rotate_angles = [90, 180, 270]
    raw_images = os.listdir(args.images_path)
    for fi in raw_images:
        try:
            im = Image.open(args.images_path + '/' + fi)
            im_bw = Image.open(args.bw_path + '/' + fi.split('.')[0] + '.png')
        except OSError:
            print('OS error with file {}'.format(fi))
            continue

        for a in rotate_angles:
            im_r = im.rotate(a, expand=True, resample=Image.BICUBIC)
            im_r.save(args.images_path + '/' + fi.split('.')[0] + 'r_' + str(a) + '.jpg')
            im_bw_r = im_bw.rotate(a, expand=True, resample=Image.BICUBIC)
            im_bw_r.save(args.bw_path + '/' + fi.split('.')[0] + 'r_' + str(a) + '.png')


def check_match_files(args):
    """check that for each raw cropped image, a density exists"""
    source = os.listdir(args.unprocessed_path)
    dest = os.listdir(args.processed_path)
    mismatch_count = 0
    for i in source:
        if i.split('.')[0] + '.' + args.dest_format not in dest:
            mismatch_count += 1
            print('File {} does not match'.format(i))

    print('Total Mismatch Count : {}'.format(mismatch_count))


def compress(args):
    """Compresses given images in the input dir by the compression ratio, and saves it to a destination dir"""
    INPUT_IMAGES_PATH = path.join(os.getcwd(), args.images_path)
    OUTPUT_IMAGES_PATH = path.join(os.getcwd(), args.dest)

    images = [f for f in os.listdir(INPUT_IMAGES_PATH) if path.isfile(path.join(INPUT_IMAGES_PATH, f))]

    output_size = (int(args.image_size / args.compression_factor), int(args.image_size / args.compression_factor))

    for image in tqdm(images):
        image_path = path.join(INPUT_IMAGES_PATH, image)

        img_obj = Image.open(image_path)
        resized_img_obj = img_obj.resize(output_size)

        resized_img_obj.save(path.join(OUTPUT_IMAGES_PATH, image))


if __name__ == '__main__':
    if __package__ is None:
        from os import sys, path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    check_args(args)
    call_algo(args)
