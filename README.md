# Preprocessing steps 

- The `dataset` contains 3 directories
    * `original_images` - the raw original images (a subset of which has been labeled)
    * `labels` - the labels of the images in `csv` format
    * `labeled_images` - empty directory that will be used to separate the images that have been labeled

1. Separate the labeled images from the original images, and place them in the `labeled_images` directory 

```
python src/preproc/separate_labeled_files.py --source "dataset/raw/original_images" --dest "dataset/raw/labeled_images" --labels "dataset/raw/labels"
```

2. Create the images which have black background and white dots at every Monarch butterfly's location

```
python src/preproc/image_process.py --algo "bw" --labels_path "dataset/raw/labels" --dest "dataset/preprocessed/bw_dots" --images_path "dataset/raw/labeled_images"
```

3. Crop input images 

```
python src/preproc/image_process.py --algo "crop" --images_path "dataset/raw/labeled_images" --dest "dataset/preprocessed/512_cropped/cropped_raw" --window 512 --slide 128 --channels 3
```

4. Crop output (bw) images

```
python src/preproc/image_process.py --algo "crop" --images_path "dataset/preprocessed/bw_dots" --dest "dataset/preprocessed/512_cropped/cropped_bw" --window 512 --slide 128 --channels 1
```

5. Augment the data by rotating it 

```
python src/preproc/image_process.py --algo "rotate" --images_path "dataset/preprocessed/512_cropped/cropped_raw" --bw_path "dataset/preprocessed/512_cropped/cropped_bw"
```

6. Create density images 

```
python src/preproc/image_process.py --algo "dens" --images_path "dataset/preprocessed/512_cropped/cropped_bw" --dest "dataset/preprocessed/512_cropped/density" --sigma=30
```

7. (Optional) Compress the images

```
# For compressing the input images 
python src/preproc/image_process.py --algo "compress" --images_path "dataset/preprocessed/1024_cropped/cropped_raw" --dest "dataset/preprocessed/512_compressed/cropped_raw" --compression_factor 2 --image_size 1024

# For compressing the output images
python src/preproc/image_process.py --algo "compress" --images_path "dataset/preprocessed/1024_cropped/density" --dest "dataset/preprocessed/512_compressed/density" --compression_factor 2 --image_size 1024
```