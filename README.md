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

# Deployment
## Connect VSCode <> Azure
1. Download and install `Azure CLI`
2. Install the `Azure Machine Learning` extension in VsCode
3. In the lower blue toolbar, select `Set default Azure ML ..`
4. In the pop up, select `AI for Good` > `train`
5. Finally, the blue toolbar should show: `Azure ML Workspace: train  Azure: <you azure account email>`

## Start Compute Instance
1. Ensure your VsCode is connected to Azure i.e. the lower blue VsCode toolbar should show: `Azure ML Workspace: train  Azure: <you azure account email>`
2. Open `compute_instance.yml`
 - [For a different gpu size](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)
 - Note: If there is an error message about exceeding quota limits, that means we'll have to talk to 
3. Right click file in editor window > `Azure ML: Create Resource`

## Delete Compute Instance
1. [Delete from UI](https://ml.azure.com/compute/list/instances?wsid=/subscriptions/d9399c18-82ea-4a59-9209-2c7dbcd73a7a/resourcegroups/train_group/workspaces/train&tid=ba5a7f39-e3be-4ab3-b450-67fa80faecad)

## Run Training Job
1. Ensure your VsCode is connected to Azure i.e. the lower blue VsCode toolbar should show: `Azure ML Workspace: train  Azure: <you azure account email>`
2. Open `train_counter_job.yml`
3. Right click file in editor window > `Azure ML: Create Resource`
4. Eventually, you'll see the following in the job's logs in the terminal. Follow the instructions to give to the job to download the dataset
    ```
    To sign in, use a web browser to open the page https://microsoft.com/devicelogin and enter the code ANZB7FLHW to authenticate.
    2021/10/03 16:40:06 Not exporting to RunHistory as the exporter is either stopped or there is no data.
    Stopped: false
    OriginalData: 1
    FilteredData: 0.
    ```
5. Once the data finishes downloading, the training will run as expected, and you will continue seeing the logs in the terminal

## Connect Git to Azure Workspace
[Follow these directions](https://docs.microsoft.com/en-us/azure/machine-learning/concept-train-model-git-integration)