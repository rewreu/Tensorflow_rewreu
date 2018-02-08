# README #

This repo contains a transfer learning implementation, the code is used to train a model based on input images.

### Model training details ###

The train a images classification model base on the image data. The images folder has to be the name of image class.
 The structure will be like:
 ```
images/
        |-- apple_pie/
        │   |-- 1005649.jpg
        │   |-- 101251.jpg
        |-- hot_dog/
        │   |-- 1011328.jpg
  ```
 To have a more effective training, the classes of images
should be <= 1000. After training, two files will be generated, model.pb and model.dict, model.pb is the protobuf 
file contains parameters and operations for tensorflow inferencing. File model.dict contains json translate number 
to string, which uses output from model.pb and point to image class name.
Test on food101 data set yields a top-1 accuracy 64 %. Top-5 accuracy is not collected

### How do I use it? ###

There are two ways to run it:

1. Single image training model:
    
    In this mode, each training step is taking on single image and feed into neural networks for parameters optimization. 
    (This mode is not recommended, it takes much longer time than batch mode)   
    Change BASE_MODEL and IMAGE_DIR according to your local setup. The BASE_MODEL used here is inception_v3,
    You can change the base model to other models and modify input/bottleneck node with the right name inside code.

    ```
    python retrain.singleImage.py

    ```

2. Batch training mode:

    Batch mode takes two steps, first it encodes images into numpy arrays(this takes about 
    1:30h on K80 GPU for food101 dataset). Second, retrain the extra layers attached to the base mode
    with image data.

    1) Encode images from jpg to numpy array, change the setup in encode_images.py and run: 
    ```
    python encode_images.py
    ```
    
    2) retrain models as batch,  change the setup in retrain.py and run: 
    ```
    python retrain.batch.py
    ```
    
3. Run inference on new images:

    Note, this only works with jpg/jpeg file. If you have other type of images, please convert them to jpg/jpeg first.
    The model attached in this repo is trained on food101 dataset.
    ```
    python inference_run.py numberOfClasses imagefilepath
    ----------
    python infer_run.py 5 chickwings.jpg
    ----------
    Input file path is:  chickwings.jpg
    Loading model...
    Finished loading model...
    Prediction results: 
    (chicken wings , probablity is 0.769 )
    (fried calamari , probablity is 0.047 )
    (fish and chips , probablity is 0.033 )
    (bruschetta , probablity is 0.013 )
    (crab cakes , probablity is 0.013 )
    ```
### Notes ###
Make sure you have tensorflow installed, ideally GPU version.


### Who do I talk to if I have questions? ###

* rewreu@gmail.com