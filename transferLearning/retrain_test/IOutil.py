import tensorflow as tf
from tensorflow import gfile
import os
import hashlib
from tensorflow.python.util import compat
import re
import numpy as np

MAX_NUM_IMAGES_PER_CLASS = 1000 ^ 2


def create_image_lists(image_dir, testing_percentage, validation_percentage,
                       load_raw_image = True):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.
      load_raw_image: choose to load from image(jpg, jpeg) or from encoded array file.

    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        if load_raw_image:
            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        else:
            extensions = ['txt']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def encodeRecursive(image_path, encode_path, sess, input, output):
    """recursively encode all images with
    bottle neck vector will be created and saved into another folder.
    Args:
      image_path: the root folder contains all images for encoding
      encode_path: the root folder where the encoded images goes to
      sess: Tensorflow session with right graph loaded
      input: input vector for images, in this case, should be binary bytes of images
      output: endcoded vector comes from model
    """
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    if os.path.isdir(image_path):
        ensure_dir_exists(encode_path)
        for i in os.listdir(image_path):
            encodeRecursive(image_path + "/" + i, encode_path + "/" + i,
                            sess, input, output)
    elif image_path[-3:] in extensions or image_path[-4:] in extensions:
        encode_path = encode_path.replace(".jpg",".txt")\
            .replace(".JPG",".txt").replace(".jpeg",".txt").replace(".JPEG",".txt")
        with open(image_path, "rb") as imagefile:
            bytestring = imagefile.read()
            feed_dict = {input:bytestring}
            bottleneck_values = sess.run([output], feed_dict=feed_dict)
            bottleneck_values = np.squeeze(bottleneck_values)
            bottleneck_string = ','.join(str(x) for x in bottleneck_values)
            with open(encode_path, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)

