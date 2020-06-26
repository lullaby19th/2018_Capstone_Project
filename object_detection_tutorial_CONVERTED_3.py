# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[ ]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
import matplotlib
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# ## Env setup

# In[ ]:


# This is needed to display the images.
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object detection imports
# Here are the imports from the object detection module.

# In[ ]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[ ]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[ ]:


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[ ]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[ ]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(2, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[ ]:


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# In[ ]:





for image_path in TEST_IMAGE_PATHS:
    # image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = Image.open(image_path)




    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)

    b, g, r = cv2.split(image_np)  # img파일을 b,g,r로 분리
    image_np = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge



    img_x, img_y = image.size
    print("전체 사진 크기: %d x %d" % (img_x, img_y))

    boxes = output_dict['detection_boxes']
    scores = output_dict['detection_scores']

    k = 0
    for i in range(len(scores)):
        if scores[i] >= 0.5:
            k = k + 1

    for t in range(k):
        xmin = boxes[t][1] * img_x
        xmax = boxes[t][3] * img_x
        ymin = boxes[t][0] * img_y
        ymax = boxes[t][2] * img_y

        center = [int((xmax + xmin) / 2), int((ymax + ymin) / 2)]

        z = scores[t] * 100
        int(z)

        print('%d. %d%%의 중심 좌표: (%d,%d)' % (t + 1, z, center[0], center[1]))



    # cv2.imshow("Original Image", cv2.resize(image_np, (800, 600)))

    cv2.imshow("Original Image", image_np)

    #중심좌표 : center[0],center[1]
    cv2.imwrite('image_result.jpg', image_np)

    file = image_path
    img = Image.open(file)

    # 좌표 가보자

    width = img_x
    height = img_y

    print("원하는 피사체의 (x, y) 중심좌표를 입력하라:  ex) 40, 200")
    x_coordinate, y_coordinate = input().split()

    x_coordinate = int(x_coordinate)
    y_coordinate = int(y_coordinate)

    tar_point = [x_coordinate, y_coordinate]

    full_size = [width, height]

    """
    point_1 = [round(width/3), round(height/3)]
    point_2 = [round(2*width/3), round(height/3)]
    point_3 = [round(width/3), round(2*height/3)]
    point_4 = [round(2*width/3), round((2*height/3))]


    x_1 = range(0, round(width/3))
    x_2 = range(round(width/3)+1, round(width/2))
    x_3 = range(round(width/2)+1, round(2*width/3))
    x_4 = range(round(2*width/3)+1, width)

    y_1 = range(0, round(height/3))
    y_2 = range(round(height/3)+1, round(height/2))
    y_3 = range(round(height/2)+1, round(2*height/3))
    y_4 = range(round(2*height/3)+1, width)
    """

    bar = 10
    frame = [10, 10, 10, 10]

    for i in range(738):
        bar = bar + 1

        if tar_point[0] <= round(width / 2) and tar_point[1] <= round(height / 2):

            xmin_2 = tar_point[0] - (1 * bar)
            xmax_2 = tar_point[0] + (2 * bar)
            ymin_2 = tar_point[1] - (1 * bar)
            ymax_2 = tar_point[1] + (2 * bar)

            frame = [xmin_2, xmax_2, ymin_2, ymax_2]


        elif tar_point[0] > round(width / 2) and tar_point[1] <= round(height / 2):

            xmin_2 = tar_point[0] - (2 * bar)
            xmax_2 = tar_point[0] + (1 * bar)
            ymin_2 = tar_point[1] - (1 * bar)
            ymax_2 = tar_point[1] + (2 * bar)

            frame = [xmin_2 , xmax_2, ymin_2, ymax_2]




        elif tar_point[0] <= round(width / 2) and tar_point[1] > round(height / 2):

            xmin_2 = tar_point[0] - (1 * bar)
            xmax_2 = tar_point[0] + (2 * bar)
            ymin_2 = tar_point[1] - (2 * bar)
            ymax_2 = tar_point[1] + (1 * bar)

            frame = [xmin_2 , xmax_2 , ymin_2 , ymax_2]



        elif tar_point[0] > round(width / 2) and tar_point[1] > round(height / 2):

            xmin_2 = tar_point[0] - (2 * bar)
            xmax_2 = tar_point[0] + (1 * bar)
            ymin_2 = tar_point[1] - (2 * bar)
            ymax_2 = tar_point[1] + (1 * bar)

            frame = [xmin_2, xmax_2, ymin_2, ymax_2]

        if frame[0] <= 0 or frame[1] >= width or frame[2] <= 0 or frame[3] >= height:
            break

    area = (xmin_2, ymin_2, xmax_2, ymax_2)
    print(area)
    cropped_image = img.crop(area)
    output = cropped_image.resize((640, 640))
    output.save('result_5.jpg')