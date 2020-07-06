# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# ## Env setup

# This is needed to display the images.
#get_ipython().run_line_magic('matplotlib', 'inline')

# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

#%%

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

#%%
# Download Model

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

#%%
# Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#%%
# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#%%
# Helper code
# img를 numpy array 형태로 바꾸어주는 코드

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


#%%
# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(37, 38)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

#%%
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

#%%

def find_bounding_boxes(image, graph):
  with graph.as_default():
      with tf.Session() as sess:
          width, height = image.size
          boxes = graph.get_tensor_by_name('detection_boxes:0')
          np.squeeze(boxes)
          ymin = boxes[0][1][0] * height
          xmin = boxes[0][1][1] * width
          ymax = boxes[0][1][2] * height
          xmax = boxes[0][1][3] * width
          print('Top left')
          print(xmin, ymin, )
          print('Bottom right')
          print(xmax, ymax)


#%%
# 피사체의 중심좌표와 박스좌표를 알기위한 빈 list 선언        
          
# PATH_TO_TEST_IMAGES_DIR = 'test_images'
          
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2)]

# Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)
          
for image_path in TEST_IMAGE_PATHS:
  #image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
  img = Image.open(image_path)


  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(img)



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



  #plt.figure(figsize=IMAGE_SIZE)
  #plt.imshow(image_np)
  
  # 여기 코드는 자체 제작.
  b, g, r = cv2.split(image_np)  # img파일을 b,g,r로 분리
  image_np = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
  img_x, img_y = img.size
  print("전체 사진 크기: %d x %d" %(img_x,img_y)) 



  boxes = output_dict['detection_boxes']
  scores = output_dict['detection_scores']

  center_dic = {}   #빈 사전 선언 center_dic은 박스 순서대로 중심좌표 등록
  box_dic = {}     # box_dic은 박스 순서대로 박스의 x, y 의 min , max 4개의 값 등록
  
  # score가 0.5(50%) 이상인 피사체를 검출 된 피사체로 선정.
  k=0
  for i in range(len(scores)):
      if scores[i] >= 0.5:
          k= k+1


  for t in range(k):

      ymin = boxes[t][0] * img_y
      xmin = boxes[t][1] * img_x
      ymax = boxes[t][2] * img_y
      xmax = boxes[t][3] * img_x
      print(ymin, xmin, ymax, xmax)

      center = [int((xmax+xmin)/2), int((ymax+ymin)/2)]

      z = scores[t]*100
      int(z)

      center_dic[t+1] = [center[0], center[1]]
      box_dic[t+1] = [ymin, xmin, ymax, xmax]

      print('%d. %d%%의 중심 좌표: (%d,%d)'%(t+1, z, center[0],center[1]))

      # cv2.imshow("Original Image", cv2.resize(image_np, (800, 600)))
  cv2.imshow("Original Image", image_np)
  cv2.imwrite('resultimage/detection_image/image_result.jpg', image_np)

  key = cv2.waitKey(0)
  if key == ord("q") or key == 27:
      exit()


# 여기가 피사체 중심좌표, 박스 좌표 찾는곳
  print('피사체 선택하라( 여러 피사체의 중심좌표를 알고 싶으면 2개 선택):  ex)1    ex)3 4)
  a = input().split()    # a로 입력받은 값을 입력 갯수만큼 리스트로 저장함.
  r = float(len(a))            # a 리스트의 크기를 가져옴 -> 피사체 선택 갯수라고 보면 된다.



# 중심좌표와 박스좌표를 알기위한 빈 list 선언
  tmp_center = [0.0, 0.0]
  y_min_box = []
  x_min_box = []
  y_max_box = []
  x_max_box = []

  for i in range(int(r)):    # 선택한 피사체 갯수만큼 반복분 돌린다
      k = int(a[i])    # a로 받은 숫자들은 str값이므로 정수형으로 바꿔준다.

      tmp_center[0] = tmp_center[0] + float(center_dic[k][0])    # 피사체들의 중심좌표중 x좌표를 더해준다.
      tmp_center[1] = tmp_center[1] + float(center_dic[k][1])    # 피사체들의 중심좌표중 y좌표를 더해준다.


      y_min_box.append(box_dic[k][0])     # 선택한 피사체 박스의 ymin값들을 하나씩 list에 추가 시켜준다
      x_min_box.append(box_dic[k][1])    #  이하 동문
      y_max_box.append(box_dic[k][2])
      x_max_box.append(box_dic[k][3])

  obeject_y_min = min(y_min_box)      # y_min_box 리스트에 있는 값중에 최소값만 뽑아낸다
  obeject_x_min = min(x_min_box)      # x_min_box 리스트에 있는 값중에 최소값만 뽑아낸다
  obeject_y_max = max(y_max_box)      # y_max_box 리스트에 있는 값중에 최대 값을 뽑아낸다.
  obeject_x_max = max(x_max_box)      # x_max_box 리스트에 있는 값중에 최대 값을 뽑아낸다.

  real_box = [obeject_y_min, obeject_x_min, obeject_y_max, obeject_x_max]   # 최종 box의 좌표가 저장된다. 


  cent_x = float(tmp_center[0])/r   # 위의 반복문 안에서 더해진 중심값들을 피사체의 갯수만큼 나눠 준다.
  cent_y = float(tmp_center[1])/r

  real_center = [float(cent_x), float(cent_y)]   # 최종적인 피사체의 중심좌표를 저장한다.

  print('피사체 프레임 좌표')
  print(real_box)

  print('피사체 중심 좌표')
  print(real_center)



  # Save height & width
  # height, width = img.shape[0:2]

  full_width, full_height = img.size
  x_min, x_max, y_min, y_max = 0.0, full_width, 0.0, full_height



  object_width = float(obeject_x_max - obeject_x_min) # bounding box의 가로 길이
  object_height = float(obeject_y_max - obeject_y_min) # bounding box의 세로 길이

  print("전체 사이즈: (%d, %d)" % (full_width, full_height))

  origin_pix_ratio = (object_height * object_width) / (full_width * full_height)
  print("원본 이미지와 피사체의 비율 :", origin_pix_ratio)


  x_coordinate = float(cent_x)
  y_coordinate = float(cent_y)

# tar_point -> 피사체의 중심좌표 x, y
  tar_point = [float(x_coordinate), float(y_coordinate)]

# 위의 For문을 통하여 전체 이미지 가로/세로 좌표, 피사체의 좌상단/우하단 좌표, 피사체 중심  x,y 좌표가 구해짐.
#%%
# 직접 코딩하여 만든 부분임.
# Zero padding을 방지하기 위한 frame_offset_function 함수

def frame_offset_funtion(width, height, x, y, a, b, c, d, q, w, e, r):
    frame_xmin, frame_xmax, frame_ymin, frame_ymax = q, w, e, r
    frame = [frame_xmin]
    x_min, x_max, y_min, y_max = a, b, c, d
    tar_point = [x, y]
    newframe = [0.0, 0.0, 0.0, 0.0]
    
    # 이 경우에는 frame_offset을 하지 않음.
    if frame_xmin >= 0 and frame_xmax <= width and frame_ymin >= 0.01 and frame_ymax <= height:
        return (q,e,w,r)

    if frame_xmin < 0 :
        print('xmin에 overpadding')
        overpadding_xmin = -1.0 * frame_xmin # 음수(-) x 음수(-) = 양수(+)
        frame_xmin = frame_xmin + overpadding_xmin # result is 0(zero)
        frame_xmax = frame_xmax + overpadding_xmin # result is 0(zero)
        
    if frame_ymin < 0 :
        print('ymin에 overpadding')
        overpadding_ymin = -1.0 * frame_ymin
        frame_ymin = frame_ymin + overpadding_ymin
        frame_ymax = frame_ymax + overpadding_ymin
        
    if frame_xmax > width :
        print('xmax에 overpadding')
        overpadding_xmax = frame_xmax - width
        frame_xmin = frame_xmin - overpadding_xmax
        frame_xmax = frame_xmax - overpadding_xmax
        
    if frame_ymax > height :
        print('ymax에 overpadding')
        overpadding_ymax = frame_ymax - height
        frame_ymin = frame_ymin - overpadding_ymax
        frame_ymax = frame_ymax - overpadding_ymax

    print("offset 임무완료")
    return (frame_xmin, frame_ymin, frame_xmax, frame_ymax)

#%%
## 직접 다 Coding한 부분
## Function Summary Regarding Cropping images ##
  
## 피사체가 작은 경우 ##
  
def D3_rectangle43_frame(width, height, x, y, z, k, a, b, c, d):
    x_min, x_max, y_min, y_max = a, b, c, d
    object_width, object_height = z, k
    tar_point = [x, y]
    bar_1 = 0.0
    bar_2 = 0.0
    
    # 반복문을 통하여 기존 논문처럼 반복적 cropping을 실행
    
    # bar_1 과 bar_2에 임의의 작은 수를 더하거나 뺌
    # 이 작업으로 피사체가 사분면중 하나에 확실하게 속하게 함.

    
    for i in range(70000):
        if width > height: # 전체 이미지가 가로형 일시 
            bar_1 = bar_1 + 0.2
            bar_2 = bar_2 + 0.15

        else: # 전체 이미지가 세로형 일시 
            bar_1 = bar_1 + 0.15
            bar_2 = bar_2 + 0.2



        if tar_point[0] < round(width / 2) and tar_point[1] <= round(height / 2): # 2사분면

            frame_xmin = tar_point[0] - (1 * bar_1)
            frame_xmax = tar_point[0] + (2 * bar_1)
            frame_ymin = tar_point[1] - (1 * bar_2)
            frame_ymax = tar_point[1] + (2 * bar_2)



            frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]


        elif tar_point[0] > round(width / 2) and tar_point[1] <= round(height / 2): # 1사분면

            frame_xmin = tar_point[0] - (2 * bar_1)
            frame_xmax = tar_point[0] + (1 * bar_1)
            frame_ymin = tar_point[1] - (1 * bar_2)
            frame_ymax = tar_point[1] + (2 * bar_2)

            frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]





        elif tar_point[0] <= round(width / 2) and tar_point[1] > round(height / 2): # 3사분면

            frame_xmin = tar_point[0] - (1 * bar_1)
            frame_xmax = tar_point[0] + (2 * bar_1)
            frame_ymin = tar_point[1] - (2 * bar_2)
            frame_ymax = tar_point[1] + (1 * bar_2)



            frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]





        elif tar_point[0] > round(width / 2) and tar_point[1] > round(height / 2): # 4사분면

            frame_xmin = tar_point[0] - (2 * bar_1)
            frame_xmax = tar_point[0] + (1 * bar_1)
            frame_ymin = tar_point[1] - (2 * bar_2)
            frame_ymax = tar_point[1] + (1 * bar_2)



            frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]


        pix_ratio = (object_height * object_width) / ((frame_xmax - frame_xmin) * (frame_ymax - frame_ymin))
        if pix_ratio <= 0.82 and pix_ratio > 0.8:
            # area = (frame_xmin, frame_ymin, frame_xmax, frame_ymax)
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('4:3 직사각형 3분할 프레임: 비율 0.82')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/D3_rectangle43_frame_082.jpg')

            continue

        if pix_ratio <= 0.56 and pix_ratio > 0.54:
            # area = (frame_xmin, frame_ymin, frame_xmax, frame_ymax)
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('4:3 직사각형 3분할 프레임: 비율 0.56')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/D3_rectangle43_frame_056.jpg')


            continue
        
        if pix_ratio <= 0.15:
            # area = (frame_xmin, frame_ymin, frame_xmax, frame_ymax)
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('4:3 직사각형 3분할 프레임: 비율 0.1')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/D3_rectangle43_frame_01.jpg')
            break
        
#%%
def D3_rectangle169_frame(width, height, x, y, z, k, a, b, c, d):
    x_min, x_max, y_min, y_max = a, b, c, d
    object_width, object_height = z, k
    tar_point = [x, y]
    bar_1, bar_2 = 0.0, 0.0
    for i in range(70000):
        if width > height:
            bar_1 = bar_1 + 0.2
            bar_2 = bar_2 + 0.1125

        else:
            bar_1 = bar_1 + 0.1125
            bar_2 = bar_2 + 0.2

        if tar_point[0] < round(width / 2) and tar_point[1] <= round(height / 2):

            frame_xmin = tar_point[0] - (1 * bar_1)
            frame_xmax = tar_point[0] + (2 * bar_1)
            frame_ymin = tar_point[1] - (1 * bar_2)
            frame_ymax = tar_point[1] + (2 * bar_2)

            frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]





        elif tar_point[0] > round(width / 2) and tar_point[1] <= round(height / 2):

            frame_xmin = tar_point[0] - (2 * bar_1)
            frame_xmax = tar_point[0] + (1 * bar_1)
            frame_ymin = tar_point[1] - (1 * bar_2)
            frame_ymax = tar_point[1] + (2 * bar_2)

            frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]





        elif tar_point[0] <= round(width / 2) and tar_point[1] > round(height / 2):

            frame_xmin = tar_point[0] - (1 * bar_1)
            frame_xmax = tar_point[0] + (2 * bar_1)
            frame_ymin = tar_point[1] - (2 * bar_2)
            frame_ymax = tar_point[1] + (1 * bar_2)

            frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]





        elif tar_point[0] > round(width / 2) and tar_point[1] > round(height / 2):

            frame_xmin = tar_point[0] - (2 * bar_1)
            frame_xmax = tar_point[0] + (1 * bar_1)
            frame_ymin = tar_point[1] - (2 * bar_2)
            frame_ymax = tar_point[1] + (1 * bar_2)

            frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]



        # newframe = frame_offset_funtion(full_width, full_height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
        # frame_xmin, frame_xmax, frame_ymin, frame_ymax)

        pix_ratio = (object_height * object_width) / ((frame_xmax - frame_xmin) * (frame_ymax - frame_ymin))
        if pix_ratio <= 0.82 and pix_ratio > 0.8:
            # area = (frame_xmin, frame_ymin, frame_xmax, frame_ymax)
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('16:9 직사각형 3분할 프레임: 비율 0.82')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/D3_rectangle169_frame_082.jpg')

            continue

        if pix_ratio <= 0.56 and pix_ratio > 0.54:
            # area = (frame_xmin, frame_ymin, frame_xmax, frame_ymax)
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('16:9 직사각형 3분할 프레임: 비율 0.56')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/D3_rectangle169_frame_056.jpg')

            continue
        
        if pix_ratio <= 0.15:
            # area = (frame_xmin, frame_ymin, frame_xmax, frame_ymax)
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('16:9 직사각형 3분할 프레임: 비율 0.1')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/D3_rectangle169_frame_01.jpg')
            break

#%%
def center_rectangle43_frame(width, height, x, y, z, k, a, b, c, d):
    x_min, x_max, y_min, y_max = a, b, c, d
    object_width, object_height = z, k
    tar_point = [x, y]
    bar_1 = 0.0
    bar_2 = 0.0

    for i in range(70000):
        if width > height:
            bar_1 = bar_1 + 0.2
            bar_2 = bar_2 + 0.15

        else:
            bar_1 = bar_1 + 0.15
            bar_2 = bar_2 + 0.2

        frame_xmin = tar_point[0] - (1 * bar_1)
        frame_xmax = tar_point[0] + (1 * bar_1)
        frame_ymin = tar_point[1] - (1 * bar_2)
        frame_ymax = tar_point[1] + (1 * bar_2)

        frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

        pix_ratio = (object_height * object_width) / ((frame_xmax - frame_xmin) * (frame_ymax - frame_ymin))
        if pix_ratio <= 0.82 and pix_ratio > 0.8:
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('4:3 직사각형 center 프레임: 비율 0.82')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/cent_rectangle43_frame_082.jpg')
            continue
        if pix_ratio <= 0.56 and pix_ratio > 0.54:
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('4:3 직사각형 center 프레임: 비율 0.56')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/cent_rectangle43_frame_056.jpg')
            continue
        if pix_ratio <= 0.15:
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('4:3 직사각형 center 프레임: 비율 0.1')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/cent_rectangle43_frame_01.jpg')
            break
        
#%%
def center_rectangle169_frame(width, height, x, y, z, k, a, b, c, d):
    x_min, x_max, y_min, y_max = a, b, c, d
    object_width, object_height = z, k
    tar_point = [x, y]
    bar_1, bar_2 = 0.0, 0.0
    pix_ratio = 0.0
    for i in range(70000):
        if width > height:
            bar_1 = bar_1 + 0.2
            bar_2 = bar_2 + 0.1125

        else:
            bar_1 = bar_1 + 0.2
            bar_2 = bar_2 + 0.1125

        frame_xmin = tar_point[0] - (1 * bar_1)
        frame_xmax = tar_point[0] + (1 * bar_1)
        frame_ymin = tar_point[1] - (1 * bar_2)
        frame_ymax = tar_point[1] + (1 * bar_2)

        frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

        pix_ratio = (object_height * object_width) / ((frame_xmax - frame_xmin) * (frame_ymax - frame_ymin))
        if pix_ratio <= 0.82 and pix_ratio > 0.8:
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('16:9 직사각형 center 프레임: 비율 0.82')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/cent_rectangle169_frame_082.jpg')
            continue
        
        if pix_ratio <= 0.56 and pix_ratio > 0.54:
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('16:9 직사각형 center 프레임: 비율 0.56')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/cent_rectangle169_frame_056.jpg')
            continue
        
        if pix_ratio <= 0.15:
            area = frame_offset_funtion(width, height, tar_point[0], tar_point[1], x_min, x_max, y_min, y_max,
                                        frame_xmin, frame_xmax, frame_ymin, frame_ymax)
            print('16:9 직사각형 center 프레임: 비율 0.1')
            print(area)
            cropped_image = img.crop(area)
            # output = cropped_image.resize((640, 640))
            cropped_image.save('resultimage/cent_rectangle169_frame_01.jpg')
            break

#%%

def oversize_btobject_Divison3_frame(width, height, x, y, z, k, a, b, c, d):
    x_min, x_max, y_min, y_max = a, b, c, d
    object_width, object_height = z, k
    tar_point = [x, y]
    bar_1, bar_2 = 0.0, 0.0
    if width > height : #원본사진 가로형----------------------------------------------------------------
        if object_height > object_width:  # 피사체 : 세로 직사각형
            print('원본사진이 가로형입니다')
            print('피사체가 세로형입니다')
            for i in range(70000):
                bar_1 = bar_1 + 0.1

                if tar_point[0] < round(width / 2):  # 세로 중심축 왼쪽
                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (2 * bar_1)
                    frame_ymin = y_min
                    if (0.9 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)


                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                elif tar_point[0] > round(width / 2): #세로 중심축 오른쪽
                    frame_xmin = tar_point[0] - (2 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min
                    if (0.9 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                if frame[0] <= 0 or frame[1] >= width:
                    break

        elif object_height < object_width:  # 피사체 : 가로 직사각형
            print('원본사진이 가로형입니다')
            print('피사체가 가로형입니다')
            for i in range(70000):
                bar_1 = bar_1 + 0.1

                if tar_point[0] < (width / 2):  # 세로 중심축 왼쪽
                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (2 * bar_1)
                    frame_ymin = y_min


                    if (0.90 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                elif tar_point[0] > (width / 2): #세로 중심축 오른쪽
                    frame_xmin = tar_point[0] - (2 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min


                    if (0.90 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                if frame[0] <= 0 or frame[1] >= width:
                    break


    elif height > width : #원본사진이 세로형--------------------------------------------------------------
        if object_height > object_width:  # 피사체 : 세로 직사각형
            print('원본사진이 세로형입니다')
            print('피사체가 세로형입니다')
            for i in range(70000):
                bar_1 = bar_1 + 0.1

                if tar_point[0] < round(width / 2):  # 세로 중심축 왼쪽

                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (2 * bar_1)
                    frame_ymin = y_min

                    if (0.9 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                elif tar_point[0] > round(width / 2): #세로 중심축 오른쪽
                    frame_xmin = tar_point[0] - (2 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min

                    if (0.9 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                if frame[0] <= 0 or frame[1] >= width:
                    break

        elif object_height < object_width:  # 피사체 : 가로 직사각형
            print('원본사진이 세로형입니다')
            print('피사체가 가로형입니다')
            for i in range(70000):
                bar_1 = bar_1 + 0.1

                if tar_point[0] < round(width / 2):  # 세로 중심축 왼쪽

                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (2 * bar_1)
                    frame_ymin = y_min
                    if (0.9 * y_max) <= obeject_y_max <= y_max:
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else:
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                elif tar_point[0] > round(width / 2): #세로 중심축 오른쪽
                    frame_xmin = tar_point[0] - (2 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min
                    if (0.9 * y_max) <= obeject_y_max <= y_max:
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else:
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                if frame[0] <= 0 or frame[1] >= width:
                    break

    area = (frame_xmin, frame_ymin, frame_xmax, frame_ymax)
    print('예외사진 3분할 프레임')
    print(area)
    cropped_image = img.crop(area)
    # output = cropped_image.resize((640, 640))
    cropped_image.save('resultimage/oversize_btobject_divison3_frame.jpg')
    
#%%
def oversize_btobject_center_frame(width, height, x, y, z, k, a, b, c, d):
    x_min, x_max, y_min, y_max = a, b, c, d
    object_width, object_height = z, k
    tar_point = [x, y]
    bar_1, bar_2 = 0.0, 0.0
    if width > height : #원본사진 가로형----------------------------------------------------------------
        if object_height > object_width:  # 피사체 : 세로 직사각형
            print('원본사진이 가로형입니다')
            print('피사체가 세로형입니다')
            for i in range(70000):
                bar_1 = bar_1 + 0.1

                if tar_point[0] < round(width / 2):  # 세로 중심축 왼쪽

                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min
                    if (0.90 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)


                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                elif tar_point[0] > round(width / 2): #세로 중심축 오른쪽
                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min

                    if (0.90 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                if frame[0] <= 0 or frame[1] >= width:
                    break

        elif object_height < object_width:  # 피사체 : 가로 직사각형
            print('원본사진이 가로형입니다')
            print('피사체가 가로형입니다')
            for i in range(70000):
                bar_1 = bar_1 + 0.1

                if tar_point[0] < (width / 2):  # 세로 중심축 왼쪽

                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min

                    if (0.90 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)


                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                elif tar_point[0] > (width / 2): #세로 중심축 오른쪽
                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min

                    if (0.90 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)


                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                if frame[0] <= 0 or frame[1] >= width:
                    break


    elif height > width : #원본사진이 세로형--------------------------------------------------------------
        if object_height > object_width:  # 피사체 : 세로 직사각형
            print('원본사진이 세로형입니다')
            print('피사체가 세로형입니다')
            for i in range(70000):
                bar_1 = bar_1 + 0.1

                if tar_point[0] < round(width / 2):  # 세로 중심축 왼쪽

                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min

                    if (0.90 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                elif tar_point[0] > round(width / 2): #세로 중심축 오른쪽
                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min

                    if (0.90 * y_max) <= obeject_y_max <= y_max :
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else :
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                if frame[0] <= 0 or frame[1] >= width:
                    break

        elif object_height < object_width:  # 피사체 : 가로 직사각형
            print('원본사진이 세로형입니다')
            print('피사체가 가로형입니다')
            for i in range(70000):
                bar_1 = bar_1 + 0.1

                if tar_point[0] < round(width / 2):  # 세로 중심축 왼쪽

                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min
                    if (0.90 * y_max) <= obeject_y_max <= y_max:
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else:
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                elif tar_point[0] > round(width / 2): #세로 중심축 오른쪽
                    frame_xmin = tar_point[0] - (1 * bar_1)
                    frame_xmax = tar_point[0] + (1 * bar_1)
                    frame_ymin = y_min
                    if (0.90 * y_max) <= obeject_y_max <= y_max:
                        frame_ymax = tar_point[1] + (0.5 * object_height)
                    else:
                        frame_ymax = tar_point[1] + (0.55 * object_height)

                    frame = [frame_xmin, frame_xmax, frame_ymin, frame_ymax]

                if frame[0] <= 0 or frame[1] >= width:
                    break

    area = (frame_xmin, frame_ymin, frame_xmax, frame_ymax)
    print('예외사진 3분할 프레임')
    print(area)
    cropped_image = img.crop(area)
    # output = cropped_image.resize((640, 640))
    cropped_image.save('resultimage/oversize_btobject_center_frame.jpg')

#%%
# Result & Image save
    
# 피사체가 큰 경우 (피사체의 크기가 전체사진의 0.43[43%] 이상 일시)
# 0.43은 현업 사진 작가와의 협의 및 실험을 통하여 얻어낸 수치    
    
if object_width >= (0.43 * full_width) or object_height >= (0.43 * full_height): #원본사진에 비해 피사체의 크기가 큰 경우
    #if tar_point[1] > (y_max / 2) : # 피사체가 아래에있는 경우
    print('피사체의 프레임이 많은 비율을 차지하는 경우에 대한 크롭 실행')

    oversize_btobject_Divison3_frame(full_width, full_height, tar_point[0], tar_point[1], \
                                   object_width, object_height, x_min, x_max, y_min, y_max)

    oversize_btobject_center_frame(full_width, full_height, tar_point[0], tar_point[1], \
                               object_width, object_height, x_min, x_max, y_min, y_max)
    print('입력한 사진에 대한 적절한 크롭을 완료 및 제시하였다. 사용자는 사진을 선택하시오.')

else: #원본사진에 비해 피사체의 크기가 작은 경우
    D3_rectangle43_frame(full_width, full_height, tar_point[0], tar_point[1],
                                       object_width, object_height, x_min, x_max, y_min, y_max)

    D3_rectangle169_frame(full_width, full_height, tar_point[0], tar_point[1],
                                       object_width, object_height, x_min, x_max, y_min, y_max)

    center_rectangle43_frame(full_width, full_height, tar_point[0], tar_point[1],
                                       object_width, object_height, x_min, x_max, y_min, y_max)

    center_rectangle169_frame(full_width, full_height, tar_point[0], tar_point[1],
                                       object_width, object_height, x_min, x_max, y_min, y_max)

    print('입력한 사진에 대한 적절한 크롭을 완료 및 제시하였다. 사용자는 사진을 선택하시오')