#helper File
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import os
from collections import OrderedDict
import json
from PIL import Image, ImageDraw
import math
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def create_mask(mask):
  color = np.array([255/255, 255/255, 255/255])
  h, w = mask.shape[-2:]
  mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
  return mask_image

def show_points(coords, labels, ax, marker_size=275):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def Load_images(df,f):
  Unique_ID = df['Well'].unique()
  IDS = []
  images = {}
  ordered_images = {}
  for file_name in tqdm(os.listdir(f)):
    img = cv2.imread(os.path.join(f, file_name))
    if img is not None:

      for ID in Unique_ID.tolist():
        start_index = file_name.find(ID)
        if start_index != -1:
          found_ID = file_name[start_index:start_index+7]
          plate_num = f[-1]
          vastdate = file_name[0:10]
          view = fetch_view_number(file_name)
          #print(vastdate)

      if ((df['Well']==found_ID[0:3]) & (df['DateVAST']==str(vastdate))).any():
        if df[(df['Well']==found_ID[0:3]) &(df['DateVAST']==str(vastdate))]['GeneralIssues'].values[0] == 'No':
          if df[(df['Well']==found_ID[0:3]) &(df['DateVAST']==str(vastdate))]['Truncated'].values[0] == 'No':

            filetered_df = df[(df.DateVAST == str(vastdate)) & (df.Plate == int(plate_num))]

            selected_row = filetered_df[filetered_df['Well'] == found_ID[0:3]]
            if selected_row.shape[0] == 0:
                continue
            if selected_row.shape[0] > 1:
                selected_row = selected_row.iloc[[0]]
            A = selected_row['Age'].item()
            G = selected_row['Genotype'].item()
            D = selected_row['DateVAST'].item()
            P = selected_row['Plate'].item()
            I = selected_row['Well'].item()
            full_id = '{A}_{G}_{D}_{P}_{I}_{V}'.format(A=A,G=G,D=D,P=P,I=I,V=view)
            images[full_id] = img

  ordered_images = OrderedDict(sorted(images.items()))
  return ordered_images

#Consolidated Parser
def JSON_Parser(file):
  f = open(file)
  data = json.load(f)
  features = [key for key,values in data.items()]
  features = features[3:-1]
  features_dict = {}

  for i in features:
    if "shape" in data[i]:
      if isinstance(data[i]["shape"], list) and  data[i]["shape"] is not None:
        pair_list = []
        for j in range(len(data[i]["shape"][0]["x"])):
          pair_list.append((data[i]["shape"][0]["y"][j], data[i]["shape"][0]["x"][j]))
        features_dict[i] = pair_list
      else:
        if data[i]['shape'] is not None:
          if isinstance(data[i]['shape']['x'],list):
            pair_list = []
            for j in range(len(data[i]["shape"]["x"])):
              pair_list.append((math.floor(data[i]["shape"]["y"][j]), math.floor(data[i]["shape"]["x"][j])))

        features_dict[i] = pair_list
  #print(features_dict.keys())
  return features_dict

def JSON_RegionProps(file):
  f = open(file)
  data = json.load(f)
  features = [key for key,values in data.items()][3:-1]
  features_dict = {}
  for i in features:
    if 'regionprops' in data[i]:
      features_dict[ i + '_' + 'regionprops'] = data[i]['regionprops']
  return features_dict

#Function to Load the JSON files from the folder, saving them as dictionaries
def Load_JSON_from_folder(f,df):

  Unique_ID = df['Well'].unique()
  JSONs = {}
  JSON_prop = {}
  json_files = [path for path in os.listdir(f) if path.endswith('.json')]
  #print(len(json_files))
  for shape in tqdm(json_files):
    #print(shape)
    for ID in Unique_ID.tolist():
      start_index = shape.find(ID)
      if start_index != -1:
        found_ID = shape[start_index:start_index+7]
        plate_num = f[-1]
        vastdate = shape[0:10]
        view = fetch_view_number(shape)
        #print(view)
    if ((df['Well']==found_ID[0:3]) & (df['DateVAST']==str(vastdate))).any():
        if df[(df['Well']==found_ID[0:3]) &(df['DateVAST']==str(vastdate))]['GeneralIssues'].values[0] == 'No':
          if df[(df['Well']==found_ID[0:3]) &(df['DateVAST']==str(vastdate))]['Truncated'].values[0] == 'No':
            #print(file_name, 'got here')
            feature_dict = JSON_Parser(os.path.join(f,shape))
            region_prop_dict = JSON_RegionProps(os.path.join(f,shape))

            filetered_df = df[(df.DateVAST == str(vastdate)) & (df.Plate == int(plate_num))]
            selected_row = filetered_df[filetered_df['Well'] == found_ID[0:3]]
            if selected_row.shape[0] == 0:
                continue
            if selected_row.shape[0] > 1:
                selected_row = selected_row.iloc[[0]]

            A = selected_row['Age'].item()
            G = selected_row['Genotype'].item()
            D = selected_row['DateVAST'].item()
            P = selected_row['Plate'].item()
            I = selected_row['Well'].item()
            full_id = '{A}_{G}_{D}_{P}_{I}_{V}'.format(A=A,G=G,D=D,P=P,I=I,V=view)

            JSONs[full_id] = feature_dict
            JSON_prop[full_id] = region_prop_dict

  JSONs = OrderedDict(sorted(JSONs.items()))
  JSON_prop = OrderedDict(sorted(JSON_prop.items()))
  return JSONs, JSON_prop


def Generate_shapes(polygon_coords, img):
  poly = Polygon(polygon_coords)
  Points_in_shape = []
  Points_not_in_shape = []
  coordinates = []

  #Coordinates that correspond to image shape
  for x in range(img.shape[0]):
    for y in range(img.shape[1]):
      coordinates.append((x,y))

  for i in range(len(coordinates)):
    p = Point(coordinates[i])
    if not p.within(poly):
      Points_not_in_shape.append(coordinates[i])
    if p.within(poly):
      Points_in_shape.append(coordinates[i])

  return Points_not_in_shape, Points_in_shape

def crop_background(img, Points_out):
  cropped_img = np.copy(img)
  for x in range(len(Points_out)):
    cropped_img[Points_out[x][0]][Points_out[x][1]] = (0,0,0)
  return cropped_img

def Find_centroid(polygon_coords, img):
  poly = Polygon(polygon_coords)
  return [poly.centroid.y,poly.centroid.x]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def detect_trunc(id,dict,border):
  flag = False
  #print(dict[id])
  mask = create_mask(dict[id])
  mask_coords = np.column_stack(np.where(rgb2gray(mask) != (0))).tolist()
  for i in border:
    if i in mask_coords:
      flag = True
      break

  return flag, len(mask_coords)

def find_border_pixels(img):
    bw = 5  # Border width
    # Ensure the image is in grayscale
    gray_img = rgb2gray(img)
    # Initialize an empty list to store the border coordinates
    border_coords = []
    # Top border
    border_coords.extend(np.column_stack(np.where(gray_img[:bw, :] != 0)))
    # Bottom border
    border_coords.extend(np.column_stack(np.where(gray_img[-bw:, :] != 0)))
    # Left border
    border_coords.extend(np.column_stack(np.where(gray_img[:, :bw] != 0)))
    # Right border
    border_coords.extend(np.column_stack(np.where(gray_img[:, -bw:] != 0)))
    return border_coords

def Full_crop(img_set,mask_set):
  cropped_set = {}
  for key in tqdm(img_set):
    if (key in img_set.keys()) and (key in mask_set.keys()):
        cropped_set[key] = (255 - img_set[key])*create_mask(mask_set[key])
    else:
        continue
  return cropped_set

def find_rightmost_point(contour):
  #contour = json_dict['B03_1_1']['contourDV_net']
  h = [i[0] for i in contour]
  w = [i[1] for i in contour]

  farthest_heigth = h[w.index(max(w))]
  farthest_width = w[w.index(max(w))]


#Function to loop through the images and get their masks:
def Segment_images(predictor, img_dict, json_dict):
  img_id_to_mask = {}
  for img in tqdm(json_dict.keys()):
    #For Angles 2 and 4
    if img[-1] == str(2):
      predictor.set_image(img_dict[img])
      if (len(list(json_dict[img].keys()))) == 1:
        contour_point = Find_centroid(json_dict[img]['contour_net'],img_dict[img])
        input_point = np.array([contour_point])
        input_label = np.array([1])
      else:
        yolk_centroid = Find_centroid(json_dict[img]['contour_net'],img_dict[img])
        eye_centroid = Find_centroid(json_dict[img]['eye_net'],img_dict[img])
        #contour_farthest_point = find_rightmost_point(json_dict[img]['contour_net'])

        input_point = np.array([[yolk_centroid[0],yolk_centroid[1]],[eye_centroid[0],eye_centroid[1]]])
        input_label = np.array([1,1])

      masks, scores, logits = predictor.predict(
      point_coords=input_point,
      point_labels=input_label,
      multimask_output=True,
      )

      #Single mask:
      mask_input = logits[np.argmax(scores), :, :]
      masks, _, _ = predictor.predict(
      point_coords=input_point,
      point_labels=input_label,
      mask_input=mask_input[None, :, :],
      multimask_output=False,
      )
      img_id_to_mask[img] = masks
    #For angles 1 and 3 
  return img_id_to_mask

#main Segmentation funciton for any perspective
def Segment_images(predictor, img_dict, json_dict): 
    img_id_to_mask = {}
    for img in tqdm(json_dict.keys()):
        #print(img)
        if int(img[-1]) % 2 == 0:
            predictor.set_image(img_dict[img])
            if (len(list(json_dict[img].keys()))) == 1:
                ctr_pt = Find_centroid(json_dict[img]['contour_net'],img_dict[img])
                in_pt = np.array([ctr_pt])
                in_lbl = np.array([1])
            else:
                yk_ctr = Find_centroid(json_dict[img]['contour_net'],img_dict[img])
                eye_ctr = Find_centroid(json_dict[img]['eye_net'],img_dict[img])
                #contour_farthest_point = find_rightmost_point(json_dict[img]['contour_net'])

            in_pt = np.array([[yk_ctr[0],yk_ctr[1]],[eye_ctr[0],eye_ctr[1]]])
            in_lbl = np.array([1,1])

        if int(img[-1]) % 2 == 1:
            predictor.set_image(img_dict[img])
            if (len(list(json_dict[img].keys()))) == 1:
                ctr_pt = Find_centroid(json_dict[img]['contourDV_net'],img_dict[img])
                in_pt = np.array([ctr_pt])
                in_lbl = np.array([1])
            else:
                yok_ctr = Find_centroid(json_dict[img]['yolkDV_net'],img_dict[img])
                eye_ctr = Find_centroid(json_dict[img]['eye1DV_net'],img_dict[img])
                contour_farthest_point = find_rightmost_point(json_dict[img]['contourDV_net'])

            in_pt = np.array([[yok_ctr[0],yok_ctr[1]],[eye_ctr[0],eye_ctr[1]]])
            in_lbl = np.array([1,1])
            #Multi_mask:

        masks, scores, logits = predictor.predict(
        point_coords=in_pt,
        point_labels=in_lbl,
        multimask_output=True,
        )

        #Single mask:
        mask_input = logits[np.argmax(scores), :, :]
        masks, _, _ = predictor.predict(
        point_coords=in_pt,
        point_labels=in_lbl,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
        )
        img_id_to_mask[img] = masks
    return img_id_to_mask

#Making dataset based on the read in values:
def create_Dataset(img_masks,cropped_imgs):
  geno, age,ids,trun,plates,dates = [],[],[],[],[],[]
  for id in cropped_imgs.keys():
    age.append(id.split('_')[0])
    geno.append(id.split('_')[1])
    ids.append(id.split('_')[4])
    plates.append(id.split('_')[3])
    dates.append(id.split('_')[2])

    border = find_border_pixels(cropped_imgs[id])
    if border == []:
      trunc = False
    trun.append(trunc)
  new_df = pd.DataFrame({'Label': geno, 'Age': age,
              'Date':dates, 'Fish_ID':ids,
              'Plate':plates, 'truncated':trun})
  return new_df

#Next step in preprocessing
#Function used for cropping the background of the image
def crop_background(img, Points_out):
  cropped_img = np.copy(img)
  for x in range(len(Points_out)):
    cropped_img[Points_out[x][0]][Points_out[x][1]] = (0,0,0)
  return cropped_img

#Cropping everything outside of the bounding box
def resize_image(img):
  gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cnts = cv2.findContours(gray_scale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  #print(cnts)
  for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      ROI = img[y:y+h, x:x+w]
  return ROI

#Function to padd images and resize them
def padding_image(img, new_height, new_width, padding_color):
  old_height, old_width, channels = img.shape
  result = np.full((new_height,new_width, channels), padding_color, dtype=np.uint8)
  # compute center offset
  x_center = (new_width - old_width) // 2
  y_center = (new_height - old_height) // 2
  # copy img image into center of result image
  result[y_center:y_center+old_height,
      x_center:x_center+old_width] = img
  # view result
  return result

#Apply the padding to a full image dictionary
def full_augment(img_dict):
  for key in img_dict:
    img_dict[key] = padding_image(img_dict[key],200,950,(0,0,0))
  return img_dict

def Process_imgs(imgs):
  processed_images = np.zeros((imgs.shape[0],200,950,3))
  i_for_rmval = []
  for i in range(imgs.shape[0]):
    img = imgs[i].astype('uint8') * 255
    resized_img = resize_image(img)
    #print('Got here')
    if  120 > resized_img.shape[0] > 50:
      print(resized_img.shape)
      padded_img = padding_image(resized_img,200,950,(0,0,0))
      inverted_img = 255 - padded_img
      wt_pixels = np.where( (inverted_img[:, :, 0] == 255) & (inverted_img[:, :, 1] == 255) & (inverted_img[:, :, 2] == 255))
      inverted_img[wt_pixels] = [0, 0 ,0]
      processed_images[i] = inverted_img
    else:
      i_for_rmval.append(i)
  return processed_images, i_for_rmval

def gen_csv(new_df):
  new_df['Issues'] = [False for i in range(new_df.shape[0])]
  new_df['Train_Or_Test'] = ['TRAIN' for i in range(new_df.shape[0])]
  new_df['Train_Or_Test'][0] = 'VALID'
  new_df['Train_Or_Test'][1] =  'VALID'
  new_df['Intensity'] = new_df['Mask_intensity']
  return new_df


def Load_Dictionaries(BASE_DIR,df):
    img_dict = {}
    json_dict = {}
    for dir in os.listdir(BASE_DIR):
        print(dir)
        t_img_dict = Load_images(df,os.path.join(BASE_DIR,dir))
        t_json_dict,_ = Load_JSON_from_folder(os.path.join(BASE_DIR,dir), df)
        img_dict[dir] = t_img_dict
        json_dict[dir] = t_json_dict
    return img_dict, json_dict

def fetch_view_number(filename):
    # Define a regular expression pattern to match the view number
    pattern = r'_([0-9]+)_(\d+)_rot'
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    if match:
        # Extract and return the view number
        return match.group(2)
    else:
        raise ValueError("View number not found in the filename.")
