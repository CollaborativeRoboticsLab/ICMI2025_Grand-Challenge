from PIL import Image, ImageDraw
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import pandas as pd
from sklearn import preprocessing 
import supervision as sv
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset


def parse_cvat_segmentation_xml(xml_content):
#Function taken from https://github.com/cj-mills/torchvision-annotation-tutorials/tree/main
    # Parse the XML content from the provided string.
    root = ET.fromstring(xml_content)
    data = {}

    for image in root.findall('image'):
        # Extract attributes for each image.
        image_id = image.get('id')
        image_name = image.get('name')
        width = image.get('width')
        height = image.get('height')

        # Initialize a dictionary to store image data.
        image_data = {
            'Image ID': int(image_id),
            'Image Name': image_name,
            'Width': int(width),
            'Height': int(height),
            'Polygons': []
        }

        # Iterate over each polygon element within the current image.
        for polygon in image.findall('polygon'):
            # Extract the label and points of the polygon.
            label = polygon.get('label')
            
            points = ','.join(polygon.get('points').split(';'))
            points = [float(point) for point in points.split(',')]
            
            # Create a dictionary to store the polygon data.
            points_data = {
                'Label': label,
                'Points': points
            }
            image_data['Polygons'].append(points_data)

        # Add the processed image data to the main data dictionary.
        data[image_id] = image_data

    # Convert the data dictionary into a pandas DataFrame and return it.
    return pd.DataFrame.from_dict(data, orient='index')

def create_polygon_mask(image_size, vertices):
    #function taken from https://github.com/cj-mills/torchvision-annotation-tutorials/tree/main
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = Image.new('L', image_size, 0)
    
    # Draw the polygon on the image. The area inside the polygon will be white (255).
    ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))

    # Return the image with the drawn polygon
    return mask_img

def assignLab(labelsUnEncoded,mask_imgs):
    labels = []
    for i in range(len(labelsUnEncoded)):
        lab = labelsUnEncoded[i]
        lab = str(lab)
        #print(lab)
        encoded = 4
        if lab == 'plastic_loose':
            encoded = 0
        if lab == 'organic':
            encoded = 1
        if lab == 'plastic_packaging':
            encoded = 2
        if lab == 'metal':
            encoded = 3
        #else:
        #    encoded = 4
        labels.append(encoded)

    masked = np.zeros((1049,2020,len(mask_imgs)))
    for i in range(len(mask_imgs)):
        #print(labels[i])
        masked[:,:,i]+=mask_imgs[i]
    masked = masked[:,250:1750,:]
    masked = masked[0:1024,0:1024]
    masked = np.float32(masked>0)

    return masked




def YOLOdataGen(annotation_file_path,path_to_annotations): #'PATH_TO_YOUR_TRAINING_ANNOTATION_XML_FILE.xml'
    #annotation_file_path = 'PATH_TO_YOUR_TRAINING_ANNOTATION_XML_FILE.xml'  # Replace with your XML file path
    with open(annotation_file_path, 'r', encoding='utf-8') as file:
        xml_content = file.read()
    # Parse the XML content
    annotation_df = parse_cvat_segmentation_xml(xml_content)
    # Add a new column 'Image ID' by extracting it from 'Image Name'
    # This assumes that the 'Image ID' is the part of the 'Image Name' before the first period
    annotation_df['Image ID'] = annotation_df['Image Name'].apply(lambda x: x.split('.')[0])
    # Set the new 'Image ID' column as the index of the DataFrame
    annotation_df = annotation_df.set_index('Image ID')
    for i in range(len(annotation_df['Polygons'])):
        polygon_points = annotation_df.iloc[i]['Polygons']
        mask_imgs = [create_polygon_mask((2020,1049), polygon['Points'],) for polygon in polygon_points]
        labelsUnEncoded = [polygon['Label'] for polygon in polygon_points]
        masked = assignLab(labelsUnEncoded,mask_imgs)
        nonzeros = np.where(np.sum(masked,axis=(0,1))>100)[0]
        maskImage = masked[:,:,nonzeros]
        maskImage = maskImage.transpose(2,0,1)

        polygons = [sv.mask_to_polygons(m) for m in maskImage]


        output = []
        labels = []

        for j in range(len(labelsUnEncoded)):
            lab = labelsUnEncoded[j]
            lab = str(lab)
            #print(lab)
            encoded = 4
            if lab == 'plastic_loose':
                encoded = 0
            if lab == 'organic':
                encoded = 1
            if lab == 'plastic_packaging':
                encoded = 2
            if lab == 'metal':
                encoded = 3
            #else:
            #    encoded = 4
            labels.append(encoded)
        labels = np.array(labels)
        labs = labels[nonzeros]
        #print(len(polygons))
        #print(polygons)
        for j in range(len(polygons)):
            #print(j)
            data_array = polygons[j][0].flatten()/1024
            dat = np.array([labs[j]])
            dat = np.concatenate((dat,data_array))
            output.append(dat)

        for j in range(len(output)):
            listed = output[j].tolist()
            listed[0] = int(listed[0])
            output[j] = str(listed).replace('[','').replace(']','').replace(',', ' ')
        idx = str(i)
        fname = path_to_anntations+idx.zfill(6)+'.txt'
        
        with open(fname, 'w') as f:
            f.write('\n'.join([''.join(l2) for l2 in output]))
        

