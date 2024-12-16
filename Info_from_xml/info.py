import os
from glob import glob
import pandas as pd
from functools import reduce
from xml.etree import ElementTree as ET
import shutil


# load all the xml files from the labels folder inside images_data folder which is also inside the data_preparation folder.
xml_list = glob('data_preparation/images_data/Annotations/*.xml')

# getting the path for train and test folders
train_folder = 'data_preparation/images_data/Train'
test_folder = 'data_preparation/images_data/Test'

# data cleaning. replacing \\ with /
xml_list = list(map(lambda x: x.replace('\\','/'),xml_list)) 

# read all xml files and from each xml file we need to extract:
# filename
# size(width,height)
# object(name,xmin,ymin,xmax,ymax)

allxml_files_data = []
for xml_file in xml_list:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
# extracting file names from the xml file.
    image_name = root.find('filename').text
    
# size(width,height)
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
# object(name,xmin,ymin,xmax,ymax)
    objects = root.findall('object')
    for object in objects:
        name =  object.find('name').text
        bndbox =  object.find('bndbox')
        xmin = bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin = bndbox.find('ymin').text
        ymax = bndbox.find('ymax').text
        
    data_list = [image_name,width,height,name,xmin,xmax,ymin,ymax]
    allxml_files_data.append(data_list)
    
df = pd.DataFrame(allxml_files_data,columns=['filename','width','height','name','xmin','xmax','ymin','ymax'])

print(df)

cols = ['width','height','xmin','xmax','ymin','ymax']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# getting the center x and center y of object

df['center_x'] = ((df['xmin']+df['xmax'])/2)/df['width']
df['center_y'] = ((df['ymin']+df['ymax'])/2)/df['height']

# getting th height and width of object

df['w'] = (df['xmax']-df['xmin'])/df['width']
df['h'] = (df['ymax']-df['ymin'])/df['height']

images = df['filename'].unique()

image_df = pd.DataFrame(images, columns=['filename'])

# selecting 80% of data for training the images

image_train = tuple(image_df.sample(frac=0.5)['filename'])

image_test = tuple(image_df.query(f'filename not in @image_train')['filename'])

train_df = df.query(f'filename in @image_train')
test_df = df.query(f'filename in @image_test')

def label_encoding(x):
    labels = {
    'background': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}
    return labels[x]

# assigning ids for the respective names
train_df.loc[:, 'id'] = train_df['name'].apply(label_encoding)
test_df.loc[:, 'id'] = test_df['name'].apply(label_encoding)


data_columns = ['filename', 'id', 'center_x', 'center_y', 'w', 'h']

group_by_object_train = train_df[data_columns].groupby('filename')
group_by_object_test = test_df[data_columns].groupby('filename')

# now moving train images to train folder and test images to test folder
def save_images_to_folder(filename, folderpath, group_obj):
    # moving images to folder path
    src = os.path.join('data_preparation/images_data/Images/', filename)
    dst = os.path.join(folderpath, filename)
    dst = dst.replace('\\','/')
    try:
        shutil.move(src, dst)
    except FileNotFoundError:
        print(f"File not found: {src}")
    
    # moving labels in the form of text to txt folderpath
    text_filename = os.path.join(folderpath, os.path.splitext(filename)[0] + '.txt')
    group_obj.get_group(filename).set_index('filename').to_csv(text_filename, sep=' ', index=False, header=False)


file_name_series_train = pd.Series(group_by_object_train.groups.keys())
file_name_series_test = pd.Series(group_by_object_test.groups.keys())

file_name_series_train.apply(save_images_to_folder, args=(train_folder,group_by_object_train))
file_name_series_test.apply(save_images_to_folder, args=(test_folder,group_by_object_test))


print(file_name_series_test)
print(file_name_series_train)  





