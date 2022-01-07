from PIL import Image
import numpy as np
import os
import cv2
import pandas as pd
import tqdm


def load_image_label(train_mask, training_path, idx):
    img_path = os.path.join(training_path, train_mask['name'][idx])
    img = cv2.imread(img_path)
    label = rle_decode(train_mask['mask'][0]).astype('float32')
    return img, label


# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def one_hot(label, num_classes):
    if np.ndim(label) == 3:
        label = np.square(label, axis=-1)
    
    assert np.ndim(label) == 2
    
    heat_map = np.ones(shape=label.shape[0: 2] + (num_classes,))
    
    for i in range(num_classes):
        heat_map[:, :, i] = np.equal(label, i).astype("float32")
    
    return heat_map
    

# cv2.resize(dsize=(w, h))
def resize_image(image, label, target_size=None):
    image = cv2.resize(image, dsize=target_size[::-1])
    label = cv2.resize(label, dsize=target_size[::-1], interpolation=cv2.INTER_NEAREST)
    return image, label


def crop_image(image, label, cropped_size):
    h, w = image.shape[:2]
    c_h, c_w = cropped_size
    
    if c_h > h or c_w > w:
        image = cv2.resize(image, dsize=(max(c_w, w), max(c_h, h)))
        label = cv2.resize(label, dsize=(max(c_w, w), max(c_h, h)))
    
    h, w = image.shape
    
    h_start = np.random.randint(0, (h - c_h))
    w_start = np.random.randint(0, (w - c_w))
    
    image = image[h_start: (h_start + c_h), w_start: (w_start + c_w)]
    label = label[h_start: (h_start + c_h), w_start: (w_start + c_w)]
    
    return image, label


def data_wash():
    csv_path = r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\train_mask.csv'
    data_path = r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\train'
    clean_csv = r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\clean\clean_mask.csv'
    
    train_mask = pd.read_csv(csv_path, sep='\t', names=['name', 'mask'])
    leng = len(train_mask)
    for i in tqdm.tqdm(range(leng)):
        img_name = train_mask['name'][i]
        img_path = os.path.join(data_path, img_name)
        mask = train_mask['mask'][i]
        
        imgFlag = os.path.exists(img_path)
        maskFlag = isinstance(mask, str)
        
        if imgFlag and maskFlag:  # 存入clean
            cur_line = '\n' + img_name + '\t' + mask if i > 0 else img_name + '\t' + mask
            # print('完整')
            with open(clean_csv, 'a') as f:
                f.write(cur_line)


def split_dataset():
    clean_csv = r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\clean\clean_mask.csv'
    
    train_csv = r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\clean\train_mask.csv'
    val_csv = r'C:\Users\wang\Desktop\bbb\code\dataset\semantic\clean\val_mask.csv'
    
    train_mask = pd.read_csv(clean_csv, sep='\t', names=['name', 'mask'])
    
    idx_list = np.arange(len(train_mask))
    np.random.shuffle(idx_list)
    
    for i in tqdm.tqdm(range(len(idx_list))):
        
        img_name = train_mask['name'][idx_list[i]]
        mask = train_mask['mask'][idx_list[i]]
        
        cur_line = img_name + '\t' + mask + '\n'
        
        if i < 4960:  # val
            print(i)
            with open(val_csv, 'a') as f:
                f.write(cur_line)
        else:  # train
            with open(train_csv, 'a') as f:
                f.write(cur_line)


































