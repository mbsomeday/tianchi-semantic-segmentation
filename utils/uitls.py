from PIL import Image
import numpy as np
import os
import cv2


def load_imageAndMask(train_mask, training_path, idx):
    img_path = os.path.join(training_path, train_mask['name'][idx])
    img = cv2.imread(img_path)
    mask = rle_decode(train_mask['mask'][0]).astype('float32')
    return img, mask


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



































