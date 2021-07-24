from __future__ import division, print_function

import os, glob, re
import os.path
import dicom
import numpy as np
from PIL import Image, ImageDraw
from keras import backend as K
import pandas as pd

import argparse
from math import ceil
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from keras import utils
from keras.preprocessing import image as keras_image
from keras.preprocessing.image import ImageDataGenerator

from keras import losses, optimizers, utils
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint

from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
import sys
from keras.regularizers import l2,l1
import matplotlib.pyplot as plt







# Model definition
def downsampling_block(input_tensor, filters, padding='valid',
                       batchnorm=False, dropout=sys.argv[1],regularizer=sys.argv[2]):
    _, height, width, _ = K.int_shape(input_tensor)
    assert height % 2 == 0
    assert width % 2 == 0

    x = Conv2D(filters, kernel_size=(3,3), padding=padding,kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(input_tensor)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3,3), padding=padding,kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return MaxPooling2D(pool_size=(2,2))(x), x

def upsampling_block(input_tensor, skip_tensor, filters, padding='valid',
                     batchnorm=False, dropout=sys.argv[1],regularizer=sys.argv[2]):
    x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2),kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(input_tensor)

    _, x_height, x_width, _ = K.int_shape(x)
    _, s_height, s_width, _ = K.int_shape(skip_tensor)
    h_crop = s_height - x_height
    w_crop = s_width - x_width
    assert h_crop >= 0
    assert w_crop >= 0
    if h_crop == 0 and w_crop == 0:
        y = skip_tensor
    else:
        cropping = ((h_crop//2, h_crop - h_crop//2), (w_crop//2, w_crop - w_crop//2))
        y = Cropping2D(cropping=cropping)(skip_tensor)

    x = Concatenate()([x, y])

    x = Conv2D(filters, kernel_size=(3,3), padding=padding,kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3,3), padding=padding,kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return x

def unet(height, width, channels, classes, features=32, depth=3,
         temperature=1.0, padding='valid', batchnorm=False, dropout=float(sys.argv[1]),regularizer=float(sys.argv[2])):
    x = Input(shape=(height, width, channels))
    inputs = x

    skips = []
    for i in range(depth):
        x, x0 = downsampling_block(x, features, padding,
                                   batchnorm, dropout,regularizer)
        skips.append(x0)
        features *= 2

    x = Conv2D(filters=features, kernel_size=(3,3), padding=padding,kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters=features, kernel_size=(3,3), padding=padding,kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    for i in reversed(range(depth)):
        features //= 2
        x = upsampling_block(x, skips[i], features, padding,
                             batchnorm, dropout,regularizer)

    x = Conv2D(filters=classes, kernel_size=(1,1),kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(x)

    logits = Lambda(lambda z: z/temperature)(x)
    probabilities = Activation('softmax')(logits)

    return Model(inputs=inputs, outputs=probabilities)













# Loss function and various metrics

def FocalLoss(targets, inputs, alpha=0.8, gamma=2):

    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)

    return focal_loss


def Combo_loss(targets, inputs):
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)

    smooth = 1

    ALPHA = 0.5
    CE_RATIO = 0.5
    e = K.epsilon()
    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, e, 1.0 - e)
    out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

    return combo


def IoULoss(targets, inputs, smooth=1e-6):

    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(targets*inputs)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU



def TverskyLoss(targets, inputs, alpha=0.5, beta=0.5, smooth=1e-6):

    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    return 1 - Tversky


def soft_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    return (2 * intersection + smooth) / (area_true + area_pred + smooth)


def hard_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_sorensen_dice(y_true_int, y_pred_int, axis, smooth)


sorensen_dice = hard_sorensen_dice


def sorensen_dice_loss(y_true, y_pred, weights):
    batch_dice_coefs = soft_sorensen_dice(y_true, y_pred, axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * dice_coefs)


def weighted_categorical_crossentropy(y_true, y_pred, weights, epsilon=1e-8):
    ndim = K.ndim(y_pred)
    ncategory = K.int_shape(y_pred)[-1]
    y_pred /= K.sum(y_pred, axis=(ndim-1), keepdims=True)
    y_pred = K.clip(y_pred, epsilon, 1-epsilon)
    w = K.constant(weights) * (ncategory / sum(weights))
    cross_entropies = -K.mean(y_true * K.log(y_pred), axis=tuple(range(ndim-1)))
    return K.sum(w * cross_entropies)














# Data loading and preprocessing

def rotate_im(image):
    height, width = image.shape
    return np.rot90(image) if width < height else image

class PatientData(object):

    def __init__(self, directory):
        self.directory = os.path.normpath(directory)

        glob_search = os.path.join(directory, "P*list.txt")
        files = glob.glob(glob_search)

        self.contour_list_file = files[0]
        match = re.search("P(..)list.txt", self.contour_list_file)
        self.index = int(match.group(1))

        self.load_images()

        try:
            self.load_masks()
        except FileNotFoundError:
            pass

    @property
    def images(self):
        return [self.all_images[i] for i in self.labeled]

    @property
    def dicoms(self):
        return [self.all_dicoms[i] for i in self.labeled]

    @property
    def dicom_path(self):
        return os.path.join(self.directory, "P{:02d}dicom".format(self.index))

    def load_images(self):
        glob_search = os.path.join(self.dicom_path, "*.dcm")
        dicom_files = sorted(glob.glob(glob_search))
        self.all_images = []
        self.all_dicoms = []
        for dicom_file in dicom_files:
            plan = dicom.read_file(dicom_file)
            image = rotate_im(plan.pixel_array)
            self.all_images.append(image)
            self.all_dicoms.append(plan)
        self.image_height, self.image_width = image.shape
        self.rotated = (plan.pixel_array.shape != image.shape)

    def load_contour(self, filename):
        match = re.search("patient../(.*)", filename)
        path = os.path.join(self.directory, match.group(1))
        x, y = np.loadtxt(path).T
        if self.rotated:
            x, y = y, self.image_height - x
        return x, y

    def contour_to_mask(self, x, y, norm=255):
        BW_8BIT = 'L'
        polygon = list(zip(x, y))
        image_dims = (self.image_width, self.image_height)
        img = Image.new(BW_8BIT, image_dims, color=0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        return norm * np.array(img, dtype='uint8')

    def load_masks(self):
        with open(self.contour_list_file, 'r') as f:
            files = [line.strip() for line in f.readlines()]

        inner_files = [path.replace("\\", "/") for path in files[0::2]]
        outer_files = [path.replace("\\", "/") for path in files[1::2]]

        self.labeled = []
        for inner_file in inner_files:
            match = re.search("P..-(....)-.contour", inner_file)
            frame_number = int(match.group(1))
            self.labeled.append(frame_number)

        self.endocardium_contours = []
        self.epicardium_contours = []
        self.endocardium_masks = []
        self.epicardium_masks = []
        for inner_file, outer_file in zip(inner_files, outer_files):
            inner_x, inner_y = self.load_contour(inner_file)
            self.endocardium_contours.append((inner_x, inner_y))
            outer_x, outer_y = self.load_contour(outer_file)
            self.epicardium_contours.append((outer_x, outer_y))

            inner_mask = self.contour_to_mask(inner_x, inner_y, norm=1)
            self.endocardium_masks.append(inner_mask)
            outer_mask = self.contour_to_mask(outer_x, outer_y, norm=1)
            self.epicardium_masks.append(outer_mask)


def load_images(data_dir, mask='both'):

    glob_search = os.path.join(data_dir, "patient*")
    patient_dirs = sorted(glob.glob(glob_search))

    images = []
    inner_masks = []
    outer_masks = []
    for patient_dir in patient_dirs:
        p = PatientData(patient_dir)
        images += p.images
        inner_masks += p.endocardium_masks
        outer_masks += p.epicardium_masks

    images = np.asarray(images)[:,:,:,None]

    masks = np.asarray(inner_masks)
    dims = masks.shape
    classes = len(set(masks[0].flatten()))
    new_shape = dims + (classes,)
    masks = utils.to_categorical(masks).reshape(new_shape)

    return images, masks

class Iterator(object):
    def __init__(self, images, masks, batch_size,
                 shuffle=True,
                 rotation_range=180,
                 ):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        augment_options = {
            'rotation_range': rotation_range,
        }
        self.idg = ImageDataGenerator(**augment_options)
        self.i = 0
        self.index = np.arange(len(images))
        if shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        return self.next()

    def next(self):
        start = self.i
        end = min(start + self.batch_size, len(self.images))

        augmented_images = []
        augmented_masks = []
        for n in self.index[start:end]:
            image = self.images[n]
            mask = self.masks[n]

            _, _, channels = image.shape

            stacked = np.concatenate((image, mask), axis=2)

            augmented = self.idg.random_transform(stacked)

            augmented_image = augmented[:,:,:channels]
            augmented_images.append(augmented_image)
            augmented_mask = np.round(augmented[:,:,channels:])
            augmented_masks.append(augmented_mask)

        self.i += self.batch_size
        if self.i >= len(self.images):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        return np.asarray(augmented_images), np.asarray(augmented_masks)


def normalize(x, epsilon=1e-7, axis=(1,2)):
    x -= np.mean(x, axis=axis, keepdims=True)
    x /= np.std(x, axis=axis, keepdims=True) + epsilon


def create_generators(data_dir, batch_size, validation_split=0.0, mask='both',
                      augmentation_args={}):
    images, masks = load_images(data_dir, mask)

    images = images.astype('float64')
    split_index = int(float(sys.argv[4])*(1-validation_split) * len(images))

    idg = ImageDataGenerator()
    train_generator = idg.flow(images[:split_index], masks[:split_index],
                               batch_size=batch_size)

    train_steps_per_epoch = ceil(split_index / batch_size)

    if validation_split > 0.0:
        idg = ImageDataGenerator()
        val_generator = idg.flow(images[split_index:], masks[split_index:],
                                 batch_size=batch_size)
    else:
        val_generator = None

    val_steps_per_epoch = ceil((len(images) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch,
            val_generator, val_steps_per_epoch)














# Optimizers
def select_optimizer(optimizer_name, optimizer_args):
    optimizers = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adam': Adam,
    }
    return optimizers[optimizer_name](**optimizer_args)



# Training
def train():

    print("Data Loading and preprocessing.....")
    augmentation_args = {
        'rotation_range': 180,
    }
    train_generator, train_steps_per_epoch, \
        val_generator, val_steps_per_epoch = create_generators(
            'TrainingSet', 8,
            validation_split= 0.5,
            mask='both',
            augmentation_args=augmentation_args)

    images, masks = next(train_generator)
    _, height, width, channels = images.shape
    _, _, _, classes = masks.shape

    string_to_model = {
        "unet": models.unet,
    }

    m = unet(height=height, width=width, channels=channels, classes=classes,
              features=32, depth=3, padding='same',
              temperature=1.0)

    # m.summary()

    callbacks = []

    optimizer_args = {
        'lr': float(sys.argv[3]),
    }


    optimizer = select_optimizer(sys.argv[5], optimizer_args)


    # Change loss function here for simulations
    def lossfunc(y_true, y_pred):
        return weighted_categorical_crossentropy(
            y_true, y_pred, [0.1,0.9])


    def dice(y_true, y_pred):
        batch_dice_coefs = sorensen_dice(y_true, y_pred, axis=[1, 2])
        dice_coefs = K.mean(batch_dice_coefs, axis=0)
        return dice_coefs[1]



    metrics = ['accuracy', dice]

    m.compile(optimizer=optimizer, loss=lossfunc, metrics=metrics)

    print("Training started ........")
    history = m.fit_generator(train_generator,
                    epochs=10,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    callbacks=callbacks,
                    verbose=2)

    if os.path.isfile('dice.csv'):
      df_dice=pd.read_csv('dice.csv')
      df_dice[str(sys.argv[3])+'+'+str(sys.argv[4])]=history.history['dice']
      df_dice.to_csv('dice.csv',index=False)
      df_val=pd.read_csv('val_dice.csv')
      df_val[str(sys.argv[3])+'+'+str(sys.argv[4])]=history.history['val_dice']
      df_val.to_csv('val_dice.csv',index=False)


    else:
      df=pd.DataFrame(history.history['dice'],columns=[str(sys.argv[3])+'+'+str(sys.argv[4])])
      df.to_csv('dice.csv')
      df=pd.DataFrame(history.history['val_dice'],columns=[str(sys.argv[3])+'+'+str(sys.argv[4])])
      df.to_csv('val_dice.csv')

if __name__ == '__main__':
    train()
