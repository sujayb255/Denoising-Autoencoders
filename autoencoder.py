#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
print("All libraries loaded")


# In[3]:


train_fpath = "/train"
train_cleaned_fpath = "/train_cleaned"
test_fpath = "/test"
print(os.listdir(train_fpath))


# In[4]:


print("No. of files in train folder = ",len(os.listdir(train_fpath)))
print("\nNo. of files in train_cleaned folder = ",len(os.listdir(train_cleaned_fpath)))
print("\nNo. of files in test folder = ",len(os.listdir(test_fpath)))


# In[5]:


def load_images(fpath):
    images = []
    for image in os.listdir(fpath):
        #print(fpath+image)
        if image!='train' and image!='train_cleaned' and image!='test':
            img = cv2.imread(fpath+image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_array = Image.fromarray(img, "RGB")

            resized_img = img_array.resize((252,252))

            images.append(np.array(resized_img))
    return images

train_images = load_images(train_fpath)
train_images = np.array(train_images)
print("No. of images loaded = ",len(train_images),"\nShape of the images loaded = ",train_images[0].shape)


# In[6]:


train_cleaned_images = load_images(train_cleaned_fpath)
train_cleaned_images = np.array(train_cleaned_images)
print("No. of images loaded = ",len(train_cleaned_images),"\nShape of the images loaded = ",train_cleaned_images[0].shape)


# In[7]:


test_images = load_images(test_fpath)
test_images = np.array(test_images)
print("No. of images loaded = ",len(test_images),"\nShape of the images loaded = ",test_images[0].shape)


# In[8]:


def display_images(images):
    n = 3
    plt.figure(figsize=(19, 6))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
print("Displaying noisy training images")
display_images(train_images)


# In[9]:


train_images = train_images.astype(np.float32)
train_cleaned_images = train_cleaned_images.astype(np.float32)
test_images = test_images.astype(np.float32)

train_images = train_images/255
train_cleaned_images = train_cleaned_images/255
test_images = test_images/255
print(train_images[0].shape, train_cleaned_images[0].shape, test_images[0].shape)


# In[10]:


print("Displaying noisy training images after normalization")
display_images(train_images)


# In[11]:



print("Displaying clean training images after normalization")
display_images(train_cleaned_images)


# In[12]:


input_img = Input(shape=(252, 252, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')
autoencoder.summary()


# In[13]:


autoencoder.fit(train_images, train_cleaned_images,
                epochs=100,
                batch_size=10,
                shuffle=True)


# In[14]:


predicted_images = autoencoder.predict(test_images)


# In[15]:


print("Displaying noisy test images")
display_images(test_images)


# In[16]:


print("Displaying predicted images for the given test noisy images input")
display_images(predicted_images)

