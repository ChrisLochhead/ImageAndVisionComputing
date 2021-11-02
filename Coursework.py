import torch
import torchvision
from matplotlib import image
from matplotlib import pyplot
import PIL
from PIL import Image
import os
from os import listdir
import numpy as np
from numpy import random
import copy
# %matplotlib inline
from skimage.util import random_noise

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Load all of training and testing into arrays

#Common labels
labels = ['angry', 'disgust', 'fear',
                   'happy', 'neutral', 'sad', 'surprise']

#Load in all images in a path
def load_images(path): 
    #Primary image container
    images = list()
    
    for label in labels:
        #Sub list container for each emotion
        sub_list = list()
        #Iterate each folder and add the corresponding images
        for filename in listdir(path + label):
            image_data = image.imread(path + label + '/' + filename)
            sub_list.append(image_data)
        #Append sub-list to the master list
        images.append(sub_list)
    #Return the images as a list of lists
    return images

#Load training and testing images
training_images = load_images('train/')
test_images = load_images('test/')

#Print examples to make sure it works
train_example = training_images[0][0]
test_example = test_images[2][5]
pyplot.imshow(train_example)
pyplot.show()
pyplot.imshow(test_example)
pyplot.show()

#Implement progressive perturbations

#Get new image container for operations to be conducted
perturbed_images = copy.deepcopy(test_images)
#Image template container
new_image = np.zeros([48, 48])

#Gaussian pixel noise specific variables
#Storage for all noise perturbed images at every scale specified in the coursework brief
scales = [0,2,4,6,8,10,12,14,16,18]
noise_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))

#Contrast increase/decrease specific variables
contrast_multipliers_up = [1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.21, 1.24, 1.27]
contrast_multipliers_down = [ 1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10 ]
contrast_up_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))
contrast_down_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))

#brightness increase/decrease specific variables
brightness_modifiers = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
brightness_down_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))
brightness_up_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))

#occlusion increase specific variables
occlusion_box_size = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
occlusion_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))

#Salt and Pepper specific variables
sp_intensity = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
sp_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))

#Gaussian Blur variables
blur_template = [[1,2,1],[2,4,2],[1,2,1]]
blur_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))

#For every scale to apply gaussian noise at
for scale in scales:
    loop_all_images('scale', scale)
    #Start with the first folder
    #folder_index = 0
    #for original_image_folder in perturbed_images:
    #    #from that folder, find the first image
    #    image_index = 0
    #    for original_image in original_image_folder:
    #        #from that image, get the first row of pixels
    #        index = [0,0]
    #        for pixel_row in original_image:
    #            #from that row, get the first pixel and apply the transformation
    #            index[1] = 0
    #            for pixel in pixel_row:
    #                sd_value = gaussian_noise(scale)
    #                new_image[index[0]][index[1]] = pixel + sd_value
    #                index[1] += 1
    #            index[0] += 1
    #        print("image: ", folder_index, image_index)
    #        perturbed_images[folder_index][image_index] = new_image
    #        pyplot.imshow(perturbed_images[folder_index][image_index])
    #        pyplot.show()
    #        image_index += 1
    #    folder_index += 1#

def clip_pixel(value):
    if value < 0: 
        value = 0
    elif value > 255:
        value = 255
    return int(value)

def gaussian_noise(scale):
    sd_value = random.normal(scale=scale)
    sd_value = clip_pixel(sd_value)
    return sd_value

def contrast_pixel(multiplier, pixel):
    m_value = multiplier * pixel
    m_value = clip_pixel(m_value)
    return m_value

def modify_brightness(mode, modifier, pixel):
    if mode == 'brightness_up':
        m_value = modifier + pixel
    else:
        m_value = modifier - pixel
    m_value = clip_pixel(m_value)
    return m_value

def occlude_image(mod, top_left, image, row_index, pixel_index, pixel):
    current_loc = [row_index, pixel_index]
    m_value = pixel
    modifier = occlusion_box_size[mod]
    
    #If its within the occlusion zone set the pixel to black
    if current_loc[0] == top_left[0] or \
    current_loc[0] < top_left[0] + modifier and current_loc[0] > top_left[0]:
        if current_loc[1] == top_left[1] or \
        current_loc[1] < top_left[1] + modifier and current_loc[1] > top_left[1]:
            m_value = 0
    
    return m_value
        
def sp_image(image, modificaton_index):
    noisy_image = random_noise(image, var=sp_intensity[modification_index])
    return noisy_image

def loop_all_images(modification, modification_index):

    #Start with the first folder
    folder_index = 0

    for original_image_folder in perturbed_images:
        #from that folder, find the first image
        image_index = 0

        for original_image in original_image_folder:
            #from that image, get the first row of pixels
            index = [0,0]
            
            if modification == 'sap':
                new_image = random_noise(original, var=sigma**2)
            else:
                #If occlusion, set a random number to be the starting point for an occlusion square
                if modification == 'occlude':
                    occlusion_index = [0,0]
                    occlusion_index[0] = random.randrange(0, 48)
                    occlusion_index[1] = random.randrange(0, 48)

                for pixel_row in original_image:
                    #from that row, get the first pixel and apply the transformation
                    index[1] = 0

                    for pixel in pixel_row:
                        if modification == 'scale':                       
                            applied_value = gaussian_noise(modification_index)
                            new_image[index[0]][index[1]] = pixel + applied_value
                        elif modification == 'contrast_up' or modification == 'contrast_down':
                            new_image[index[0]][index[1]] = contrast_pixel(modification_index, pixel)
                        elif modification == 'brightness_up' or modification == 'brightness_down':
                            new_image[index[0]][index[1]] = modify_brightness(modification, modification_index, pixel)
                        elif modification == 'occlude':
                            new_image[index[0]][index[1]] = occlude_image(modification_index, occlusion_index, original_image, \
                            index[0], index[1], pixel)
                        index[1] += 1

                index[0] += 1
            if modification == 'scale':
                noise_perturbed_images[folder_index][image_index] = new_image
            elif modification == 'contrast_up':
                contrast_up_perturbed_images[folder_index][image_index] = new_image
            elif modification == 'contrast_down':
                contrast_down_perturbed_images[folder_index][image_index] = new_image
            elif modification == 'brightness_up':
                brightness_up_perturbed_images[folder_index][image_index] = new_image
            elif modification == 'brightness_down':
                brightness_down_perturbed_images[folder_index][image_index] = new_image
            elif modification == 'occlude':
                occlusion_perturbed_images[folder_index][image_index] = new_image
            elif modification == 'sap':
                sp_perturbed_images[folder_index][image_index] = new_image
            image_index += 1

        folder_index += 1
    

#develop feature descriptor using historgram of gradients with dlib

#Implement SVM

#Implement Resnet18