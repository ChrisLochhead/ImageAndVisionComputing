

import torch
import torchvision
from matplotlib import image
from matplotlib import pyplot
import PIL
from PIL import Image
import os
from os import getlogin, listdir, pardir
import numpy as np
from numpy import append, random
import copy
# %matplotlib inline
from skimage.util import random_noise
import dlib
import random as Rand

import cv2
import math
 
#Storage for all noise perturbed images at every scale specified in the coursework brief
scales = [0,2,4,6,8,10,12,14,16,18]
#Contrast increase/decrease specific variables
contrast_multipliers_up = [1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.21, 1.24, 1.27]
contrast_multipliers_down = [ 1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10 ]
#brightness increase/decrease specific variables
brightness_modifiers_up = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
brightness_modifiers_down = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45]
#occlusion increase specific variables
occlusion_box_size = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
#Salt and pepper frequency variable
sp_intensity = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
#Gaussian Blur variables
blur_sd = [0,1,2,3,4,5,6,7,8,9]
#Common labels
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

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
    print(len(images), len(images[0]), len(images[1]))
    return images

def clip_pixel(value):
    if value < 0: 
        value = 0
    elif value > 255:
        value = 255
    return int(value)

def gaussian_noise(args, pixel):
    sd_value = random.normal(scale=args["modifier"])
    sd_value = clip_pixel(sd_value)
    return sd_value + pixel["pixel"]

def contrast_pixel(args, pixel):
    m_value = args["modifier"] * pixel["pixel"]
    m_value = clip_pixel(m_value)
    return m_value

# this was simplified to assume if brigtness decreased a negative modifier was given
def modify_pixel_brightness(args, pixel):
    m_value = args["modifier"] + pixel["pixel"]
    m_value = clip_pixel(m_value)
    return m_value

def occlude_pixel(args, pix):
    current_loc = [pix["row_idx"], pix["pix_idx"]]
    m_value = pix["pixel"]
    
    #If its within the occlusion zone set the pixel to black
    if current_loc[0] == args["occ_idx"][0] or current_loc[0] < args["occ_idx"][0] + args["modifier"] and current_loc[0] > args["occ_idx"][0]:
        if current_loc[1] == args["occ_idx"][1] or current_loc[1] < args["occ_idx"][1] + args["modifier"] and current_loc[1] > args["occ_idx"][1]:
            m_value = 0
    
    return m_value

def loop_pixels(orig_image, new_image, perturb_func, args):
    for (row_idx, pixel_row) in enumerate(orig_image):
        for (pix_idx, pixel) in enumerate(pixel_row):
            new_image[row_idx][pix_idx] = perturb_func(args, {"pixel":pixel, "row_idx":row_idx, "pix_idx":pix_idx})

def salt_and_pepper(original_image, modifier):
    new_image = random_noise(original_image, mode="s&p", amount=modifier)
    return new_image
    
def occlude(original_image, modifier):
    # set a random number to be the starting point for an occlusion square
    occlusion_index = [0,0]
    occlusion_index[0] = Rand.randrange(0, 48)
    occlusion_index[1] = Rand.randrange(0, 48)

    new_image = np.zeros([48, 48])
    loop_pixels(original_image, new_image, occlude_pixel, {"modifier":modifier, "occ_idx":occlusion_index})
    return new_image

def modify_contrast(original_image, modifier):
    new_image = np.zeros([48, 48])
    loop_pixels(original_image, new_image, contrast_pixel, {"modifier":modifier})
    return new_image

def modify_brightness(original_image, modifier):
    new_image = np.zeros([48, 48])
    loop_pixels(original_image, new_image, modify_pixel_brightness, {"modifier":modifier})
    return new_image

def scale(original_image, modifier):
    new_image = np.zeros([48, 48])
    loop_pixels(original_image, new_image, gaussian_noise, {"modifier":modifier})
    return new_image

class Image_Manipulator():

    def __init__(self, create_images = True):
        #Load in the normal images
        #self.training_images = load_images('train/')
        self.test_images = load_images('test/')

        #Get new image container for operations to be conducted
        self.perturbed_images = copy.deepcopy(self.test_images)
        
        #Gaussian pixel noise specific variables
        self.noise_perturbed_images = copy.deepcopy(self.perturbed_images)
        #Contrast increase/decrease specific variables
        self.contrast_up_perturbed_images = copy.deepcopy(self.perturbed_images) 
        self.contrast_down_perturbed_images = copy.deepcopy(self.perturbed_images) 
        #brightness increase/decrease specific variables
        self.brightness_down_perturbed_images = copy.deepcopy(self.perturbed_images)
        self.brightness_up_perturbed_images = copy.deepcopy(self.perturbed_images)
        #occlusion increase specific variables
        self.occlusion_perturbed_images = copy.deepcopy(self.perturbed_images)
        #Salt and Pepper specific variables
        self.sp_perturbed_images = copy.deepcopy(self.perturbed_images) 
        #Gaussian Blur variables
        self.blur_perturbed_images = copy.deepcopy(self.perturbed_images) 

        #Print examples to make sure it works
        #train_example = self.training_images[0][0]
        #test_example = self.test_images[2][5]
        #pyplot.imshow(train_example)
        #pyplot.show()
        #pyplot.imshow(test_example)
        #pyplot.show()

        #Check if images dont already exist, otherwise
        if create_images == True:
            print("generating images")
            self.initialise_permutations()

    def get_images(self, path_folder):
        image_collection = list()
        #print("starting get images")
        for mod_lvl_subdir in os.listdir(path_folder):
            mod_list = list()
            #print("new modification: ", mod_lvl_subdir)
            expression_path = os.path.join(path_folder, mod_lvl_subdir)
            for expression_lvl_subdir in os.listdir(expression_path):
                #print("new expression: ", expression_lvl_subdir)
                expression_list = list()
                for image_file in os.scandir(os.path.join(expression_path, expression_lvl_subdir)):
                    if image_file.path.endswith(".jpg") and image_file.is_file():
                        expression_list.append(image.imread(os.path.join(expression_lvl_subdir, image_file)))
                mod_list.append(expression_list)
            image_collection.append(mod_list)
        return image_collection

        #print("image load completed ")

        #test_example = image_collection[0][0][5]
        #pyplot.imshow(test_example)
        #pyplot.show()   

        #test_example = image_collection[2][1][5]
        #pyplot.imshow(test_example)
        #pyplot.show()  

        #test_example = image_collection[4][2][5]
        #pyplot.imshow(test_example)
        #pyplot.show()  

        #test_example = image_collection[5][3][5]
        #pyplot.imshow(test_example)
        #pyplot.show()  



    def create_perturbed_images(self, path_folder, modifier_type, location, perturb_func):
        if not os.path.isdir(os.getcwd() + path_folder):
            os.mkdir(os.getcwd() + path_folder)
        self.loop_all_images(self.test_images, perturb_func, location, \
        modifier_type, os.getcwd() + path_folder)

    def initialise_permutations(self):
        #For each of the 8 mutations, starting with gaussian noise 
        #If the first results are missing, initialise all results folders
        if not os.path.isdir(os.getcwd() + "/results/gaussian_noise"):
            os.makedirs(os.getcwd() + "/results/gaussian_noise")
            os.makedirs(os.getcwd() + "/results/contrast_up")
            os.makedirs(os.getcwd() + "/results/contrast_down")
            os.makedirs(os.getcwd() + "/results/brightness_up")
            os.makedirs(os.getcwd() + "/results/brightness_down")
            os.makedirs(os.getcwd() + "/results/occlusion")
            os.makedirs(os.getcwd() + "/results/salt_and_pepper")
            os.makedirs(os.getcwd() + "/results/gaussian_blur")

        #Iterate through all modifications and all modification settings, save and store all
        #resulting images for reloading later.
    
        for scale_modifier in scales:
            self.create_perturbed_images("/results/gaussian_noise/" + str(scale_modifier),
             scale_modifier, self.noise_perturbed_images, scale)
        print("scale done")
        for contrast_mod in contrast_multipliers_up:
            self.create_perturbed_images("/results/contrast_up/" + str(contrast_mod),
             contrast_mod, self.contrast_up_perturbed_images, modify_contrast)
        print("contrast done")
        for contrast_mod in contrast_multipliers_down:
            self.create_perturbed_images("/results/contrast_down/" + str(contrast_mod),
             contrast_mod, self.contrast_down_perturbed_images, modify_contrast)
        print("contrast done")
        for brightness_mod in brightness_modifiers_up:
            self.create_perturbed_images("/results/brightness_up/" + str(brightness_mod),
             brightness_mod, self.brightness_up_perturbed_images, modify_brightness)
        print("brightness done")
        for brightness_mod in brightness_modifiers_down:
            self.create_perturbed_images("/results/brightness_down/" + str(brightness_mod),
             brightness_mod, self.brightness_down_perturbed_images, modify_brightness)
        print("brightness done")
        for occlusion_mod in occlusion_box_size:
            self.create_perturbed_images("/results/occlusion/" + str(occlusion_mod),
             occlusion_mod, self.occlusion_perturbed_images, occlude)
        print("occlusion done")
        for sp_mod in sp_intensity:
            self.create_perturbed_images("/results/salt_and_pepper/" + str(sp_mod),
             sp_mod, self.sp_perturbed_images, salt_and_pepper)
        print("salt and pepper done")
        for blur_mod in blur_sd:
            self.create_perturbed_images("/results/gaussian_blur/" + str(blur_mod),
             blur_mod, self.blur_perturbed_images, gaussian_blur)

        print("all images completed.")



    # clean images = copy of original images, perturb fun = function to perturb image with, perturbed_images = list to put modified images in, modifier = modifier value
    def loop_all_images(self, clean_images, perturb_func, perturbed_images, modifier, path):
        image_index = 0
        folder_index = 0


        for image_folder in clean_images:
            image_index = 0

            if not os.path.isdir(path + "/" + labels[folder_index]):
                os.mkdir(path + "/" + labels[folder_index])

            for ind_image in image_folder:
                new_image = perturb_func(ind_image, modifier)
                #print("range:", folder_index, image_index)
                perturbed_images[folder_index][image_index] = new_image
                image.imsave(path + "/" + labels[folder_index] + "/" + str(image_index) + ".jpg", new_image)
                image_index += 1
            folder_index += 1

        

def robustness_exploration():

    #Initialise an image manipulator (set to false if images already made on device)
    manipulator = Image_Manipulator(False)
    test_path = os.getcwd() + "/results/gaussian_noise/"
    manipulator.get_images(test_path)

    #Test the modifications
    #Gaussian noise
    #test_image = scale(test_example, scales[9])
    #pyplot.imshow(test_image)
    #pyplot.show()

    #Contrast up then down
    #test_image = modify_contrast(test_example, contrast_multipliers_up[5])
    #pyplot.imshow(test_image)
    #pyplot.show()
    #test_image = modify_contrast(test_example, contrast_multipliers_down[5])
    #pyplot.imshow(test_image)
    #pyplot.show()

    #Brightness up then down
    #test_image = modify_brightness(test_example, brightness_modifiers[8])
    #pyplot.imshow(test_image)
    #pyplot.show()
    #test_image = modify_brightness(test_example, -brightness_modifiers[8])
    #pyplot.imshow(test_image)
    #pyplot.show()

    #Occlusion box
    #test_image = occlude(test_example, occlusion_box_size[3])
    #pyplot.imshow(test_image)
    #pyplot.show()

    #Salt and pepper
    #test_image = salt_and_pepper(test_example, sp_intensity[7])
    #pyplot.imshow(test_image)
    #pyplot.show()

    #Gaussian blur
    #test_image = gaussian_blur(test_example, blur_sd[4])
    #pyplot.imshow(test_image)
    #pyplot.show()


#def get_all_image_landmarks(imgs, predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")):
#    detector = dlib.get_frontal_face_detector()
#
#    processed = list()#
#
#    for i in imgs:
#        processed_images = list()
#        for j in i:
#            dets = detector(j, 2)
#            if (len(dets) != 1):
#                continue
#            shape = predictor(j, dets[0])
#            if len(shape.parts()) != 68:
#                continue
#            processed_images.append(shape.parts())
#        processed.append(processed_images)
#    
#    return processed



def convolution(image, kernel, average=False):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
 
    output = np.zeros(image.shape)
 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
 
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output

def gaussian_kernel():
    kernel_2D = np.array([[1,2,1],[2,4,2],[1,2,1]]) 
    return kernel_2D
 
 
def gaussian_blur(image, modifier):
    kernel = gaussian_kernel()
    convolved_image = image
    for i in range(0, modifier):
        convolved_image = convolution(convolved_image, kernel, average=True)
    return convolved_image
 

if __name__ == "__main__":
    robustness_exploration()
    #training_images = load_images('train/')
    #result = get_all_image_landmarks(training_images)
    print("done")

#Implement SVM

#Implement Resnet18