import copy
import os

import random as Rand
import numpy as np

from numpy import random
from multiprocessing import Process
from matplotlib import image
from skimage.util import random_noise

from Globals import *

#Remove negative or abnormal values and constrain them within the RGB format
def clip_pixel(value):
    if value < 0: 
        value = 0
    elif value > 255:
        value = 255
    return int(value)

#Apply a random value from a normal distribution to the pixel
def gaussian_noise(args, pixel):
    sd_value = random.normal(scale=args["modifier"])
    sd_value = clip_pixel(sd_value)
    return sd_value + pixel["pixel"]

#Increase the contrast by multiplying the pixel by the specified modifier
def contrast_pixel(args, pixel):
    m_value = args["modifier"] * pixel["pixel"]
    m_value = clip_pixel(m_value)
    return m_value

#Modify brightness by applying the corresponding modifier to the pixel
def modify_pixel_brightness(args, pixel):
    m_value = args["modifier"] + pixel["pixel"]
    m_value = clip_pixel(m_value)
    return m_value

#Check if the pixel is being occluded and black it out if so
def occlude_pixel(args, pix):
    current_loc = [pix["row_idx"], pix["pix_idx"]]
    m_value = pix["pixel"]
    
    #If its within the occlusion zone set the pixel to black
    if current_loc[0] == args["occ_idx"][0] or current_loc[0] < args["occ_idx"][0] + args["modifier"] and current_loc[0] > args["occ_idx"][0]:
        if current_loc[1] == args["occ_idx"][1] or current_loc[1] < args["occ_idx"][1] + args["modifier"] and current_loc[1] > args["occ_idx"][1]:
            m_value = 0
    
    return m_value

#Iterate through all pixels in the image, applying the relelvant perturbation function
def loop_pixels(orig_image, new_image, perturb_func, args):
    for (row_idx, pixel_row) in enumerate(orig_image):
        for (pix_idx, pixel) in enumerate(pixel_row):
            new_image[row_idx][pix_idx] = perturb_func(args, {"pixel":pixel, "row_idx":row_idx, "pix_idx":pix_idx})

#Apply random noise to the image for a salt and pepper effect
def salt_and_pepper(original_image, modifier):
    new_image = random_noise(original_image, mode="s&p", amount=modifier)
    return new_image

#Create a randomly placed sqaure of a size determined by the modifier and black out all the pixels in that area.    
def occlude(original_image, modifier):
    occlusion_index = [0,0]
    occlusion_index[0] = Rand.randrange(0, 48)
    occlusion_index[1] = Rand.randrange(0, 48)
    new_image = np.zeros([48, 48])
    loop_pixels(original_image, new_image, occlude_pixel, {"modifier":modifier, "occ_idx":occlusion_index})
    return new_image

#Apply a contrast modifier 
def modify_contrast(original_image, modifier):
    new_image = np.zeros([48, 48])
    loop_pixels(original_image, new_image, contrast_pixel, {"modifier":modifier})
    return new_image

#Apply a brightness modifier
def modify_brightness(original_image, modifier):
    new_image = np.zeros([48, 48])
    loop_pixels(original_image, new_image, modify_pixel_brightness, {"modifier":modifier})
    return new_image

#Change the blur of the image dependant on scale
def scale(original_image, modifier):
    new_image = np.zeros([48, 48])
    loop_pixels(original_image, new_image, gaussian_noise, {"modifier":modifier})
    return new_image

def convolution(image, kernel, average=False):
    #Initialise iterators and output
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    
    #Padd the image 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    #Iterate the image, applying the blur to each pixel
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output

#Apply the Gaussian blur
def gaussian_blur(image, modifier):
    kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])
    convolved_image = image
    for i in range(0, modifier):
        convolved_image = convolution(convolved_image, kernel, average=True)
    return convolved_image
 
class Image_Manipulator():

    def __init__(self, create_images = True):
        #Load in the normal images
        print("loading images")
        self.training_images, self.flat_training_images, self.train_labels = load_images('train/')
        #Print examples to make sure it works
        #train_example = resize(self.training_images[0], (48, 48))
        #pyplot.imshow(train_example)
        #pyplot.show()
        self.test_images, self.flat_test_images, self.test_labels = load_images('test/')
        print("image loading complete")
 

        #Check if images dont already exist, otherwise
        if create_images == True:
            print("generating images")
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
            self.initialise_permutations()

    def get_images(self, path_folder):
        image_collection = list()
        for mod_lvl_subdir in os.listdir(path_folder):
            mod_list = list()
            expression_path = os.path.join(path_folder, mod_lvl_subdir)
            for expression_lvl_subdir in os.listdir(expression_path):
                expression_list = list()
                for image_file in os.scandir(os.path.join(expression_path, expression_lvl_subdir)):
                    if image_file.path.endswith(".jpg") and image_file.is_file():
                        expression_list.append(image.imread(os.path.join(expression_lvl_subdir, image_file)))
                mod_list.append(expression_list)
            image_collection.append(mod_list)
        return image_collection

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

        processes = []
        for scale_modifier in scales:
            p = Process(target=self.create_perturbed_images, args=("/results/gaussian_noise/" + str(scale_modifier), scale_modifier, self.noise_perturbed_images, scale))
            p.start()
            processes.append(p)
            print("Starting scale process")

        for procs in processes:
            procs.join()
        print("scale done")

        processes = []
        for contrast_mod in contrast_multipliers_up:
            p = Process(target=self.create_perturbed_images, args=("/results/contrast_up/" + str(contrast_mod), contrast_mod, self.contrast_up_perturbed_images, modify_contrast))
            p.start()
            processes.append(p)
            print("Starting Contrast up process")

        for procs in processes:
            procs.join()
        print("Contrast up done")

        processes = []
        for contrast_mod in contrast_multipliers_down:
            p = Process(target=self.create_perturbed_images, args=("/results/contrast_down/" + str(contrast_mod), contrast_mod, self.contrast_down_perturbed_images, modify_contrast))
            p.start()
            processes.append(p)
            print("Starting Contrast down process")

        for procs in processes:
            procs.join()
        print("Contrast down done")

        processes = []
        for brightness_mod in brightness_modifiers_up:
            p = Process(target=self.create_perturbed_images, args=("/results/brightness_up/" + str(brightness_mod), brightness_mod, self.brightness_up_perturbed_images, modify_brightness))
            p.start()
            processes.append(p)
            print("Starting brightness up process")

        for procs in processes:
            procs.join()
        print("Brightness up done")


        processes = []
        for brightness_mod in brightness_modifiers_down:
            p = Process(target=self.create_perturbed_images, args=("/results/brightness_down/" + str(brightness_mod), brightness_mod, self.brightness_down_perturbed_images, modify_brightness))
            p.start()
            processes.append(p)
            print("Starting brightness down process")

        for procs in processes:
            procs.join()
        print("Brightness down done")

        processes = []
        for occlusion_mod in occlusion_box_size:
            p = Process(target=self.create_perturbed_images, args=("/results/occlusion/" + str(occlusion_mod), occlusion_mod, self.occlusion_perturbed_images, occlude))
            p.start()
            processes.append(p)
            print("Starting occlusion process")

        for procs in processes:
            procs.join()
        print("occlusion done")


        processes = []
        for sp_mod in sp_intensity:
            p = Process(target=self.create_perturbed_images, args=("/results/salt_and_pepper/" + str(sp_mod), sp_mod, self.sp_perturbed_images, salt_and_pepper))
            p.start()
            processes.append(p)
            print("Starting sp process")

        for procs in processes:
            procs.join()
        print("sp done")


        processes = []
        for blur_mod in blur_sd:
            p = Process(target=self.create_perturbed_images, args=("/results/gaussian_blur/" + str(blur_mod), blur_mod, self.blur_perturbed_images, gaussian_blur))
            p.start()
            processes.append(p)
            print("Starting blur process")

        for procs in processes:
            procs.join()
        print("blur done")


        print("all images generated.")

    #Utility function to loop through all the newly created images and save them all for later use
    def loop_all_images(self, clean_images, perturb_func, perturbed_images, modifier, path):
        image_index = 0
        folder_index = 0
        
        for image_folder in clean_images:
            image_index = 0

            if not os.path.isdir(path + "/" + labels[folder_index]):
                os.mkdir(path + "/" + labels[folder_index])

            for ind_image in image_folder:
                new_image = perturb_func(ind_image, modifier)
                perturbed_images[folder_index][image_index] = new_image
                image.imsave(path + "/" + labels[folder_index] + "/" + str(image_index) + ".jpg", new_image)
                image_index += 1
            folder_index += 1