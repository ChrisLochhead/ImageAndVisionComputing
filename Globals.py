from matplotlib import image
from os import listdir
import numpy as np

#Common labels
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
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


#Load in all images in a path
def load_images(path): 
    #Containers for normal images for manipulation, flat images for SVM and class labels separate
    images = []
    flat_images = []
    class_labels = []
    for label in labels:
        #Iterate each folder and add the corresponding images
        for filename in listdir(path + label):
            image_data = image.imread(path + label + '/' + filename)
            images.append(image_data)
            flat_images.append(image_data.flatten())
            class_labels.append(label)
    #Return the images as numpy arrays
    return np.array(images), np.array(flat_images), np.array(class_labels)