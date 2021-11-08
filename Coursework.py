from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

############################################################

import torch
import torchvision
from matplotlib import image
from matplotlib import pyplot
import PIL
from PIL import Image
import os
from os import getlogin, listdir
import numpy as np
from numpy import random, singlecomplex
import copy
# %matplotlib inline
from skimage.util import random_noise
import dlib

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
    modifier = occlusion_box_size[args["modifier"]] # I am not quite sure what this is meant to be, I can't find any ref to it in your code
    
    #If its within the occlusion zone set the pixel to black
    if current_loc[0] == args["occ_idx"][0] or current_loc[0] < args["occ_idx"][0] + modifier and current_loc[0] > args["occ_idx"][0]:
        if current_loc[1] == args["occ_idx"][1] or current_loc[1] < args["occ_idx"][1] + modifier and current_loc[1] > args["occ_idx"][1]:
            m_value = 0
    
    return m_value

# Don't think this function is ever used anywhere? Delete if I am right
# def sp_image(image, modification_index):
#     noisy_image = random_noise(image, var=sp_intensity[modification_index])
#     return noisy_image

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
    occlusion_index[0] = random.randrange(0, 48)
    occlusion_index[1] = random.randrange(0, 48)

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

# clean images = copy of original images, perturb fun = function to perturb image with, perturbed_images = list to put modified images in, modifier = modifier value
def loop_all_images(clean_images, perturb_func, perturbed_images, modifier):
    for (f_idx, orig_image_folder) in enumerate(clean_images):
        for (im_idx, orig_image) in enumerate(orig_image_folder):
            new_image = perturb_func(orig_image, modifier)
            perturbed_images[f_idx][im_idx] = new_image


def robustness_exploration():
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
    
    #Gaussian pixel noise specific variables
    #Storage for all noise perturbed images at every scale specified in the coursework brief
    scales = [0,2,4,6,8,10,12,14,16,18]
    noise_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))
    # noise_perturb(list(copy.deepcopy(perturbed_images) for i in range(len(scales))))


    #Contrast increase/decrease specific variables
    contrast_multipliers_up = [1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.21, 1.24, 1.27]
    contrast_multipliers_down = [ 1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10 ]
    contrast_up_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))
    contrast_down_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))

    #brightness increase/decrease specific variables
    brightness_modifiers = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    brightness_down_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))
    brightness_up_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))

    #Salt and Pepper specific variables
    sp_intensity = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    sp_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))

    #Gaussian Blur variables
    blur_template = [[1,2,1],[2,4,2],[1,2,1]]
    blur_perturbed_images = list(copy.deepcopy(perturbed_images) for i in range(len(scales)))

    #For every scale to apply gaussian noise at
    for scale in scales:
        # loop_all_images('scale', scale)
        #Start with the first folder
        folder_index = 0
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

# URL of model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
def get_all_image_landmarks(imgs, predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")):
    detector = dlib.get_frontal_face_detector()
    processed = list()

    for i in imgs:
        processed_images = list()
        for j in i:
            dets = detector(j, 2)
            if (len(dets) != 1):
                continue
            shape = predictor(j, dets[0])
            if len(shape.parts()) != 68:
                continue
            processed_images.append(shape.parts())
        processed.append(processed_images)
    
    return processed

##################################not my code in this section
from torchvision import transforms

def test_model(model):
    input_image = Image.open("Test_img.jpg")

    # if (input_image.mode != "RGB"):
    #     input_image = input_image.convert("RGB")

    preprocess = transforms.Compose([
        transforms.Lambda(lambda x : x.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")
        print("Cuda activated")

    with torch.no_grad():
        output = model(input_batch)
    
    print(output[0])

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)

    # top5_prob, top5_catid = torch.topk(probabilities, 7)
    # for i in range(top5_prob.size(0)):
    #     print(labels[top5_catid[i]], top5_prob[i].item())


## this is teh fuynction that decides what to newly train adn what not to - currently freezes everything currently in network I think?
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False   


def train_model(model, dataloaders, criterion, optimizer, num_epochs=32, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if is_inception and phase == "train":
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train": 
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.sum(preds == labels.data))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def get_resnet_model():
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True) ## is a 1000 output net as it was trained as 1000 image classifier
    model.eval()    
    return model

def freeze_layers(model):
    for layer in model.parameters():
        layer.requires_grad = False

def add_new_layers(model):
    model.fc = torch.nn.Linear(512, len(labels))

def resnet():
    model = get_resnet_model()
    freeze_layers(model)
    add_new_layers(model)





    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")
        print("Cuda activated")

### https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
### https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
### https://pytorch.org/hub/pytorch_vision_resnet/
### https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

############################################################### not my code in above section


if __name__ == "__main__":
    # robustness_exploration()
    # training_images = load_images('train/')
    # result = get_all_image_landmarks(training_images)
    # print("done")

    test_model(get_resnet_model())

#Implement SVM

#Implement Resnet18