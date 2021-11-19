import torch
import torch.nn as nn
from matplotlib import image
from PIL import Image
import os
from os import listdir
import numpy as np
from numpy import random
import copy
from skimage.util import random_noise
import dlib
import random as Rand
# from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from multiprocessing import Process
from tqdm import tqdm
import math
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from skimage.feature import hog
from skimage import data, exposure

import pickle


#Common labels
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

########################################################################################################################################
## Image manipulation stuff

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
    SVM_Controller = SVM(manipulator.training_images)
    SVM_Controller.run_SVM(manipulator.test_images)
    # test_path = os.getcwd() + "/results/gaussian_noise/"
    # manipulator.get_images(test_path)

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
 
#################################################################################################################################################

################################################################################################################################################
#DLIB STUFF

# URL of model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
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

##################################################################################################################################################

##################################################################################################################################################
##RESNET SECTION

### https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
### https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
### https://pytorch.org/hub/pytorch_vision_resnet/
### https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
### https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

label_map = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'neutral':4, 'sad':5, 'surprise':6}
my_device = "cuda" if torch.cuda.is_available() else "cpu"

def freeze_layers(model):
    for layer in model.parameters():
        layer.requires_grad = False

def add_new_layers(model, extra_nodes=0):
    if extra_nodes != 0:
        model.fc = torch.nn.Linear(512, extra_nodes)
        model = nn.Sequential(
            model,
            torch.nn.Linear(extra_nodes, len(labels))
        )
    else:
        model.fc = torch.nn.Linear(512, len(labels))

    return model

def get_resnet_model():
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True) ## is a 1000 output net as it was trained as 1000 image classifier
    return model

def flat_load_images(dir="train/"):
    #Primary image container
    my_images = []
    my_labels = []

    for label in labels:
        #Iterate each folder and add the corresponding images
        for filename in listdir(dir + label):
            my_im = Image.open(dir + label + '/' + filename)
            my_images.append(my_im.copy())
            my_labels.append(label)
            my_im.close()
    #Return the images and corresponding labels
    return my_images, my_labels


class CustomDataset(Dataset):
    def __init__(self, dir, transform=None):
        images, labels = flat_load_images(dir)
        self.transform = transform
        self.img_labels = labels
        self.my_images = images

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.my_images[idx])
        else:
            image = self.my_images[idx]

        label = self.img_labels[idx]

        return image, label_map[label] 
        

def get_dataloaders(validation_ratio=0.2):

    preprocess = T.Compose([
        T.Lambda(lambda x : x.convert("RGB")),
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply(transforms=[T.RandomRotation(degrees=(0,360))], p=0.5),
        T.RandomPerspective(distortion_scale=0.5, p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset("train/", transform=preprocess)
    train_dataset, validation_dataset = random_split(train_dataset, [math.ceil(len(train_dataset) * (1-validation_ratio)), math.floor(len(train_dataset) * validation_ratio)])

    test_dataset = CustomDataset("test/")

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False) # no need to shuffle validation data either
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False) # no need to shuffle test data

    return train_dataloader, validation_dataloader, test_dataloader


def train_model(model, t_dataloader, v_dataloader, learn_rate=1e-3, min_val_loss_improvement=0.1, improvement_buffer=5, absolute_max_epochs=2):
    model.train() # set model to training mode
    model.to(my_device) # send model to device

    learning_layers = list() # get the layers that are not to be "frozen"
    for name, layer in model.named_parameters():
        if layer.requires_grad:
            learning_layers.append(layer)
            # print("Training layer: ", name)
    
    loss_func = nn.CrossEntropyLoss()
    optimizer_func = torch.optim.SGD(learning_layers, learn_rate)

    best_loss = 1e10
    best_weights = copy.deepcopy(model.state_dict())

    loss_thresh_epoch = 0
    running_loss_thresh = 1e10

    for epoch in range(absolute_max_epochs):
        model.train()
        running_training_loss = 0
        for (X, y) in tqdm(t_dataloader):
            X, y = X.to(my_device), y.to(my_device)

            prediction = model(X)
            loss = loss_func(prediction, y)
            running_training_loss += loss

            optimizer_func.zero_grad()
            loss.backward()
            optimizer_func.step()

        model.eval()
        num_correct = 0
        running_val_loss = 0
        for (X, y) in v_dataloader:
            X, y = X.to(my_device), y.to(my_device)

            prediction = model(X)
            loss = loss_func(prediction, y)
            
            running_val_loss += loss

            _, predicted_labels = torch.max(prediction, 1)
            batch_correct = (predicted_labels == y).sum()
            num_correct += batch_correct

        epoch_val_loss = running_val_loss / len(v_dataloader)
        epoch_val_acc = num_correct / len(v_dataloader.dataset)

        print(f"Epoch {epoch}: Training Loss: {running_training_loss / len(t_dataloader)}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {epoch_val_acc}")

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            print("new best loss")
        
        if running_loss_thresh - epoch_val_loss > min_val_loss_improvement:
            print("Improved by threshold so resetting epoch count")
            running_loss_thresh = epoch_val_loss
            loss_thresh_epoch = epoch
        else:
            print(f"current epoch count {epoch - loss_thresh_epoch}")
        if epoch - loss_thresh_epoch >= improvement_buffer:
            print(f"Early stopping as no significant improvement in {improvement_buffer} epochs")
            break
            
        
    # torch.save(best_weights, f"model_loss_{best_loss}.pth")

    return best_weights, best_loss



def hyper_parameter_search(model, train_dataloader, validation_dataloader):

    extra_layer_nodes = [0, 7, 16, 32, 64, 128]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    best_model = {"best_loss":1e10}

    for ex_layer_nodes in extra_layer_nodes:
        for learn_rate in learning_rates:
            print("*" * 50)
            print(f"Training with {ex_layer_nodes} nodes with {learn_rate} learn rate")

            model_copy = copy.deepcopy(model)
            model_copy = add_new_layers(model_copy, ex_layer_nodes)

            w, l = train_model(model_copy, train_dataloader, validation_dataloader, learn_rate, min_val_loss_improvement=0.03)

            if (l < best_model["best_loss"]):
                best_model["best_loss"] = l
                best_model["best_weights"] = w
                best_model["layer_nodes"] = ex_layer_nodes
                best_model["learn_rate"] = learn_rate

    with open(f"model-{best_model['best_loss']:.4f}.pkl", "wb") as f: ## remember to load with "rb" flag
        pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)



def resnet():
    model = get_resnet_model()
    freeze_layers(model)
    

    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders()

    hyper_parameter_search(model, train_dataloader, validation_dataloader)

    # train_model(model, train_dataloader, validation_dataloader, 20)

################################################################################################################################################################

class SVM():       
    def __init__(self, training_data):
        self.train = pd.DataFrame.from_records(training_data)

        print("initialising")

        self.X_train = np.array(self.train.drop([0], axis=1))
        self.y_train = np.array(self.train[0])

        #pass all images through dlib


        #pass all images through hog

        
        print("data retrieved", self.X_train.shape, self.y_train.shape)

        self.SVM = SVC(kernel='linear')
        print("beginning fit")
        self.SVM.fit(self.X_train, self.y_train)
        print("completed sucessfully")
        return SVM

    def get_HOG_representation(self, image):
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        return hog_image

    def run_SVM(self, testing_data):
        self.test = pd.DataFrame.from_records(testing_data)
        self.X_test = [x[0] for x in self.test[1]]
        self.y_test = self.test[0]
        y_pred = SVM.predict(self.X_test)
        print(confusion_matrix(self.y_test,y_pred))
        print(classification_report(self.y_test,y_pred))

   

if __name__ == "__main__":
    # robustness_exploration()
    resnet()
    #training_images = load_images('train/')
    #result = get_all_image_landmarks(training_images)
    # print("done")

#Implement SVM
