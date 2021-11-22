import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from tqdm import tqdm
from os import listdir

import pickle
import copy
import math
import csv

from Globals import labels

#convert labels to numerical categories
label_map = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'neutral':4, 'sad':5, 'surprise':6}
# move all data and model to GPU if available
my_device = "cuda" if torch.cuda.is_available() else "cpu"

# set the requires grad parameter on model to false for all layers
def freeze_layers(model):
    for layer in model.parameters():
        layer.requires_grad = False

# modify the final layer or add an extra hidden layer before that of a given size
def add_new_layers(model, extra_nodes=0):
    if extra_nodes != 0:
        model.fc = torch.nn.Linear(512, extra_nodes) # must take 512 inputs
        model = nn.Sequential(
            model,
            torch.nn.Linear(extra_nodes, len(labels))
        )
    else:
        model.fc = torch.nn.Linear(512, len(labels)) # must take 512 inputs from previous layer

    return model

# download (or access cahced) model, will get pretrained versino if unspecified
def get_resnet_model(pretrained=True):
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=pretrained) ## is a 1000 output net as it was trained as 1000 image classifier
    return model

# load all the images into a single list
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

# custom dataset class to work by loading all images into memory to be used, only works as images are small so can be easily held in memory
class CustomDataset(Dataset):
    def __init__(self, dir, transform=None):
        images, labels = flat_load_images(dir) # load all images
        self.transform = transform
        self.img_labels = labels
        self.my_images = images

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.my_images[idx]) # apply given transform if it exists
        else:
            image = self.my_images[idx]

        label = self.img_labels[idx]

        return image, label_map[label] 
        
# get the validation and training dataloaders that will have split training dataset by a given ratio
def get_train_dataloaders(validation_ratio=0.2):

    preprocess = T.Compose([
        T.Lambda(lambda x : x.convert("RGB")),
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(p=0.5), # randomly flip
        T.RandomVerticalFlip(p=0.5), # randomly flip
        T.RandomApply(transforms=[T.RandomRotation(degrees=(0,360))], p=0.5), # randomly apply random rotation
        T.RandomPerspective(distortion_scale=0.5, p=0.5), # randomly apply a random perspective
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalise tensor for resnet as specified
    ])

    train_dataset = CustomDataset("train/", transform=preprocess)
    #split train and validation dataset randomly
    train_dataset, validation_dataset = random_split(train_dataset, [math.ceil(len(train_dataset) * (1-validation_ratio)), math.floor(len(train_dataset) * validation_ratio)])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False) # no need to shuffle validation data

    return train_dataloader, validation_dataloader

# train the model
def train_model(model, t_dataloader, v_dataloader, learn_rate=1e-3, min_val_loss_improvement=0.1, improvement_buffer=5, absolute_max_epochs=25):
    model.train() # set model to training mode
    model.to(my_device) # send model to device

    learning_layers = list() # get the layers that are not to be "frozen"
    for name, layer in model.named_parameters():
        if layer.requires_grad:
            learning_layers.append(layer)
            # print("Training layer: ", name)
    
    # specify loss and optimiser functions
    loss_func = nn.CrossEntropyLoss()
    optimizer_func = torch.optim.SGD(learning_layers, learn_rate)

    best_loss = 1e10 # set initial loss
    best_weights = copy.deepcopy(model.state_dict()) # get initial best weights which are whatever they have been initialised to

    loss_thresh_epoch = 0
    running_loss_thresh = 1e10

    for epoch in range(absolute_max_epochs):
        model.train() # set model to train mode
        running_training_loss = 0
        for (X, y) in tqdm(t_dataloader): # for each batch in the dataloader
            X, y = X.to(my_device), y.to(my_device)

            # make preditiction and calculate loss
            prediction = model(X)
            loss = loss_func(prediction, y)
            running_training_loss += loss

            # update model weights based off of prediction
            optimizer_func.zero_grad()
            loss.backward()
            optimizer_func.step()

        model.eval() # set model to eval mode
        num_correct = 0
        running_val_loss = 0
        for (X, y) in v_dataloader: # for each batch in validation dataset
            X, y = X.to(my_device), y.to(my_device)

            prediction = model(X)
            loss = loss_func(prediction, y)
            
            running_val_loss += loss

            _, predicted_labels = torch.max(prediction, 1)
            batch_correct = (predicted_labels == y).sum()
            num_correct += batch_correct

        # calculate metrics
        epoch_val_loss = running_val_loss / len(v_dataloader)
        epoch_val_acc = num_correct / len(v_dataloader.dataset)

        print(f"Epoch {epoch}: Training Loss: {running_training_loss / len(t_dataloader)}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {epoch_val_acc}")

        # if there is a new best loss, i.e. best model
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            print("new best loss")

        # if the loss has improved by the minimum threshold
        if running_loss_thresh - epoch_val_loss > min_val_loss_improvement:
            print("Improved by threshold so resetting epoch count")
            running_loss_thresh = epoch_val_loss
            loss_thresh_epoch = epoch
        else:
            print(f"current epoch count {epoch - loss_thresh_epoch}")

        # if the loss has not improved by the buffer amount in the given number of epochs, early stop training
        if epoch - loss_thresh_epoch >= improvement_buffer:
            print(f"Early stopping as no significant improvement in {improvement_buffer} epochs")
            break
            
        
    # torch.save(best_weights, f"model_loss_{best_loss}.pth")

    return best_weights, best_loss # return weights and loss


# perform hyperparameter grid search
def hyper_parameter_search(model, train_dataloader, validation_dataloader):

    extra_layer_nodes = [0, 7, 16, 32, 64, 128] # search different model shapes
    learning_rates = [0.0001, 0.001, 0.01, 0.1] # search different learning rates

    best_model = {"best_loss":1e10}

    for ex_layer_nodes in extra_layer_nodes:
        for learn_rate in learning_rates:
            print("*" * 50)
            print(f"Training with {ex_layer_nodes} nodes with {learn_rate} learn rate")

            model_copy = copy.deepcopy(model)
            model_copy = add_new_layers(model_copy, ex_layer_nodes)

            w, l = train_model(model_copy, train_dataloader, validation_dataloader, learn_rate, min_val_loss_improvement=0.03)

            if (l < best_model["best_loss"]): # if new best model save its details to dictionary
                best_model["best_loss"] = l
                best_model["best_weights"] = w
                best_model["layer_nodes"] = ex_layer_nodes
                best_model["learn_rate"] = learn_rate

    with open(f"model-{best_model['best_loss']:.4f}.pkl", "wb") as f: ## remember to load with "rb" flag
        pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)

# get the model and train the last few layers of the resnet model
def resnet_train():
    model = get_resnet_model()
    freeze_layers(model)

    train_dataloader, validation_dataloader = get_train_dataloaders()

    hyper_parameter_search(model, train_dataloader, validation_dataloader)


# get the dataloader for the test dataset
def get_test_dataloader(path):

    preprocess = T.Compose([ # shouldn't apply any data augmentation transforms, only to get in right shape for tensor
        T.Lambda(lambda x : x.convert("RGB")),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = CustomDataset(path, preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return test_dataloader


# iterate over test dataset
def resnet_test(path):
    with open(path, "rb") as f:
        model_dict = pickle.load(f)

    model = get_resnet_model(pretrained=False) # get blank resnet model
    model = add_new_layers(model, model_dict["layer_nodes"]) # make model into correct shape

    model.load_state_dict(model_dict["best_weights"]) # apply best model weights

    model.eval()

    model.to(my_device)

    for perturb_type in tqdm(listdir("results/")):

        p_amounts = []
        accs = []
        for p_amount in tqdm(listdir("results/" + perturb_type + "/")):
            
            t_dataloader = get_test_dataloader(f"results/{perturb_type}/{p_amount}/")
            num_correct = 0

            for (X, y) in t_dataloader:
                X, y = X.to(my_device), y.to(my_device)

                # make prediction and calcualte accuracy
                predictions = model(X)
                _, predicted_labels = torch.max(predictions, 1)
                batch_correct = (predicted_labels == y).sum()
                num_correct += batch_correct

            accuracy = num_correct / len(t_dataloader.dataset)
            p_amounts.append(p_amount)
            accs.append(accuracy.item())
        
        # write calculated accuracy to csv file
        with open(f"Resnet-{perturb_type}.csv", 'w', newline="") as res_f:
            writer = csv.writer(res_f)
            writer.writerow(p_amounts)
            writer.writerow(accs)