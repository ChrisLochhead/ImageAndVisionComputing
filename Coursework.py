
#SKlearn
from skimage.feature import hog
from skimage import exposure
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#Numpy
import numpy as np
#Matplotlib
from matplotlib import image
#Processing
import dlib

from ImagePerturbGen import Image_Manipulator # these must be below above definitions
from Resnet import resnet_train, resnet_test

from Globals import *

################################################################################################################################################
#DLIB STUFF
def get_all_image_landmarks(imgs, predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")):
    #Initialise detector, iterator and storage for new images
    detector = dlib.get_frontal_face_detector()
    image_index = 0
    processed_images = []
    empty = np.zeros(136)
    #Cycle through each image
    for i in imgs:
        #Get all of the detection points, checking for edge cases and skipping the process if so
        dets = detector(i, 2)
        if (len(dets) != 1):
            processed_images.append(empty)
            image_index += 1
            continue
        shape = predictor(i, dets[0])
        if len(shape.parts()) != 68:
            processed_images.append(empty)
            image_index += 1
            continue

        #For detected faces, extract the values of the points
        array_parts = []

        for landmark in range(68):
            array_parts.append(shape.part(landmark).x)
            array_parts.append(shape.part(landmark).y)
 
        #Append locational data
        image_index += 1
        processed_images.append(np.array(array_parts))
    return np.array(processed_images)

##################################################################################################################################################

class SVM():       
    def __init__(self, training_data, training_labels):

        print("initialising SVM")
        self.X_train = training_data
        self.y_train = training_labels

        #pass all images through dlib and then HOG 
        self.dlib_train = get_all_image_landmarks(self.X_train)
        self.HOG_train = self.get_HOG_images(self.X_train)

        index = 0
        self.mod_train = []
        for image in self.HOG_train:
            #print(len(self.HOG_train[index]))
            #print(len(self.dlib_train[index]))
            self.mod_train.append(np.concatenate([self.HOG_train[index], self.dlib_train[index]]))
            index += 1

        self.mod_train = np.array(self.mod_train)
        print("Dlib and HOG modifications complete")
        self.SVM = SVC()
        print("length of image input: ", len(self.mod_train[0]))
        # Now train the model.  The return value is the trained model capable of making predictions.
        print("beginning fit", len(self.mod_train), len(self.y_train))
        self.SVM.fit(self.mod_train, self.y_train)
        print("Fit completed sucessfully")

    #Convert images to HOG representation
    def get_HOG_images(self, images):
        HOG_images = []
        indexes = 0
        #For each image, apply the HOG algorithm, flatten the result and return it
        for image in images:
            fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            indexes += 1
            HOG_images.append(hog_image_rescaled.flatten())
        return np.array(HOG_images)

    def prepare_test_data(self, input):
        input_data, flat_input_data, input_labels = load_images(input)
        dlib_data = get_all_image_landmarks(input_data)
        HOG_data = self.get_HOG_images(input_data)
        index = 0
        output = []
        for image in self.HOG_test:
            output.append(np.concatenate([HOG_data[index], dlib_data[index]]))
            index += 1
    
        return np.array(output), input_labels

    def run_SVM_study(self, testing_data, testing_labels):
        #For each perturbation, run a prediction on all images
        print("Preparing Study perturbations")
        self.dlib_test = get_all_image_landmarks(testing_data)
        self.HOG_test = self.get_HOG_images(testing_data)

        index = 0
        self.mod_test = []
        for image in self.HOG_test:
            self.mod_test.append(np.concatenate([self.HOG_test[index], self.dlib_test[index]]))
            index += 1
    
        self.mod_test = np.array(self.mod_test)
        print("making predictions")
        predictions = self.SVM.predict(self.mod_test)
        print("Prediction accuracy: ", accuracy_score(testing_labels, predictions) * 100)


        ##Entire study 
        for i in brightness_modifiers_down:
            test_data, test_labels = self.prepare_test_data("results/brightness_down/" + i + "/")
            predictions = self.SVM.predict(test_data)
            print("Prediction accuracy- Brightness down permutation: ", i, " ; ", accuracy_score(test_labels, predictions) * 100)

        for i in brightness_modifiers_up:
            test_data, test_labels = self.prepare_test_data("results/brightness_up/" + i + "/")
            predictions = self.SVM.predict(test_data)
            print("Prediction accuracy- Brightness up permutation: ", i, " ; ", accuracy_score(test_labels, predictions) * 100)

        for i in contrast_multipliers_down:
            test_data, test_labels = self.prepare_test_data("results/contrast_down/" + i + "/")
            predictions = self.SVM.predict(test_data)
            print("Prediction accuracy- Contrast down permutation: ", i, " ; ", accuracy_score(test_labels, predictions) * 100)

        for i in contrast_multipliers_up:
            test_data, test_labels = self.prepare_test_data("results/contrast_up/" + i + "/")
            predictions = self.SVM.predict(test_data)
            print("Prediction accuracy- Contrast up permutation: ", i, " ; ", accuracy_score(test_labels, predictions) * 100)

        for i in scales:
            test_data, test_labels = self.prepare_test_data("results/gaussian_noise/" + i + "/")
            predictions = self.SVM.predict(test_data)
            print("Prediction accuracy- Gaussian noise permutation: ", i, " ; ", accuracy_score(test_labels, predictions) * 100)

        for i in blur_sd:
            test_data, test_labels = self.prepare_test_data("results/gaussian_blur/" + i + "/")
            predictions = self.SVM.predict(test_data)
            print("Prediction accuracy- Gaussian blur permutation: ", i, " ; ", accuracy_score(test_labels, predictions) * 100)

        for i in occlusion_box_size:
            test_data, test_labels = self.prepare_test_data("results/occlusion/" + i + "/")
            predictions = self.SVM.predict(test_data)
            print("Prediction accuracy- Occlusion permutation: ", i, " ; ", accuracy_score(test_labels, predictions) * 100)

        for i in sp_intensity:
            test_data, test_labels = self.prepare_test_data("results/salt_and_pepper/" + i + "/")
            predictions = self.SVM.predict(test_data)
            print("Prediction accuracy Salt and Pepper permutation: ", i, " ; ", accuracy_score(test_labels, predictions) * 100)
            
        print("Study complete sucessfully.")

def robustness_exploration():
    #Initialise an image manipulator (set to false if images already made on device)
    print("Robustness Exploration called")
    manipulator = Image_Manipulator(False)
    print("Manipulator initialised")
    SVM_Controller = SVM(manipulator.training_images, manipulator.train_labels)
    print("SVM initialised")
    SVM_Controller.run_SVM_study(manipulator.test_images, manipulator.test_labels)
    print("SVM study complete")


if __name__ == "__main__":
    # robustness_exploration()
    # resnet_test("model-1.5820.pkl")
    resnet_test("model-1.5820.pkl")
    print("Program done")
