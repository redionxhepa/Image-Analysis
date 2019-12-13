import numpy as np
from PIL import Image
import os
import IPython.display
import torch
from resnet import resnet50
import math

model = resnet50()

class Data():
    
    
    def __init__(self, data_path="data", model = resnet50(), verbose=False):
        self.model = model
        self.train_data_path = data_path + "/train"
        self.test_data_path = data_path + "/test"
        self.num_classes = -1
        self.clsname2clsidx = {
            "n02504458": 0,
            "n02481823": 1,
            "n02422699": 2,
            "n02410509": 3,
            "n02391049": 4,
            "n02317335": 5,
            "n02129604": 6,
            "n02123159": 7,
            "n02099601": 8,
            "n01615121": 9
        }
        self.verbose = verbose
    
    def get_test_images(self):
        i = 0
        dt = self.test_data_path + "/images"
        X_test = []
        y_test = np.zeros([100, 5], dtype=np.uint32)
        y_hat_test_localization = np.zeros([100, 4], dtype=np.uint32)
        
        entries = [tuple(line.split(", ")) for line in open(self.test_data_path + "/bounding_box.txt", "r").readlines()]
        idx = 0
        for clsname, startX, startY, endX, endY in entries:
                        
            clsidx = self.clsname2clsidx[clsname]
            
            imgname = str(idx) + ".JPEG"
            image_path = dt + "/" + imgname
            image = Image.open(image_path).convert('RGB') # load an image
            image_original = np.asarray(image) # convert to a numpy array
            #print(imgname)
            #print(image_original.shape)
            y = [int(clsidx), int(startX), int(startY),int(endX), int(endY)]
            X_test.append(image_original)
            y_test[idx] = y
            
            idx += 1
            
        X_test_with_box = np.zeros((5000,2048))
        entries = [tuple(line.split(",")) for line in open(self.test_data_path + "/predicted_bounding_box.txt", "r").readlines()]
        idx = 0
        for image_name, startX,startY, endX, endY in entries:
            index=int(image_name[:-5])
            #print(index)
            image = X_test[index]
            #print(image.shape)
            extracted_box = image[int(startX): int(endX), int(startY):int(endY),:]     
            if(extracted_box.shape[0]==0):
                print('Problem')
                print(image_name)
                print(extracted_box.shape)
            if(extracted_box.shape[1]==0):
                print(image_name)

                print('Problem')
                print(extracted_box.shape)

            feature = self.get_feature_vector(extracted_box)
            X_test_with_box[idx] = feature
            
            if idx % 50 == 0:
                y_hat_test_localization[int(idx/50)] = [int(startX), int(startY), int(endX), int(endY)]
            
            idx += 1
            
        return np.asarray(X_test_with_box), y_test, y_hat_test_localization

    def get_feature_vector(self, image_original):
        h = image_original.shape[0]
        w = image_original.shape[1]

        if h > w:
            pw1 = int((h - w)/2)
            pw2 = h - w - pw1
            pad_width = [(0, 0), (pw1, pw2), (0, 0)]
        else:
            pw1 = int((w - h)/2)
            pw2 = w - h - pw1
            pad_width = [(pw1, pw2), (0, 0), (0, 0)]

        
        image_padded = np.asarray(np.pad(image_original, pad_width, 'constant'))

        #Resizing
        image_resized = np.asarray(Image.fromarray(image_padded).resize((224, 224)))

        image = np.array(image_resized, dtype=np.float32)

        # Normalization
        image /= 255
        image[:, :, 0] -= 0.485
        image[:, :, 1] -= 0.456
        image[:, :, 2] -= 0.406
        image[:, :, 0] /= 0.229
        image[:, :, 1] /= 0.224
        image[:, :, 2] /= 0.225

        # we append an augmented dimension to indicate batch_size, which is one
        image = np.reshape(image, [1, 224, 224, 3])

        # model takes as input images of size [batch_size, 3, im_height, im_width]
        image = np.transpose(image, [0, 3, 1, 2])

        # convert the Numpy image to torch.FloatTensor
        image = torch.from_numpy(image)

        # extract features
        feature_vector = model(image)
        feature_vector = feature_vector.detach().numpy()
        
        return feature_vector


    def get_train_features(self):

        i = 0
        dt = self.train_data_path
        X_train = []
        y_train = []
        self.num_classes = len(os.listdir(dt))
        classes = enumerate(os.listdir(dt))

        for clsname in self.clsname2clsidx.keys():
            clsidx = self.clsname2clsidx[clsname]
            for imgname in os.listdir(dt + "/" + clsname):
                if ".JPEG" in imgname:
                    image_path = dt + "/" + clsname + "/" + imgname
                    image = Image.open(image_path).convert('RGB') # load an image
                    image_original = np.asarray(image) # convert to a numpy array

                    feature_vector = self.get_feature_vector(image_original)
                    X_train.append(feature_vector)
                    y_train.append(clsidx)


                    if i < 5 and self.verbose:
                        print(image_original.shape, image_padded.shape, image_resized.shape, image.dtype, feature_vector.shape)
                        IPython.display.display(Image.fromarray(image_original))
                        IPython.display.display(Image.fromarray(image_padded))
                        IPython.display.display(Image.fromarray(image_resized))
                        i += 1

        X_train = np.array(X_train)
        y_train = np.eye(self.num_classes)[y_train]

        return X_train, y_train, self.num_classes
    
    def get_features(self):
        X_train, y_train, num_classes = self.get_train_features()
       # X_test, y_test = self.get_test_features()
        
        #return X_train, y_train, X_test, y_test
        return X_train, y_train
   
    
if __name__ == "__main__":
    data = Data()
    data.get_train_features()