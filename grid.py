#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data import Data
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def change_labels(target_label, labels):
    new_labels = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        if(labels[i]==target_label):
            new_labels[i]=1
    return new_labels

def convert_one_hot_to_normal_label(label):
    new_label = np.zeros(label.shape[0])
    for m in range(label.shape[0]):
        for i in range(label.shape[1]):
            if(label[m][i]!=0):
                new_label[m] = i
    return new_label

def convert_normal_to_one_hot_label(label):
    new_label = np.zeros((label.shape[0],10),dtype=np.uint8)
    for m in range(label.shape[0]):
        new_label[m,label[m]] = 1
    #print(new_label)
    return new_label


kernels = ["sigmoid", "rbf", "linear", "poly", "sigmoid"]
Cs = [0.1, 1, 10, 100]

for kernel in kernels:
    for C in Cs:
        X_train = np.load("X_train.npy")
        y_train = np.load("y_train.npy")
        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy")

        X_train = X_train.squeeze()

        print("--------", kernel, "C ", C, "-----------")
        # In[2]:


        y_train = convert_one_hot_to_normal_label(y_train)


        # In[3]:


        y_train0 = change_labels(0, y_train)
        y_train1 = change_labels(1, y_train)
        y_train2 = change_labels(2, y_train)
        y_train3 = change_labels(3, y_train)
        y_train4 = change_labels(4, y_train)
        y_train5 = change_labels(5, y_train)
        y_train6 = change_labels(6, y_train)
        y_train7 = change_labels(7, y_train)
        y_train8 = change_labels(8, y_train)
        y_train9 = change_labels(9, y_train)


        # In[4]:


        classifier0 = SVC(kernel=kernel, C=C)
        classifier0 = classifier0.fit(X_train, y_train0)
        classifier1 = SVC(kernel=kernel, C=C)
        classifier1 = classifier1.fit(X_train, y_train1)
        classifier2 = SVC(kernel=kernel, C=C)
        classifier2 = classifier2.fit(X_train, y_train2)


        # In[5]:


        classifier3 = SVC(kernel=kernel, C=C)
        classifier3 = classifier3.fit(X_train, y_train3)


        # In[6]:


        classifier4 = SVC(kernel=kernel, C=C)
        classifier4 = classifier4.fit(X_train, y_train4)


        # In[7]:


        classifier5 = SVC(kernel=kernel, C=C)
        classifier5 = classifier5.fit(X_train, y_train5)


        # In[8]:


        classifier6 = SVC(kernel=kernel, C=C)
        classifier6 = classifier6.fit(X_train, y_train6)


        # In[9]:


        classifier7 = SVC(kernel=kernel, C=C)
        classifier7 = classifier7.fit(X_train, y_train7)


        # In[10]:


        classifier8 = SVC(kernel=kernel, C=C)
        classifier8 = classifier8.fit(X_train, y_train8)


        # In[11]:


        classifier9 = SVC(kernel=kernel, C=C)
        classifier9 = classifier9.fit(X_train, y_train9)


        # In[12]:


        scores0 = classifier0.decision_function(X_test)
        scores1 = classifier1.decision_function(X_test)
        scores2 = classifier2.decision_function(X_test)
        scores3 = classifier3.decision_function(X_test)
        scores4 = classifier4.decision_function(X_test)
        scores5 = classifier5.decision_function(X_test)
        scores6 = classifier6.decision_function(X_test)
        scores7 = classifier7.decision_function(X_test)
        scores8 = classifier8.decision_function(X_test)
        scores9 = classifier9.decision_function(X_test)


        # In[13]:


        scores0 = np.reshape(scores0,(100,50,1))
        scores1 = np.reshape(scores1,(100,50,1))
        scores2 = np.reshape(scores2,(100,50,1))
        scores3 = np.reshape(scores3,(100,50,1))
        scores4 = np.reshape(scores4,(100,50,1))
        scores5 = np.reshape(scores5,(100,50,1))
        scores6 = np.reshape(scores6,(100,50,1))
        scores7 = np.reshape(scores7,(100,50,1))
        scores8 = np.reshape(scores8,(100,50,1))
        scores9 = np.reshape(scores9,(100,50,1))


        # In[14]:


        scores = np.concatenate((scores0,scores1,scores2,scores3,scores4,scores5,scores6,scores7,scores8,scores9),axis=2)


        # In[15]:


        real_predictions = np.zeros(100, dtype=np.uint32) 
        scores_for_roc = np.zeros((100,10))
        y_max_score_idx = []
        for i in range(100):
            test_image = scores[i]
            real_scores = np.max(test_image, axis=0)
            max_scores = np.argmax(test_image, axis=0)
            scores_for_roc[i] = real_scores
            real_predictions[i] = np.argmax(real_scores)
            y_max_score_idx.append(max_scores[real_predictions[i]])


        # In[16]:


        y_gt=np.asarray([i[0] for i in y_test])


        # In[17]:


        y_gt


        # In[18]:


        #FOR VISUALIZATION PURPOSES!!!!
        #BELOW ROC CURVE CODE IS DIRECTLY TAKEN and ADAPTED FROM: 
        #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
        #and
        #https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
        def plot_confusion_matrix(cm, classes,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()

        # In[19]:


        from sklearn.metrics import accuracy_score, classification_report



        # In[20]:


        print("Classification Accuracy", accuracy_score(y_gt, real_predictions))


        # In[21]:


        X_test.shape


        # In[22]:


        entries_pred = [tuple(line.split(",")) for line in open("data/test/predicted_bounding_box.txt", "r").readlines()]
        y_hat_test_localization = []
        for i, idx in enumerate(y_max_score_idx):
        #    print(entries_pred[i*50+idx][1:])
            y_hat_test_localization.append(entries_pred[i*50+idx][1:])


        # In[23]:


        correctClasses = np.zeros((10, 1))
        correctClass = 0
        for i in range(100):

            startX, startY, endX, endY = y_test[i, 1:]
            startY_hat, startX_hat, endY_hat, endX_hat = y_hat_test_localization[i]

            startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
            startX_hat, startY_hat, endX_hat, endY_hat = int(startX_hat), int(startY_hat), int(endX_hat), int(endY_hat)
            #print("GT", startX, startY, endX, endY)
            #print("Pred", startX_hat, startY_hat, endX_hat, endY_hat)

            overlapX = max(0, min(endX, endX_hat) - max(startX, startX_hat))
            overlapY = max(0, min(endY, endY_hat) - max(startY, startY_hat))
            overlap = overlapX * overlapY

            area_original = (endX - startX) * (endY - startY)
            area_pred = (endX_hat - startX_hat) * (endY_hat - startY_hat)
            union = area_original + area_pred - overlap
            score = overlap/union
            #print(score)
            #print(overlapX, overlapY)
            #print(overlap)

            if score >= 0.5 and int(y_gt[i]) == int(real_predictions[i]):
                correctClasses[y_gt[i]] += 1
                correctClass += 1

        print("Localization Accuracy", correctClass / 100)