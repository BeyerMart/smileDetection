import cv2
import os
import glob
import itertools
#from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time 

#time start
ts = time.time()
#import sklearn
cascPathface = os.path.dirname(cv2.__file__) + "\data\haarcascade_frontalface_alt2.xml" #Windows path
cascPathsmile = os.path.dirname(cv2.__file__) + "\data\haarcascade_smile.xml" #Windows path
#cascPathsmile = os.path.dirname(cv2.__file__) + "/data/haarcascade_smile.xml" #unix path
#cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml" #unix path
print(cascPathface)

faceCascade = cv2.CascadeClassifier(cascPathface)
smileCascade = cv2.CascadeClassifier(cascPathsmile)

class SmileDetector:
    def __init__(self):
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0 
        self.positivesImages = self.getFilenames("positives")
        self.negativesImages = self.getFilenames("negatives")
        self.y_true = []
        self.y_pred = []
    

    # TP - img in positive folder was detected as smile
    # FP - img in negative folder was detected as smile
    # TN - img in negative folder was detected as not smile
    # FN - img in positive folder was detected as not smile
    
    def getFilenames(self, type):
        fnames = []
        #for files in glob.glob('dataset/' + type + '/*.jpg', recursive=True): # unix path
        for files in glob.glob('dataset\\' + type + '\\*.jpg', recursive=True): # windows path
            fnames.append(files)
        return fnames
    

    def detectSmiles(self, frame, type):
        frame = cv2.imread(frame)

    
        smiles = smileCascade.detectMultiScale(frame)
        if type == 'positive':
            self.y_true.append(1)
            #True Positives
            if len(smiles) > 0:
                self.TP += 1
                self.y_pred.append(1)

            #False Negative
            else:
                self.FN += 1
                self.y_pred.append(0)

                
        elif type == 'negative':
            self.y_true.append(0)
            #False Positive
            if len(smiles) > 0:
                self.FP += 1
                self.y_pred.append(1)
            #True Negative
            else:
                self.TN += 1
                self.y_pred.append(0)

        

    def classifyAllImages(self):
        for frame in self.positivesImages:
            self.detectSmiles(frame, 'positive')
        for frame in self.negativesImages:
            self.detectSmiles(frame, 'negative')

    def printConfusionMatrix(self):
        print("Amount of detected smiles: ", self.TP + self.FP) 
        print("Amount of images: ", len(self.positivesImages) + len(self.negativesImages))

        print("True Positives: ", self.TP)
        print("False Positives: ", self.FP)
        print("True Negatives: ", self.TN)
        print("False Negatives: ", self.FN)

        #print(self.y_true)
        disp = ConfusionMatrixDisplay.from_predictions(self.y_true, self.y_pred, normalize='all')
        disp.plot()
        plt.show()
    
smileDetector = SmileDetector()
smileDetector.classifyAllImages()
smileDetector.printConfusionMatrix()
print('Execution time:', time.time()-ts)