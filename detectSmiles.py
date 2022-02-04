import cv2
import os
import glob
import itertools
#from sklearn import metrics
#import sklearn
#cascPathface = os.path.dirname(cv2.__file__) + "\data\haarcascade_frontalface_alt2.xml" #Windows path
#cascPathsmile = os.path.dirname(cv2.__file__) + "\data\haarcascade_smile.xml" #Windows path
cascPathsmile = os.path.dirname(cv2.__file__) + "/data/haarcascade_smile.xml" #unix path
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml" #unix path
print(cascPathface)

faceCascade = cv2.CascadeClassifier(cascPathface)
smileCascade = cv2.CascadeClassifier(cascPathsmile)

class SmileDetector:
    def __init__(self):
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0 
        self.positivesImages = self.getFilenames("positives")
        self.negativesImages = self.getFilenames("negatives")
    

    # TP - img in positive folder was detected as smile
    # FP - img in negative folder was detected as smile
    # TN - img in negative folder was detected as not smile
    # FN - img in positive folder was detected as not smile
    
    def getFilenames(self, type):
        fnames = []
        for files in glob.glob('dataset/SMILEsmileD/SMILEs/' + type + '/*.jpg', recursive=True): # unix path
        #for files in glob.glob('dataset\\SMILEsmileD\\SMILEs\\' + type + '\\*.jpg', recursive=True): # windows path
            fnames.append(files)
        return fnames
    

    def detectSmiles(self, frame, type):
        frame = cv2.imread(frame)
    
        smiles = smileCascade.detectMultiScale(frame)
        if type == 'positive':
            #True Positives
            if len(smiles) > 0:
                self.TP += 1
            #False Negative
            else:
                self.FN += 1
                
        elif type == 'negative':
            #False Positive
            if len(smiles) > 0:
                self.FP += 1
            #True Negative
            else:
                self.TN += 1

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
    
smileDetector = SmileDetector()
smileDetector.classifyAllImages()
smileDetector.printConfusionMatrix()
