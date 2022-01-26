import cv2
import os
cascPathface = os.path.dirname(cv2.__file__) + "\data\haarcascade_frontalface_alt2.xml" #Windows path
cascPathsmile = os.path.dirname(cv2.__file__) + "\data\haarcascade_smile.xml" #Windows path
print(cascPathface)
faceCascade = cv2.CascadeClassifier(cascPathface)
smileCascade = cv2.CascadeClassifier(cascPathsmile)

#do for one image only
frames = []
#frame = cv2.imread("G:\\OneDrive\\OneDrive - uibk.ac.at\\uibk\\5th\\VC\\PS\\smiledetectionbemame\\dataset\\SMILEsmileD\\SMILEs\\positives\\positives7\\3.jpg", cv2.COLOR_BGR2GRAY)
#frame = cv2.imread("dataset\\SMILEsmileD\\SMILEs\\positives\\positives7\\15.jpg")
happyFrame = cv2.imread("dataset\\SMILEsmileD\\SMILEs\\negatives\\negatives7\\7.jpg")
sadFrame = cv2.imread("dataset\\SMILEsmileD\\SMILEs\\negatives\\negatives7\\9.jpg")
frames.append(happyFrame)
frames.append(sadFrame)
print(frames)

def detectSmile(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #faces = faceCascade.detectMultiScale(frame_gray)
    smiles = smileCascade.detectMultiScale(frame_gray)#print(faces)
    if len(smiles) > 0: #or smiles.any()
        print(smiles.size)
        for (x2,y2,w2,h2) in smiles:
            smile_center = ( x2 + w2//2, y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, smile_center, radius, (255, 0, 0 ), 4)
    return frame

finishedImages = []
for frame in frames:
    img = detectSmile(frame)
    #cv2.imshow('Capture - Smile detection', img)
    finishedImages.append(img)
im_v = cv2.vconcat([finishedImages[0], finishedImages[1]])
cv2.imshow('Capture - Smile detection', im_v)


k = cv2.waitKey(0)
