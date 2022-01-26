import cv2
import os
cascPathface = os.path.dirname(cv2.__file__) + "\data\haarcascade_frontalface_alt2.xml" #Windows path
cascPathsmile = os.path.dirname(cv2.__file__) + "\data\haarcascade_smile.xml" #Windows path
print(cascPathface)
faceCascade = cv2.CascadeClassifier(cascPathface)
smileCascade = cv2.CascadeClassifier(cascPathsmile)

#do for one image only
import os
print("pwd=" + os.getcwd())

#frame = cv2.imread("G:\\OneDrive\\OneDrive - uibk.ac.at\\uibk\\5th\\VC\\PS\\smiledetectionbemame\\dataset\\SMILEsmileD\\SMILEs\\positives\\positives7\\3.jpg", cv2.COLOR_BGR2GRAY)
#frame = cv2.imread("dataset\\SMILEsmileD\\SMILEs\\positives\\positives7\\15.jpg")
#frame = cv2.imread("dataset\\SMILEsmileD\\SMILEs\\negatives\\negatives7\\5.jpg")
frame = cv2.imread("1.jpg")

frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame_gray = cv2.equalizeHist(frame_gray)

faces = faceCascade.detectMultiScale(frame_gray)
#print(faces)
#smiles = smileCascade.detectMultiScale(frame_gray,
#                                         scaleFactor=1.1,
#                                         minNeighbors=5,
#                                         minSize=(60, 60),
#                                         flags=cv2.CASCADE_SCALE_IMAGE)
#print(smiles)
for (x,y,w,h) in faces:
    print("found face" + " x=" + str(x) + " y=" +  str(y) + " w=" + str(w) +  " h=" + str(h))
    center = (x + w//2, y + h//2)
    frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    faceROI = frame_gray[y:y+h,x:x+w]
    #-- In each face, detect smile
    smiles = smileCascade.detectMultiScale(faceROI)
    #print(smiles)
    #print(smiles.shape)
    for (x2,y2,w2,h2) in smiles:
        smile_center = (x + x2 + w2//2, y + y2 + h2//2)
        radius = int(round((w2 + h2)*0.25))
        frame = cv2.circle(frame, smile_center, radius, (255, 0, 0 ), 4)


#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


cv2.imshow('Capture - Smile detection', frame)

k = cv2.waitKey(0)
