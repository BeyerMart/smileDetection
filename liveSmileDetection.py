import cv2
import os
cascPathface = os.path.dirname(
    cv2.__file__) + "\data\haarcascade_frontalface_alt2.xml" #Windows path
cascPathsmile = os.path.dirname(
    cv2.__file__) + "\data\haarcascade_smile.xml" #Windows path
print(cascPathface)
faceCascade = cv2.CascadeClassifier(cascPathface)
smileCascade = cv2.CascadeClassifier(cascPathsmile)

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
        faceROI = frame[y:y+h,x:x+w]
        grayFace = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
        smiles = smileCascade.detectMultiScale(grayFace)

        for idx, (x2, y2, w2, h2) in enumerate(smiles):
            cv2.rectangle(frame, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2),(255- (idx * 50),0,0), 2)
            #smile_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            #radius = int(round((w2 + h2) * 0.25))
            #frame = cv2.circle(frame, smile_center, radius, (255, 0, 0), 4)q

        # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('./screenshot.png', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()