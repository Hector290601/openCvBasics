import cv2
import os

dataPath = '/home/hector/openCvBasics/deteccionDeRostros/savingsFaces/faces'
imagePaths = os.listdir(dataPath)
print('imagePaths ' + str(imagePaths))

faceRecognizerEFR = cv2.face.EigenFaceRecognizer_create()
faceRecognizerFFR = cv2.face.FisherFaceRecognizer_create()
faceRecognizerLBPHF = cv2.face.LBPHFaceRecognizer_create()

faceRecognizerEFR.read('modeloEFR.xml')
#faceRecognizerFFR.read('modeloFFR.xml')
faceRecognizerLBPHF.read('modeloLBPHF.xml')

cap = cv2.VideoCapture(0)

frontalFaceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profileFaceClassif = cv2.CascadeClassifier('haarcascade_profileface.xml')

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    facesFrontal = frontalFaceClassif.detectMultiScale(gray, 1.3, 5)
    facesProfile = profileFaceClassif.detectMultiScale(gray, 1.3, 5)
    for (x, y, h, w) in facesFrontal:
        cara = auxFrame[y:y+h, x:x+w]
        cara = cv2.resize(cara, (200, 200), interpolation=cv2.INTER_CUBIC)
        #resultEFR = faceRecognizerEFR.predict(cara)
        resultLBPHF = faceRecognizerLBPHF.predict(cara)
        #cv2.putText(frame, '{}'.format(resultEFR), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, '{}'.format(resultLBPHF), (x, y-15), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        """
        if resultEFR[1] < 5700:
            cv2.putText(frame, '{}'.format(imagePaths[resultEFR[0]]), (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'UNKNOW', (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
        """
        if resultLBPHF[1] < 70:
            cv2.putText(frame, '{}'.format(imagePaths[resultLBPHF[0]]), (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'UNKNOW', (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
    """
    for (x, y, h, w) in facesProfile:
        cara = auxFrame[y:y+h, x:x+w]
        cara = cv2.resize(cara, (200, 200), interpolation=cv2.INTER_CUBIC)
        resultEFR = faceRecognizerEFR.predict(cara)
        resultLBPHF = faceRecognizerLBPHF.predict(cara)
        cv2.putText(frame, '{}'.format(resultEFR), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, '{}'.format(resultLBPHF), (x, y-15), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        if resultEFR[1] < 5700:
            cv2.putText(frame, '{}'.format(imagePaths[resultEFR[0]]), (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'UNKNOW', (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
        if resultLBPHF[1] < 70:
            cv2.putText(frame, '{}'.format(imagePaths[resultLBPHF[0]]), (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'UNKNOW', (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
    """
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()

