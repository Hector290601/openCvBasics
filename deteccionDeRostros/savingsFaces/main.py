import cv2
import time
import os

i = 510

frontalFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
profileFace = cv2.CascadeClassifier("haarcascade_profileface.xml")

def savingFaces(frame, x, y, w, h, auxFrame, key):
    global i
    face = auxFrame[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('findedFace{}'.format(i), face)
    cv2.imwrite('faces/face_{}.jpg'.format(i), face)
    print("faces/face_{}.jpg".format(i))
    cv2.destroyWindow('findedFace{}'.format(i))
    i += 1

def showFaces(faces, frame, key):
    auxFrame = frame.copy()
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(frame, (x1, y1), (x1+x2, y1+y2), (255, 0, 0), 2)
        savingFaces(frame, x1, y1, x2, y2, auxFrame, key)

def findingFaces(frame, key):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesFrontal = frontalFace.detectMultiScale(gray, 1.3, 5)
    facesProfile = profileFace.detectMultiScale(gray, 1.3, 5)
    if facesFrontal is not None:
        showFaces(facesFrontal, frame, key)
    if facesProfile is not None:
        showFaces(facesProfile, frame, key)

def main():
    if not os.path.exists('faces'):
        os.makedirs('faces')
        print("faces folder created")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        findingFaces(frame, k)
        cv2.imshow('frame', frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
