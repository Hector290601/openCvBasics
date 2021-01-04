import cv2

frontalFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
profileFace = cv2.CascadeClassifier("haarcascade_profileface.xml")

def showFaces(faces, frame):
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(frame, (x1, y1), (x1+x2, y1+y2), (255, 0, 0), 2)

def findingFaces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesFrontal = frontalFace.detectMultiScale(gray, 1.3, 5)
    facesProfile = profileFace.detectMultiScale(gray, 1.3, 5)
    if facesFrontal is not None:
        showFaces(facesFrontal, frame)
    if facesProfile is not None:
        showFaces(facesProfile, frame)

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        findingFaces(frame)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == ord('s'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
