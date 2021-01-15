import cv2
import time
from datetime import datetime

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

def savePhoto(frame, nameString = "foto"):
    now = datetime.now()
    dateName = now.strftime("%d/%m/%Y %H:%M:%S")
    name = nameString + "at" + now.strftime("%d-%m-%Y-%H-%M-%S") + ".jpeg"
    frame = cv2.putText(frame, dateName, org, font, fontScale, color, thickness, cv2.LINE_AA)
    print(name)
    cv2.imwrite(name, frame)

def main():
    cap = cv2.VideoCapture(0)
    name = "dani"
    delay = 1
    init = time.time()
    start = init
    while cap.isOpened():
        delta = time.time() - start
        ret, frame = cap.read()
        if(delta >= delay):
            savePhoto(frame, name)
            start = time.time()
            print("Foto tomada")
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)
        if k == ord('s'):
            break;
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
