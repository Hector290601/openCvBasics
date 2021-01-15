import cv2
import time

def savePhoto(frame, nameString = "foto"):
    name = nameString + str(time.time()) + ".jpeg"
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
