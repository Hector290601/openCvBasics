import cv2
import time

def savePhoto(frame, nameString = "foto"):
    name = nameString + str(time.time()) + ".jpeg"
    cv2.imwrite(name, frame)

def main():
    cap = cv2.VideoCapture(0)
    name = input("Ingrese el nombre para sus imágenes: ")
    delay = int(input("Ingrese el delay entre imágenes:"))
    des = int(input("Desea mostrar la imagen? (si escoge que no, se mostrará hasta antes de la primera captura de imagen , después desaparecerá)\n1)Sí\n2)No\n"))
    num = 0
    if(des == 2):
        num = int(input("Cuántas imágenes desa?"))
    i = 1
    desAp = 1
    init = time.time()
    start = init
    while cap.isOpened():
        delta = time.time() - start
        ret, frame = cap.read()
        if(delta >= 2):
            savePhoto(frame, name)
            start = time.time()
            print("Foto tomada")
            if(des == 2):
                desAp = 0
                i += 1
                cv2.destroyAllWindows()
        if(desAp == 1):
            cv2.imshow("frame", frame)
        k = cv2.waitKey(1)
        if k == ord('s') or i - 1 == num:
            break;
    cap.release()
    if(desAp == 1):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
