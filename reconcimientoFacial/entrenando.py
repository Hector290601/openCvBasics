import cv2
import os
import numpy as np

path = '/home/hector/openCvBasics/deteccionDeRostros/savingsFaces/faces'
listNames = os.listdir(path)

print('personas:', listNames)

labels = []
facesData = []
label = 0

faceRecognizerEFR = cv2.face.EigenFaceRecognizer_create()
faceRecognizerLBPHF = cv2.face.LBPHFaceRecognizer_create()
faceRecognizerFFR = cv2.face.FisherFaceRecognizer_create()

for name in listNames:
    namePath = path + '/' + name
    print("Leyendo las imagenes de " + namePath)
    for fileName in os.listdir(namePath):
        print(fileName)
        labels.append(label)
        image = cv2.imread(namePath + '/' + fileName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        facesData.append(image)
    label += 1
print("Entrenando Eigenfaces... ")
faceRecognizerEFR.train(facesData, np.array(labels))
#print("Entrenando Fisherfaces... ")
#faceRecognizerFFR.train(facesData, np.array(labels))
print("Entrenando Local Binary Patterns Histograms... ")
faceRecognizerLBPHF.train(facesData, np.array(labels))
print("Guardando los archivos xml")
faceRecognizerEFR.write('modeloEFR.xml')
#faceRecognizerFFR.write('modeloFFR.xml')
faceRecognizerLBPHF.write('modeloLBPHF.xml')
print("Modelos almacenados")

