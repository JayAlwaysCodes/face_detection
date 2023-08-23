import cv2
from random import randrange

#Load some pretained data on face frontals from open cv
trained_face_data = cv2.CascadeClassifier('face_trained_data.xml')
smiles = cv2.CascadeClassifier('smile.xml')
#choose an image to detect face in
img = cv2.imread('test_immg_3.jpg')

#to capture via webcam
webcam = cv2.VideoCapture(0)

#Iterate over each frame for ever
while True:
    #read the current frame
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read: break

#convert to grayscale for optimization
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame, scaleFactor=1.1, minNeighbors=10)
    
#print(face_coordinates)

#draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0 ), 2)
        the_smile = frame[y:y+h, x:x+w]
        smile_grayscale = cv2.cvtColor(the_smile, cv2.COLOR_BGR2GRAY)
        smiles_coordinates = smiles.detectMultiScale(smile_grayscale, scaleFactor=1.7, minNeighbors=35)
        if len(smiles_coordinates) >0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0,0,0))


#show image
    cv2.imshow('Face detector', frame)

    key= cv2.waitKey(1)
    if key==81 or key==113: #when ever q or Q is pressed
        break
webcam.release()
print('Code Completed!!')