import cv2
import dlib

img = cv2.imread('/home/buddha/Desktop/cara3.jpg')
detector = dlib.get_frontal_face_detector()
dets = detector(img, 1)
face = dets[0]
cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
cv2.imwrite('out.jpg', img)