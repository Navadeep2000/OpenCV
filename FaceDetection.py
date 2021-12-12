import cv2

faceCascade = cv2.CascadeClassifier("Test_Images/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == 'q':
        break
