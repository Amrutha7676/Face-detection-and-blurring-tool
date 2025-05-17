# Face-detection-and-blurring-tool
import cv2

def blur_faces(image_path, output_path):
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Blur each detected face
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        image[y:y+h, x:x+w] = blurred_face

    # Save and show result
    cv2.imwrite(output_path, image)
    cv2.imshow("Blurred Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

blur_faces("myphoto.png.png", "output.jpg")



