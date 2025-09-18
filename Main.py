import cv2

image_file = 'ExampleImage/example.jpg'
# Load the image
image = cv2.imread(image_file)

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert image to grayscale for detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Blur each detected face
for (x, y, w, h) in faces:
    face_region = image[y:y+h, x:x+w]
    face_blurred = cv2.GaussianBlur(face_region, (51, 51), 30)  # Adjust kernel for blur
    image[y:y+h, x:x+w] = face_blurred

# Save and display
cv2.imwrite('BlurredExamples/blurred_faces.jpg', image)
cv2.imshow('Blurred Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
