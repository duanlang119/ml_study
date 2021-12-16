import cv2 as cv
import pytesseract as tess

image = cv.imread("D:/workspace/ziping/3.png")
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
text = tess.image_to_string(image_rgb, lang="chi_sim")
print(text)
h, w, c = image.shape
boxes = tess.image_to_boxes(image)
for b in boxes.splitlines():
    b = b.split(' ')
    image = cv.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv.imshow('text detect', image)
cv.waitKey(0)
cv.destroyAllWindows()