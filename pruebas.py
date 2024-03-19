import numpy as np
import cv2
import matplotlib.pyplot as plt

# Supongamos que tienes una matriz tridimensional RGB de una imagen de 10x10
rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Ejemplo de una matriz RGB aleatoria
print ("rgb image:",rgb_image)

# Convertir la imagen RGB a escala de grises
grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
print ("grayscale image",grayscale_image)


# Mostrar la matriz en escala de grises utilizando OpenCV
cv2.imshow('Grayscale Image', grayscale_image)
cv2.imshow('rgb Image', rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#**************************************************************************
# import numpy as np
# import cv2

# # Generar una matriz aleatoria de 10x10 píxeles con valores entre 0 y 255
# random_matrix = np.random.randint(0, 256, (10, 10), dtype=np.uint8)

# # Mostrar la matriz en escala de grises utilizando OpenCV
# cv2.imshow('Grayscale Image', random_matrix)
# cv2.waitKey(0)
# cv2.destroyAllWindows()