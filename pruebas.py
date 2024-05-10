# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# # Supongamos que tienes una matriz tridimensional RGB de una imagen de 10x10
# rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Ejemplo de una matriz RGB aleatoria
# print ("rgb image:",rgb_image)

# # Convertir la imagen RGB a escala de grises
# grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
# print ("grayscale image",grayscale_image)


# # Mostrar la matriz en escala de grises utilizando OpenCV
# cv2.imshow('Grayscale Image', grayscale_image)
# cv2.imshow('rgb Image', rgb_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#**************************************************************************
# import numpy as np
# import cv2

# # Generar una matriz aleatoria de 10x10 píxeles con valores entre 0 y 255
# random_matrix = np.random.randint(0, 256, (10, 10), dtype=np.uint8)

# # Mostrar la matriz en escala de grises utilizando OpenCV
# cv2.imshow('Grayscale Image', random_matrix)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # codigo para transformar las mascaras en formato que yolo puede interpretar
# import os
# import cv2

# input_dir = './Img/tmp/masks'
# output_dir = './Img/tmp/labels'

# # Ejecutar la rutina sobre todos los archivos del directorio masks
# for j in os.listdir(input_dir):
#     image_path = os.path.join(input_dir, j)
    
#     # load the binary mask and get its contours
#     mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

#     H, W = mask.shape
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # convert the contours to polygons
#     polygons = []
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 200:
#             polygon = []
#             for point in cnt:
#                 x, y = point[0]
#                 polygon.append(x / W)
#                 polygon.append(y / H)
#             polygons.append(polygon)

#     # print the polygons
#     with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
#         for polygon in polygons:
#             for p_, p in enumerate(polygon):
#                 if p_ == len(polygon) - 1:
#                     f.write('{}\n'.format(p))
#                 elif p_ == 0:
#                     f.write('0 {} '.format(p))
#                 else:
#                     f.write('{} '.format(p))

#         f.close()


#aplicar filtro con la transformada de fourier 2D

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path =  r"/Users/lizette/MCC_coding/Muerdago/Img/binarias/b_SingleShot0000.jpg"

# Cargar la imagen
img = cv2.imread(img_path, 0)  # Cargar la imagen en escala de grises

# Calcular la transformada de Fourier 2D
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Calcular el espectro de amplitud
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Crear un filtro de paso bajo
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# Aplicar el filtro
fshift = fshift * mask
f_ishift = np.fft.ifftshift(fshift)
img_filtered = np.fft.ifft2(f_ishift)
img_filtered = np.abs(img_filtered)

# Mostrar la imagen original y la imagen filtrada
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Imagen original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_filtered, cmap='gray')
plt.title('Imagen filtrada'), plt.xticks([]), plt.yticks([])
plt.show()
