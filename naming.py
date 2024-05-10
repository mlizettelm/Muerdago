import cv2 as cv
import os

# Directorio que contiene las imágenes originales
original_path =  r"/Users/lizette/Documents/MCC/Proyecto muerdago/Plugable Digital Viewer/Timed Shots/001-Casuarina"
dest_path = r"/Users/lizette/Documents/MCC/Proyecto muerdago/dataset/casuarina"
total_images = 0 

# Iterar sobre las imágenes en el directorio
for idx, filename in enumerate(os.listdir(original_path)):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Filtrar solo archivos de imagen
        img_path = os.path.join(original_path, filename)
        
        # Leer la imagen en formato RGB
        img = cv.imread(img_path)
        
        # Crear un nombre nuevo para cada imagen
        new_filename = 'casuarina_' + str(idx+1) + '.jpg'
        
        # Guardar la imagen en otro directorio con el nuevo nombre
        cv.imwrite(os.path.join(dest_path, new_filename), img)
        total_images+=1
        
print(f"Naming completado para las {total_images} imágenes en formato RGB.")