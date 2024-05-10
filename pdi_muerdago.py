import os
import tkinter as tk
from tkinter import filedialog
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
#import math
import torch



# Definir la variable global para el archivo seleccionado
archivo_seleccionado = None

#Seleccionar archivo con cuadro de dialogo
def seleccionar_archivo():
    
    global archivo_seleccionado
    # Abre el cuadro de diálogo para seleccionar un archivo
    archivo_seleccionado = filedialog.askopenfilename()

    # Verificar si se seleccionó un archivo
    if archivo_seleccionado:
        label_archivo.config(text="Archivo seleccionado: " + archivo_seleccionado)
        imagen = Image.open(archivo_seleccionado)
        imagen = imagen.resize((int(imagen.width*30/100),int(imagen.height*30/100)))  # Redimensionar la imagen si es necesario
        imagen_tk = ImageTk.PhotoImage(imagen)
        
        label_imagen.config(image=imagen_tk)
        label_imagen.image = imagen_tk  # Mantener una referencia para evitar que la imagen sea eliminada por el recolector de basura

    else:
        label_archivo.config(text="No se seleccionó ningún archivo.")


def obtener_modelo_color():
    global archivo_seleccionado
    ruta_archivo = archivo_seleccionado
    try:
        # Intenta cargar la imagen
        imagen = cv.imread(ruta_archivo)

        # Verifica si la imagen se cargó correctamente
        if imagen is not None:
            # Convierte la imagen a HSV
            imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)

            # Verifica las dimensiones de la imagen HSV
            altura, ancho, canales = imagen_hsv.shape

            # Si la imagen tiene 3 canales, está en RGB; si tiene 3, está en HSV
            if canales == 3:
                print("La imagen está en el modelo de color RGB.")
            elif canales == 4:
                print("La imagen está en el modelo de color HSV.")
        else:
            print("No se pudo cargar la imagen.")
    except Exception as e:
        print("Error:", e)

# Llamada a la función con la ruta de tu archivo de imagen
#obtener_modelo_color()

def convertirBGRtoRGB():
    global archivo_seleccionado
    ruta_archivo = archivo_seleccionado
    # Cargar la imagen en formato BGR
    imagen_BGR = cv.imread(ruta_archivo)

    # Convertir la imagen de BGR a RGB
    imagen_RGB = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2RGB)

    # Mostrar ambas imágenes con matplotlib
    plt.figure(figsize=(10, 5))

    # Mostrar la imagen original en formato BGR
    plt.subplot(1, 2, 1)
    plt.imshow(imagen_BGR)
    plt.title('Imagen BGR')

    # Mostrar la imagen convertida en formato RGB
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_RGB)
    plt.title('Imagen RGB')

    # Ajustar los márgenes y mostrar el gráfico
    plt.tight_layout()
    plt.show()


#Separacion de canales
def separacioncanales():
    global archivo_seleccionado
    ruta_archivo = archivo_seleccionado
    imagen_BGR = cv.imread(ruta_archivo)

    #imgRGB = convertirBGRtoRGB()
    imagen_RGB = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2RGB)
    B, G, R = cv.split(imagen_RGB)  #ojo: esta funcion separa BGR

    # Convertir la imagen original a escala de grises
    imagen_BGR_gris = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2GRAY)
    
    fig, axs = plt.subplots(2,4, figsize = (15,15))
    axs[0,0].imshow(B)
    axs[0,0].set_title("Blue")
    
    axs[0,1].imshow(G)
    axs[0,1].set_title("Green")
    
    axs[0,2].imshow(R)
    axs[0,2].set_title("Red")
    
    axs[0,3].imshow(imagen_RGB)
    axs[0,3].set_title("Imagen RGB")
    
    axs[1,0].imshow(B,cmap = "gray")
    axs[1,0].set_title("Blue Gris")
    
    axs[1,1].imshow(G,cmap = "gray")
    axs[1,1].set_title("Green Gris")
    
    axs[1,2].imshow(R, cmap = "gray")
    axs[1,2].set_title("Red Gris")
    
    axs[1,3].imshow(imagen_BGR_gris, cmap = "gray")
    axs[1,3].set_title("Imagen BGR Gris")

    # Ajustar los márgenes y mostrar el gráfico
    plt.tight_layout()
    plt.show()

#Mostrar imagen en Ventanas con imshow

def mostrarImagen(title, img):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


#Binarización
def binarizacion():
    global archivo_seleccionado
    ruta_archivo = archivo_seleccionado
    imagen_BGR = cv.imread(ruta_archivo)
    
    imagen_RGB = cv.cvtColor(imagen_BGR,cv.COLOR_BGR2RGB)
    #imagen_GRAY1 = cv.cvtColor(imagen_BGR,cv.COLOR_BGR2GRAY)
    imagen_GRAY2 = cv.cvtColor(imagen_RGB,cv.COLOR_RGB2GRAY)


    # Threshold (Umbralizar/Binarizar la imagen)
    #ret1, thres1 = cv.threshold(imagen_GRAY2,120,255,cv.THRESH_BINARY)
    #ret2, thres2 = cv.threshold(imagen_GRAY2,120,255,cv.THRESH_BINARY_INV)

    # Apply adaptive thresholding with ADAPTIVE_THRESH_MEAN_C uses the mean of the neighborhood area
    adaptive_mean_a = cv.adaptiveThreshold(imagen_GRAY2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # Apply adaptive thresholding with ADAPTIVE_THRESH_GAUSSIAN_C uses the mean of the neighborhood area
    adaptive_gaussian_a = cv.adaptiveThreshold(imagen_GRAY2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    
    
    # Apply adaptive thresholding with ADAPTIVE_THRESH_MEAN_C uses the mean of the neighborhood area
    adaptive_mean_b = cv.adaptiveThreshold(imagen_GRAY2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 6)
    # Apply adaptive thresholding with ADAPTIVE_THRESH_GAUSSIAN_C uses the mean of the neighborhood area
    adaptive_gaussian_b = cv.adaptiveThreshold(imagen_GRAY2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 6)

    
    fig, axs = plt.subplots(2,3,figsize = (10,10))
    
    axs[0,0].imshow(imagen_GRAY2,cmap="gray")
    axs[0,0].set_title("Escala de grises")
    axs[0,1].imshow(adaptive_mean_a, cmap="gray")
    axs[0,1].set_title("Adaptative Mean")
    axs[0,2].imshow(adaptive_gaussian_a, cmap = "gray")
    axs[0,2].set_title("Adaptative Gaussian")  
    
    axs[1,0].imshow(imagen_GRAY2,cmap="gray")
    axs[1,0].set_title("Escala de grises")
    axs[1,1].imshow(adaptive_mean_b, cmap="gray")
    axs[1,1].set_title("Adaptative Mean")
    axs[1,2].imshow(adaptive_gaussian_b, cmap = "gray")
    axs[1,2].set_title("Adaptative Gaussian")  
    
    
    
    # Ajustar los márgenes y mostrar el gráfico
    plt.tight_layout()
    plt.show()
   
    based_image_name = os.path.basename(ruta_archivo)
    bin_image_file = '/Users/lizette/MCC_coding/Muerdago/img/binarias/b_' + based_image_name
    
    #Paso 8: Guardar la imagen
    cv.imwrite(bin_image_file, adaptive_gaussian_b)
    print(f'Imagen corregida guardada en: {bin_image_file}')

#Corrección de imagen
def correcciones():
    global archivo_seleccionado
    ruta_archivo = archivo_seleccionado
    
    # Paso 1: Carga de la imagen
    image = cv.imread(ruta_archivo)

    # Paso 2: Corrección de exposición y contraste
    #exposure_corrected = cv.convertScaleAbs(image, alpha=1.2, beta=10)
    exposure_corrected = cv.convertScaleAbs(image, alpha=1.8, beta=0)


    # Paso 3: Reducción de ruido
    noise_reduced = cv.fastNlMeansDenoisingColored(exposure_corrected, None, 10, 10, 7, 21)


    # Paso 3.1: umbralización
    # Aplicar thresholding
    # Apply adaptive thresholding with ADAPTIVE_THRESH_GAUSSIAN_C uses the mean of the neighborhood area
    
    imagen_RGB = cv.cvtColor(noise_reduced,cv.COLOR_BGR2RGB)
    #imagen_GRAY1 = cv.cvtColor(imagen_BGR,cv.COLOR_BGR2GRAY)
    imagen_GRAY2 = cv.cvtColor(imagen_RGB,cv.COLOR_RGB2GRAY)
    binary_image2 = cv.adaptiveThreshold(imagen_GRAY2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 2)

    binary_image = cv.adaptiveThreshold(binary_image2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 103, 2)



    #Paso 4: Ajuste de nitidez
    sharpen_kernel = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])
    
    
    
    # sharpen_kernel = np.array([[-1, 0, 1],
    #                             [0, 0, -0],
    #                             [1, 0, -1]])
    
    #filtro para deteccion de bordes
    
    # sharpen_kernel = np.array([[-1, 0, 1],
    #                             [-2, 0, 2],
    #                             [-1, 0, 1]])
    
    
    
    #sharpened = cv.filter2D(noise_reduced, -1, sharpen_kernel)
    sharpened = cv.filter2D(binary_image, -1, sharpen_kernel)

    imagen_BGR = cv.cvtColor(sharpened,cv.COLOR_GRAY2BGR)

    # Paso 5: Corrección de color (ajuste de balance de blanco)
    color_corrected = cv.cvtColor(imagen_BGR, cv.COLOR_BGR2LAB)
    
    l_channel, a_channel, b_channel = cv.split(color_corrected)
    
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    color_corrected = cv.merge((l_channel, a_channel, b_channel))
    color_corrected = cv.cvtColor(color_corrected, cv.COLOR_LAB2BGR)
    gray_color_corrected = cv.cvtColor(color_corrected,cv.COLOR_BGR2GRAY)

    adaptive_gaussian_b = cv.adaptiveThreshold(gray_color_corrected, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 111, 3)
    
    fig, axs = plt.subplots(1,2,figsize = (10,10))
    
    axs[0].imshow(image, cmap = "gray")
    axs[0].set_title("Imagen original")
    axs[1].imshow(adaptive_gaussian_b,cmap = "gray")
    axs[1].set_title("Imagen corregida")

    
    # Ajustar los márgenes y mostrar el gráfico
    plt.tight_layout()
    plt.show()

    based_image_name = os.path.basename(ruta_archivo)
    corrected_image_file = '/Users/lizette/MCC_coding/Muerdago/img/corregidas/c_' + based_image_name
    
    #Paso 8: Guardar la imagen
    cv.imwrite(corrected_image_file, color_corrected)  
    print(f'Imagen corregida guardada en: {corrected_image_file}')

# Paso 6: Redimensión y recorte (opcional)
# resized = cv2.resize(color_corrected, (new_width, new_height))
# cropped = color_corrected[y1:y2, x1:x2]

# Paso 7: Aplicación de filtros creativos (opcional)
# Aplicar filtros creativos como efectos de viñeta, etc.

# Paso 8: Guardar la imagen
# cv.imwrite('/Users/lizette/MCC_coding/ProyectoM/img/tmp/TimeShot017-0013_corregida.jpg', color_corrected)  

def sobel():
    global archivo_seleccionado
    ruta_archivo = archivo_seleccionado
    
    # Paso 1: Carga de la imagen
    image = cv.imread(ruta_archivo)
    
    # Convertir la imagen a escala de grises
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Aplicar el filtro de suavizado gaussiano
    blurred_image_gaussian = cv.GaussianBlur(gray_image, (7, 7), 0)

    # Aplicar el filtro de suavizado bilateral
    blurred_image_bilateral = cv.bilateralFilter(gray_image, 9, 80, 80)

    # Aplicar el filtro de Sobel en las direcciones X e Y
    sobel_x = cv.Sobel(blurred_image_bilateral, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(blurred_image_bilateral, cv.CV_64F, 0, 1, ksize=3)

    fig, axs = plt.subplots(1,4,figsize = (10,10))
    
    axs[0].imshow(gray_image, cmap = "gray")
    axs[0].set_title("Imagen original")
    axs[1].imshow(sobel_x,cmap = "gray")
    axs[1].set_title("Filtro de Sobel (X)")
    axs[2].imshow(sobel_y,cmap = "gray")
    axs[2].set_title("Filtro de Sobel (Y)")
    axs[3].imshow(blurred_image_gaussian,cmap = "gray")
    axs[3].set_title("Filtro Gaussiano")
    

    
    # Ajustar los márgenes y mostrar el gráfico
    plt.tight_layout()
    plt.show()

    based_image_name = os.path.basename(ruta_archivo)
    sobel_image_file = '/Users/lizette/MCC_coding/Muerdago/img/sobel/s_' + based_image_name
    
    #Paso 8: Guardar la imagen
    cv.imwrite(sobel_image_file, sobel_x)
    print(f'Imagen corregida guardada en: {sobel_image_file}')

def contornos():
    global archivo_seleccionado
    ruta_archivo = archivo_seleccionado
    
    # Carga de la imagen
    image = cv.imread(ruta_archivo)
    
    # Convertir la imagen a escala de grises
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Aplicar un suavizado para reducir el ruido
    #image_blurred = cv.GaussianBlur(gray_image, (5, 5), 0)

    # Aplicar la umbralización para segmentar las células
    #_, thresholded_image = cv.threshold(image_blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    _, thresholded_image = cv.threshold(gray_image, 120, 255, cv.THRESH_BINARY)
    
    # Encontrar contornos en la imagen umbralizada
    contours, _ = cv.findContours(thresholded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Definir el área mínima de los contornos a considerar (ajusta este valor según sea necesario)
    #min_contour_area = 1

    # Filtrar los contornos por área mínima
    #filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]

    merged_contours = np.zeros_like(gray_image)
    
    
    # Dibujar los contornos encontrados en la imagen original
    #image_with_contours = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for contour in contours:
        cv.drawContours(merged_contours, [contour], -1, (255, 255, 255), 2)

    # Crear una figura y ejes usando Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar la imagen original en el primer eje
    axs[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    axs[0].set_title('Imagen Original')

    # Mostrar la imagen con contornos en el segundo eje
    axs[1].imshow(cv.cvtColor(merged_contours, cv.COLOR_BGR2RGB))
    axs[1].set_title('Imagen con Contornos')

    # Mostrar la figura
    plt.show()
   
def canny():
    global archivo_seleccionado
    ruta_archivo = archivo_seleccionado
    
    # Carga de la imagen
    image = cv.imread(ruta_archivo)
    
    # Convertir la imagen a escala de grises
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    
    # Aplicar la detección de bordes con Canny
    edges = cv.Canny(gray_image, 50, 150, apertureSize=3)

    # Aplicar la Transformada de Hough para detectar líneas
    lines = cv.HoughLines(edges, 1, np.pi/180, 150)

    # Dibujar las líneas detectadas sobre la imagen original
    image_with_lines = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
    
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Crear una figura y ejes usando Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar la imagen original en el primer eje
    axs[0].imshow(cv.cvtColor(gray_image, cv.COLOR_BGR2RGB))
    axs[0].set_title('Imagen Original')

    # Mostrar la imagen con contornos en el segundo eje
    axs[1].imshow(cv.cvtColor(image_with_lines, cv.COLOR_BGR2RGB))
    axs[1].set_title('Imagen con Contornos')

    # Mostrar la figura
    plt.show()
    
    
def GaussLaplace():
    global archivo_seleccionado
    ruta_archivo = archivo_seleccionado
    
    # Carga de la imagen
    image = cv.imread(ruta_archivo)
    
    # Convertir la imagen a escala de grises
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    
    # Aplicar un filtro Gaussiano para suavizar la imagen
    image_blurred = cv.GaussianBlur(gray_image, (3, 3), 0)

    # Aplicar el operador Laplaciano
    laplacian = cv.Laplacian(image_blurred, cv.CV_64F)

    # Convertir el resultado a un rango de 0-255 (uint8)
    laplacian = np.uint8(np.absolute(laplacian))

    # Mostrar los bordes en la imagen original
    image_with_edges = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
    
    image_with_edges[laplacian > 50] = [0, 0, 255]  # Pintar los bordes de rojo



    # Crear una figura y ejes usando Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar la imagen original en el primer eje
    axs[0].imshow(cv.cvtColor(image_blurred, cv.COLOR_BGR2RGB))
    axs[0].set_title('Imagen Gaussian')

    # Mostrar la imagen con contornos en el segundo eje
    axs[1].imshow(cv.cvtColor(image_with_edges, cv.COLOR_BGR2RGB))
    axs[1].set_title('Imagen Laplace')

    # Mostrar la figura
    plt.show()
    

def erosiondilatacion():
    global archivo_seleccionado
    ruta_archivo = archivo_seleccionado
    
    # Carga de la imagen
    image = cv.imread(ruta_archivo)
    
    # Convertir la imagen a escala de grises
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Definir el kernel para las operaciones morfológicas
    kernel = np.ones((5,5), np.uint8)

    # Aplicar erosión para suavizar los bordes
    erosion = cv.erode(gray_image, kernel, iterations=1)

    # Aplicar dilatación para resaltar los bordes
    dilation = cv.dilate(gray_image, kernel, iterations=1)


    # Crear una figura y ejes usando Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar la imagen original en el primer eje
    axs[0].imshow(cv.cvtColor(erosion, cv.COLOR_BGR2RGB))
    axs[0].set_title('Erosion')

    # Mostrar la imagen con contornos en el segundo eje
    axs[1].imshow(cv.cvtColor(dilation, cv.COLOR_BGR2RGB))
    axs[1].set_title('Dilatacion')

    # Mostrar la figura
    plt.show()
    

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#Ventana principal
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

root = tk.Tk()
root.title("Proyecto Muérdago - Identificación de Estomas")
root.state("zoomed")

frame01= tk.Frame() #define la instancia del frame01
frame01.pack() #empaqueta el frame para que este contenido en la raiz
frame01.config(width="1050",height="650") #da un tamaño al frame
frame01.config(bg="LightSkyBlue1") #pone un color de back ground
frame01.config(bd=15) #da un ancho de borde
frame01.config(relief="ridge") #pone un relieve
frame01.place(x=20,y=50)
frame01.grid_propagate(False) #evita que el frame sea esclavo del tamaño de sus componenetes cuando usamos un grid

label01= tk.Label(frame01,text="Procesamiento Preliminar de la imagen")
label01.config(font=("Arial",28))
label01.config(bg="LightSkyBlue1") 
label01.config(fg="white")
label01.grid(row=0,column=0,sticky="w",pady=10)

buttonOpc0 = tk.Button(frame01,text="Selecciona el Archivo", command=seleccionar_archivo)
buttonOpc0.grid(row=1,column=0,pady=8,sticky="e")
buttonOpc0.config(cursor="hand2")
buttonOpc0.config(bg="bisque")
buttonOpc0.config(font=("Arial",12))

label_archivo = tk.Label(frame01,text="")
label_archivo.config(font=("Arial",12))
label_archivo.config(bg="LightSkyBlue1") 
label_archivo.config(fg="Black")
label_archivo.grid(row=1,column=1,sticky="w",pady=10)

# Crear un widget Label para mostrar la imagen en el marco
label_imagen = tk.Label(frame01)
label_imagen.grid(row=0,column=1)

buttonOpc2 = tk.Button(frame01,text="Conversion BGR to RGB", command=convertirBGRtoRGB)
buttonOpc2.grid(row=2,column=0,pady=8,sticky="e")
buttonOpc2.config(cursor="hand2")
buttonOpc2.config(bg="bisque")
buttonOpc2.config(font=("Arial",12))

buttonOpc3 = tk.Button (frame01,text="Separación de Canales", command=separacioncanales)
buttonOpc3.grid(row=3,column=0,pady=8,sticky="e")
buttonOpc3.config(cursor="hand2")
buttonOpc3.config(bg="bisque")
buttonOpc3.config(font=("Arial",12))

buttonOpc4 = tk.Button (frame01,text="Binarización", command=binarizacion)
buttonOpc4.grid(row=4,column=0,pady=8,sticky="e")
buttonOpc4.config(cursor="hand2")
buttonOpc4.config(bg="bisque")
buttonOpc4.config(font=("Arial",12))

buttonOpc5 = tk.Button(frame01,text="Correcciones", command=correcciones)
buttonOpc5.grid(row=5,column=0,pady=8,sticky="e")
buttonOpc5.config(cursor="hand2")
buttonOpc5.config(bg="bisque")
buttonOpc5.config(font=("Arial",12))

buttonOpc6 = tk.Button (frame01,text="Filtro Sobel", command=sobel)
buttonOpc6.grid(row=6,column=0,pady=8,sticky="e")
buttonOpc6.config(cursor="hand2")
buttonOpc6.config(bg="bisque")
buttonOpc6.config(font=("Arial",12))

buttonOpc7 = tk.Button (frame01,text="Contornos", command=contornos)
buttonOpc7.grid(row=7,column=0,pady=8,sticky="e")
buttonOpc7.config(cursor="hand2")
buttonOpc7.config(bg="bisque")
buttonOpc7.config(font=("Arial",12))

buttonOpc8 = tk.Button (frame01,text="Contornos Canny", command=canny)
buttonOpc8.grid(row=8,column=0,pady=8,sticky="e")
buttonOpc8.config(cursor="hand2")
buttonOpc8.config(bg="bisque")
buttonOpc8.config(font=("Arial",12))

buttonOpc9 = tk.Button (frame01,text="Gauss and Laplace", command=GaussLaplace)
buttonOpc9.grid(row=9,column=0,pady=8,sticky="e")
buttonOpc9.config(cursor="hand2")
buttonOpc9.config(bg="bisque")
buttonOpc9.config(font=("Arial",12))


buttonOpc8 = tk.Button (frame01,text="Erosion / Dilatacion", command=erosiondilatacion)
buttonOpc8.grid(row=10,column=0,pady=8,sticky="e")
buttonOpc8.config(cursor="hand2")
buttonOpc8.config(bg="bisque")
buttonOpc8.config(font=("Arial",12))

root.mainloop() #ciclo principal para que aparezca el root o pantalla principal