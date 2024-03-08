import tkinter as tk
from tkinter import filedialog
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Binarización
def binarizacion():
    path3 = r"/Users/lizette/MCC_coding/ProyectoM/img/tmp/TimeShot017-0013.jpg"
    letras = cv.imread(path3,0)

    imgGray = cv.cvtColor(letras,cv.COLOR_BGR2RGB)

    # Threshold (Umbralizar/Binarizar la imagen)
    ret, thres1 = cv.threshold(imgGray,120,255,cv.THRESH_BINARY)
    ret, thres2 = cv.threshold(imgGray,120,255,cv.THRESH_BINARY_INV)

    fig, axs = plt.subplots(1,3,figsize = (10,10))
    
    axs[0].imshow(imgGray)
    axs[0].set_title("Escala de grises")
    axs[1].imshow(thres1)
    axs[1].set_title("Binaria Est.")
    axs[2].imshow(thres2)
    axs[2].set_title("Binaria Inv. Est.")
    
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    #mng.full_screen_toggle()
    
    # Get the screen width and height
    #root = tk.Tk()
    #screen_width = root.winfo_screenwidth()
    #screen_height = root.winfo_screenheight()
    #root.destroy()

    # Set figure size to match screen size
    #mng.resize(screen_width, screen_height)


    plt.show()
   
   

# path3 = r"/Users/lizette/MCC_coding/Muerdago/img/tmp/TimeShot017-0013.jpg"
# # Paso 1: Carga de la imagen
# image = cv.imread(path3)

# # Paso 2: Corrección de exposición y contraste
# exposure_corrected = cv.convertScaleAbs(image, alpha=1.2, beta=10)

# # Paso 3: Reducción de ruido
# noise_reduced = cv.fastNlMeansDenoisingColored(exposure_corrected, None, 10, 10, 7, 21)

# # Paso 4: Ajuste de nitidez
# sharpen_kernel = np.array([[-1, -1, -1],
#                             [-1, 9, -1],
#                             [-1, -1, -1]])
# sharpened = cv.filter2D(noise_reduced, -1, sharpen_kernel)

# # Paso 5: Corrección de color (ajuste de balance de blanco)
# color_corrected = cv.cvtColor(sharpened, cv.COLOR_BGR2LAB)
# l_channel, a_channel, b_channel = cv.split(color_corrected)
# clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# l_channel = clahe.apply(l_channel)
# color_corrected = cv.merge((l_channel, a_channel, b_channel))
# color_corrected = cv.cvtColor(color_corrected, cv.COLOR_LAB2BGR)




# Paso 6: Redimensión y recorte (opcional)
# resized = cv2.resize(color_corrected, (new_width, new_height))
# cropped = color_corrected[y1:y2, x1:x2]

# Paso 7: Aplicación de filtros creativos (opcional)
# Aplicar filtros creativos como efectos de viñeta, etc.

# Paso 8: Guardar la imagen
# cv.imwrite('/Users/lizette/MCC_coding/ProyectoM/img/tmp/TimeShot017-0013_corregida.jpg', color_corrected)  


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#Ventana principal
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



def seleccionar_archivo():
    # Abre el cuadro de diálogo para seleccionar un archivo
    archivo = filedialog.askopenfilename()
    # Muestra la ruta del archivo seleccionado en la etiqueta
    #etiqueta.config(text="Archivo seleccionado: " + archivo)
    print(archivo)





root = tk.Tk()
root.title("Proyecto Muerdago - Identificación Índice Estomático")
root.state("zoomed")

frame01= tk.Frame() #define la instancia del frame01
frame01.pack() #empaqueta el frame para que este contenido en la raiz
frame01.config(width="1050",height="650") #da un tamaño al frame
frame01.config(bg="LightSkyBlue1") #pone un color de back ground
frame01.config(bd=15) #da un ancho de borde
frame01.config(relief="ridge") #pone un relieve
frame01.place(x=20,y=50)
frame01.grid_propagate(False) #evita que el frame sea esclavo del tamñao de sus componenetes cuando usamos un grid


label01= tk.Label(frame01,text="Opciones de Procesamiento")
label01.config(font=("Arial",28))
label01.config(bg="LightSkyBlue1") 
label01.config(fg="white")
label01.grid(row=0,column=0,sticky="w",pady=10)


buttonOpc0 = tk.Button(frame01,text="Seleccionar Archivo", command=seleccionar_archivo)
buttonOpc0.grid(row=0,column=1,pady=8,sticky="e")
buttonOpc0.config(cursor="hand2")
buttonOpc0.config(bg="bisque")
buttonOpc0.config(font=("Arial",12))
buttonOpc0.config(cursor="hand2")

# Etiqueta para mostrar la ruta del archivo seleccionado
etiqueta = tk.Label(frame01, text="")
# etiqueta.pack(pady=5)
etiqueta.grid(row=0,column=2,sticky="w",pady=10)

# buttonOpc1 = Button(frame01,text="Convoluciones", command=convoluciones)
# buttonOpc1.grid(row=1,column=0,pady=8,sticky="e")
# buttonOpc1.config(cursor="hand2")
# buttonOpc1.config(bg="bisque")
# buttonOpc1.config(font=("Arial",12))
# buttonOpc1.config(cursor="hand2")

# buttonOpc2 = Button (frame01,text="Separación de Canales", command=canales)
# buttonOpc2.grid(row=2,column=0,pady=8,sticky="e")
# buttonOpc2.config(cursor="hand2")
# buttonOpc2.config(bg="bisque")
# buttonOpc2.config(font=("Arial",12))
# buttonOpc2.config(cursor="hand2")

buttonOpc3 = tk.Button (frame01,text="Binarización", command=binarizacion)
buttonOpc3.grid(row=3,column=0,pady=8,sticky="e")
buttonOpc3.config(cursor="hand2")
buttonOpc3.config(bg="bisque")
buttonOpc3.config(font=("Arial",12))
buttonOpc3.config(cursor="hand2")

# buttonOpc4 = Button (frame01,text="Operaciones lógicas", command=operaciones)
# buttonOpc4.grid(row=4,column=0,pady=8,sticky="e")
# buttonOpc4.config(cursor="hand2")
# buttonOpc4.config(bg="bisque")
# buttonOpc4.config(font=("Arial",12))
# buttonOpc4.config(cursor="hand2")

# buttonOpc5 = Button (frame01,text="Filtros", command=filtros)
# buttonOpc5.grid(row=5,column=0,pady=8,sticky="e")
# buttonOpc5.config(cursor="hand2")
# buttonOpc5.config(bg="bisque")
# buttonOpc5.config(font=("Arial",12))
# buttonOpc5.config(cursor="hand2")

# buttonOpc6 = Button (frame01,text="Operaciones Morfologicas", command=morfologicas)
# buttonOpc6.grid(row=6,column=0,pady=8,sticky="e")
# buttonOpc6.config(cursor="hand2")
# buttonOpc6.config(bg="bisque")
# buttonOpc6.config(font=("Arial",12))
# buttonOpc6.config(cursor="hand2")


# buttonOpc7 = Button (frame01,text="Segmentación K-means", command=segmentacion)
# buttonOpc7.grid(row=7,column=0,pady=8,sticky="e")
# buttonOpc7.config(cursor="hand2")
# buttonOpc7.config(bg="bisque")
# buttonOpc7.config(font=("Arial",12))
# buttonOpc7.config(cursor="hand2")

# buttonOpc8 = Button (frame01,text="Algoritmo Felzenszwalb", command=algoritmoFelzen)
# buttonOpc8.grid(row=8,column=0,pady=8,sticky="e")
# buttonOpc8.config(cursor="hand2")
# buttonOpc8.config(bg="bisque")
# buttonOpc8.config(font=("Arial",12))
# buttonOpc8.config(cursor="hand2")

root.mainloop() #ciclo principal para que aparezca el root o pantalla principal