# TRABAJO DE TÍTULO - SANTIAGO GUZMÁN
# CREACIÓN DE PROMEDIOS DE LA SEÑAL EMG BASADA EN CWT

# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
import glob
import os
import heapq
from statistics import mean
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import itertools
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import wilcoxon
#Recorrido de la fila completa por columna
def obtener_columna(matriz, indice_columna):
	return [fila[indice_columna] if indice_columna < len(fila) else None for fila in matriz]

#Valores de cada columna para su posterior Normalización
def mat2gray(vector):
        V = vector - np.min(vector)
        V = V / np.max(vector)
        return V
    
folder_path_paciente = r'C:/Users/santi/Desktop/Tesis/MYO Thalmic bracelet/Código de Comparaciones/NPZ CWT EMG de 20 segundos sin Normalización/NPZ Completo'
#NPZ Paciente 01
file_list_paciente01 = glob.glob(folder_path_paciente + "/Canal*Paciente01.npz")
#NPZ Paciente 02
file_list_paciente02 = glob.glob(folder_path_paciente + "/Canal*Paciente02.npz")
#NPZ Paciente 03
file_list_paciente03 = glob.glob(folder_path_paciente + "/Canal*Paciente03.npz")
#NPZ Paciente 04
file_list_paciente04 = glob.glob(folder_path_paciente + "/Canal*Paciente04.npz")
#NPZ Paciente 05
file_list_paciente05 = glob.glob(folder_path_paciente + "/Canal*Paciente05.npz")
#NPZ Paciente 06
file_list_paciente06 = glob.glob(folder_path_paciente + "/Canal*Paciente06.npz")
#NPZ Paciente 07
file_list_paciente07 = glob.glob(folder_path_paciente + "/Canal*Paciente07.npz")
#NPZ Paciente 08
file_list_paciente08 = glob.glob(folder_path_paciente + "/Canal*Paciente08.npz")
#NPZ Paciente 09
file_list_paciente09 = glob.glob(folder_path_paciente + "/Canal*Paciente09.npz")
#NPZ Paciente 10
file_list_paciente10 = glob.glob(folder_path_paciente + "/Canal*Paciente10.npz")

file = [file_list_paciente01, file_list_paciente02, file_list_paciente03, file_list_paciente04, file_list_paciente05, 
        file_list_paciente06, file_list_paciente07, file_list_paciente08, file_list_paciente09, file_list_paciente10]

voltajespaciente = []
for i in range(len(file)):  
    for j in range(len(file[i])):  
        main_dataframe = np.load(file[i][j])
        emg_funcional = main_dataframe['a']
        voltajespaciente.append(emg_funcional)


voltajesP03 = []
for j in range(len(file[0])):  
    main_dataframe = np.load(file[2][j])
    emg_funcional = main_dataframe['a']
    voltajesP03.append(emg_funcional)
    
    
voltajesP05 = []
for j in range(len(file[0])):  
    main_dataframe = np.load(file[4][j])
    emg_funcional = main_dataframe['a']
    voltajesP05.append(emg_funcional)
    
voltajesP06 = []
for j in range(len(file[0])):  
    main_dataframe = np.load(file[5][j])
    emg_funcional = main_dataframe['a']
    voltajesP06.append(emg_funcional)
    
voltajesP07 = []
for j in range(len(file[0])):  
    main_dataframe = np.load(file[6][j])
    emg_funcional = main_dataframe['a']
    voltajesP07.append(emg_funcional)
    
voltajesP08 = []
for j in range(len(file[0])):  
    main_dataframe = np.load(file[7][j])
    emg_funcional = main_dataframe['a']
    voltajesP08.append(emg_funcional)
    
voltajesP09 = []
for j in range(len(file[0])):  
    main_dataframe = np.load(file[8][j])
    emg_funcional = main_dataframe['a']
    voltajesP09.append(emg_funcional)

emgpacientes03 = [voltajesP03[1], voltajesP03[4], voltajesP03[5], voltajesP03[7]]

emg_funcional = np.array(emgpacientes03[0])
#Lectura archivo 
samplerate = 1e3 #Frecuencia de muestreo
length = emg_funcional.shape[0] / samplerate #Duración total Electromiografía en segundos
#print("Duración Total = {length}s")

#Parametros CWT 
num = 4000 #Número de sampleo generado
wavelet = 'cmor1.0-1.5' # Bugs Fixed https://pywavelets.readthedocs.io/en/latest/release.1.1.0.html 
maxesc = np.log2(1500)
widths = np.logspace(0,maxesc, num=num, base=2,dtype='int')
#print("Escalas Usadas: ", widths)
dt = 1/samplerate # diferencia de intervalos de tiempo
frequencies = pywt.scale2frequency(wavelet, widths) / dt # Obtención de frecuencias correspondientes a escalas Modificar aquí precisi´´on
#print("Freqs asociadas a Escalas: ", frequencies)
# Calculo transformación continua de ondículas de los anchos de matriz numpy
wavelet_coeffs, freqs = pywt.cwt(emg_funcional, widths, wavelet=wavelet, sampling_period=samplerate, method='fft')
#print("Tamaño de wavelet transform: ", wavelet_coeffs.shape)
#print("Freqs asociadas a wavelet transform: ", freqs)

#Valores Absolutos de la CWT
abscwt = abs(wavelet_coeffs) # Devolución de ABS de los números complejos es la magnitud
gesto_1 = abscwt[:,763:2242]
gesto_2 = abscwt[:,4045:5628]
gesto_3 = abscwt[:,9070:11027]
gesto_4 = abscwt[:,14568:16474]
gesto_5 = abscwt[:,19795:22041]
gesto_6 = abscwt[:,25667:27748]
gesto_7 = abscwt[:,30558:32232]
gesto_8 = abscwt[:,34109:35928] 
gesto_9 = abscwt[:,39642:41659] 
gesto_10 = abscwt[:,45363:47547] 
gesto_11 = abscwt[:,50808:52751] 
gesto_12 = abscwt[:,56004:57708]

normxgsts = abscwt
#División de segmentos desde 0 a 57,69 Hz

#Segmentación 1
normxgstsg1freqs1 = normxgsts[3800:4000,762:2243]
normxgstsg1freqs2 = normxgsts[3600:3800,762:2243]
normxgstsg1freqs3 = normxgsts[3400:3600,762:2243]
normxgstsg1freqs4 = normxgsts[3200:3400,762:2243]
normxgstsg1freqs5 = normxgsts[3000:3200,762:2243]
normxgstsg1freqs6 = normxgsts[2800:3000,762:2243]
normxgstsg1freqs7 = normxgsts[2600:2800,762:2243]
normxgstsg1freqs8 = normxgsts[2400:2600,762:2243]
normxgstsg1freqs9 = normxgsts[2200:2400,762:2243]
normxgstsg1freqs10 = normxgsts[2000:2200,762:2243]
normxgstsg1freqs11 = normxgsts[1800:2000,762:2243]
#Transformación matrices de NumPy en vectores por columna
lista_F1G1 = normxgstsg1freqs1.flatten(order='F')
lista_F2G1 = normxgstsg1freqs2.flatten(order='F')
lista_F3G1 = normxgstsg1freqs3.flatten(order='F')
lista_F4G1 = normxgstsg1freqs4.flatten(order='F')
lista_F5G1 = normxgstsg1freqs5.flatten(order='F')
lista_F6G1 = normxgstsg1freqs6.flatten(order='F')
lista_F7G1 = normxgstsg1freqs7.flatten(order='F')
lista_F8G1 = normxgstsg1freqs8.flatten(order='F')
lista_F9G1 = normxgstsg1freqs9.flatten(order='F')
lista_F10G1 = normxgstsg1freqs10.flatten(order='F')
lista_F11G1 = normxgstsg1freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G1 = mean(lista_F1G1)
Promedio_lista_F2G1 = mean(lista_F2G1)
Promedio_lista_F3G1 = mean(lista_F3G1)
Promedio_lista_F4G1 = mean(lista_F4G1)
Promedio_lista_F5G1 = mean(lista_F5G1)
Promedio_lista_F6G1 = mean(lista_F6G1)
Promedio_lista_F7G1 = mean(lista_F7G1)
Promedio_lista_F8G1 = mean(lista_F8G1)
Promedio_lista_F9G1 = mean(lista_F9G1)
Promedio_lista_F10G1 = mean(lista_F10G1)
Promedio_lista_F11G1 = mean(lista_F11G1)
PromediosG1 = []
PromediosG1.append(Promedio_lista_F1G1)
PromediosG1.append(Promedio_lista_F2G1)
PromediosG1.append(Promedio_lista_F3G1)
PromediosG1.append(Promedio_lista_F4G1)
PromediosG1.append(Promedio_lista_F5G1)
PromediosG1.append(Promedio_lista_F6G1)
PromediosG1.append(Promedio_lista_F7G1)
PromediosG1.append(Promedio_lista_F8G1)
PromediosG1.append(Promedio_lista_F9G1)
PromediosG1.append(Promedio_lista_F10G1)
PromediosG1.append(Promedio_lista_F11G1)

#Segmentación G2
normxgstsg2freqs1 = normxgsts[3800:4000,4044:5629]
normxgstsg2freqs2 = normxgsts[3600:3800,4044:5629]
normxgstsg2freqs3 = normxgsts[3400:3600,4044:5629]
normxgstsg2freqs4 = normxgsts[3200:3400,4044:5629]
normxgstsg2freqs5 = normxgsts[3000:3200,4044:5629]
normxgstsg2freqs6 = normxgsts[2800:3000,4044:5629]
normxgstsg2freqs7 = normxgsts[2600:2800,4044:5629]
normxgstsg2freqs8 = normxgsts[2400:2600,4044:5629]
normxgstsg2freqs9 = normxgsts[2200:2400,4044:5629]
normxgstsg2freqs10 = normxgsts[2000:2200,4044:5629]
normxgstsg2freqs11 = normxgsts[1800:2000,4044:5629]
#Transformación matrices de NumPy en vectores por columna
lista_F1G2 = normxgstsg2freqs1.flatten(order='F')
lista_F2G2 = normxgstsg2freqs2.flatten(order='F')
lista_F3G2 = normxgstsg2freqs3.flatten(order='F')
lista_F4G2 = normxgstsg2freqs4.flatten(order='F')
lista_F5G2 = normxgstsg2freqs5.flatten(order='F')
lista_F6G2 = normxgstsg2freqs6.flatten(order='F')
lista_F7G2 = normxgstsg2freqs7.flatten(order='F')
lista_F8G2 = normxgstsg2freqs8.flatten(order='F')
lista_F9G2 = normxgstsg2freqs9.flatten(order='F')
lista_F10G2 = normxgstsg2freqs10.flatten(order='F')
lista_F11G2 = normxgstsg2freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G2 = mean(lista_F1G2)
Promedio_lista_F2G2 = mean(lista_F2G2)
Promedio_lista_F3G2 = mean(lista_F3G2)
Promedio_lista_F4G2 = mean(lista_F4G2)
Promedio_lista_F5G2 = mean(lista_F5G2)
Promedio_lista_F6G2 = mean(lista_F6G2)
Promedio_lista_F7G2 = mean(lista_F7G2)
Promedio_lista_F8G2 = mean(lista_F8G2)
Promedio_lista_F9G2 = mean(lista_F9G2)
Promedio_lista_F10G2 = mean(lista_F10G2)
Promedio_lista_F11G2 = mean(lista_F11G2)
#Lista de Promedios
PromediosG2 = []
PromediosG2.append(Promedio_lista_F1G2)
PromediosG2.append(Promedio_lista_F2G2)
PromediosG2.append(Promedio_lista_F3G2)
PromediosG2.append(Promedio_lista_F4G2)
PromediosG2.append(Promedio_lista_F5G2)
PromediosG2.append(Promedio_lista_F6G2)
PromediosG2.append(Promedio_lista_F7G2)
PromediosG2.append(Promedio_lista_F8G2)
PromediosG2.append(Promedio_lista_F9G2)
PromediosG2.append(Promedio_lista_F10G2)
PromediosG2.append(Promedio_lista_F11G2)
                               
#Segmentación G3
normxgstsg3freqs1 = normxgsts[3800:4000,9069:11028]
normxgstsg3freqs2 = normxgsts[3600:3800,9069:11028]
normxgstsg3freqs3 = normxgsts[3400:3600,9069:11028]
normxgstsg3freqs4 = normxgsts[3200:3400,9069:11028]
normxgstsg3freqs5 = normxgsts[3000:3200,9069:11028]
normxgstsg3freqs6 = normxgsts[2800:3000,9069:11028]
normxgstsg3freqs7 = normxgsts[2600:2800,9069:11028]
normxgstsg3freqs8 = normxgsts[2400:2600,9069:11028]
normxgstsg3freqs9 = normxgsts[2200:2400,9069:11028]
normxgstsg3freqs10 = normxgsts[2000:2200,9069:11028]
normxgstsg3freqs11 = normxgsts[1800:2000,9069:11028]
#Transformación matrices de NumPy en vectores por columna
lista_F1G3 = normxgstsg3freqs1.flatten(order='F')
lista_F2G3 = normxgstsg3freqs2.flatten(order='F')
lista_F3G3 = normxgstsg3freqs3.flatten(order='F')
lista_F4G3 = normxgstsg3freqs4.flatten(order='F')
lista_F5G3 = normxgstsg3freqs5.flatten(order='F')
lista_F6G3 = normxgstsg3freqs6.flatten(order='F')
lista_F7G3 = normxgstsg3freqs7.flatten(order='F')
lista_F8G3 = normxgstsg3freqs8.flatten(order='F')
lista_F9G3 = normxgstsg3freqs9.flatten(order='F')
lista_F10G3 = normxgstsg3freqs10.flatten(order='F')
lista_F11G3 = normxgstsg3freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G3 = mean(lista_F1G3)
Promedio_lista_F2G3 = mean(lista_F2G3)
Promedio_lista_F3G3 = mean(lista_F3G3)
Promedio_lista_F4G3 = mean(lista_F4G3)
Promedio_lista_F5G3 = mean(lista_F5G3)
Promedio_lista_F6G3 = mean(lista_F6G3)
Promedio_lista_F7G3 = mean(lista_F7G3)
Promedio_lista_F8G3 = mean(lista_F8G3)
Promedio_lista_F9G3 = mean(lista_F9G3)
Promedio_lista_F10G3 = mean(lista_F10G3)
Promedio_lista_F11G3 = mean(lista_F11G3)
#Lista de Promedios
PromediosG3 = []
PromediosG3.append(Promedio_lista_F1G3)
PromediosG3.append(Promedio_lista_F2G3)
PromediosG3.append(Promedio_lista_F3G3)
PromediosG3.append(Promedio_lista_F4G3)
PromediosG3.append(Promedio_lista_F5G3)
PromediosG3.append(Promedio_lista_F6G3)
PromediosG3.append(Promedio_lista_F7G3)
PromediosG3.append(Promedio_lista_F8G3)
PromediosG3.append(Promedio_lista_F9G3)
PromediosG3.append(Promedio_lista_F10G3)
PromediosG3.append(Promedio_lista_F11G3)   

#Segmentación G4
normxgstsg4freqs1 = normxgsts[3800:4000,14567:16475]
normxgstsg4freqs2 = normxgsts[3600:3800,14567:16475]
normxgstsg4freqs3 = normxgsts[3400:3600,14567:16475]
normxgstsg4freqs4 = normxgsts[3200:3400,14567:16475]
normxgstsg4freqs5 = normxgsts[3000:3200,14567:16475]
normxgstsg4freqs6 = normxgsts[2800:3000,14567:16475]
normxgstsg4freqs7 = normxgsts[2600:2800,14567:16475]
normxgstsg4freqs8 = normxgsts[2400:2600,14567:16475]
normxgstsg4freqs9 = normxgsts[2200:2400,14567:16475]
normxgstsg4freqs10 = normxgsts[2000:2200,14567:16475]
normxgstsg4freqs11 = normxgsts[1800:2000,14567:16475]
#Transformación matrices de NumPy en vectores por columna
lista_F1G4 = normxgstsg4freqs1.flatten(order='F')
lista_F2G4 = normxgstsg4freqs2.flatten(order='F')
lista_F3G4 = normxgstsg4freqs3.flatten(order='F')
lista_F4G4 = normxgstsg4freqs4.flatten(order='F')
lista_F5G4 = normxgstsg4freqs5.flatten(order='F')
lista_F6G4 = normxgstsg4freqs6.flatten(order='F')
lista_F7G4 = normxgstsg4freqs7.flatten(order='F')
lista_F8G4 = normxgstsg4freqs8.flatten(order='F')
lista_F9G4 = normxgstsg4freqs9.flatten(order='F')
lista_F10G4 = normxgstsg4freqs10.flatten(order='F')
lista_F11G4 = normxgstsg4freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G4 = mean(lista_F1G4)
Promedio_lista_F2G4 = mean(lista_F2G4)
Promedio_lista_F3G4 = mean(lista_F3G4)
Promedio_lista_F4G4 = mean(lista_F4G4)
Promedio_lista_F5G4 = mean(lista_F5G4)
Promedio_lista_F6G4 = mean(lista_F6G4)
Promedio_lista_F7G4 = mean(lista_F7G4)
Promedio_lista_F8G4 = mean(lista_F8G4)
Promedio_lista_F9G4 = mean(lista_F9G4)
Promedio_lista_F10G4 = mean(lista_F10G4)
Promedio_lista_F11G4 = mean(lista_F11G4)
#Lista de Promedios
PromediosG4 = []
PromediosG4.append(Promedio_lista_F1G4)
PromediosG4.append(Promedio_lista_F2G4)
PromediosG4.append(Promedio_lista_F3G4)
PromediosG4.append(Promedio_lista_F4G4)
PromediosG4.append(Promedio_lista_F5G4)
PromediosG4.append(Promedio_lista_F6G4)
PromediosG4.append(Promedio_lista_F7G4)
PromediosG4.append(Promedio_lista_F8G4)
PromediosG4.append(Promedio_lista_F9G4)
PromediosG4.append(Promedio_lista_F10G4)
PromediosG4.append(Promedio_lista_F11G4)    

#Segmentación G5
normxgstsg5freqs1 = normxgsts[3800:4000,19794:22042]
normxgstsg5freqs2 = normxgsts[3600:3800,19794:22042]
normxgstsg5freqs3 = normxgsts[3400:3600,19794:22042]
normxgstsg5freqs4 = normxgsts[3200:3400,19794:22042]
normxgstsg5freqs5 = normxgsts[3000:3200,19794:22042]
normxgstsg5freqs6 = normxgsts[2800:3000,19794:22042]
normxgstsg5freqs7 = normxgsts[2600:2800,19794:22042]
normxgstsg5freqs8 = normxgsts[2400:2600,19794:22042]
normxgstsg5freqs9 = normxgsts[2200:2400,19794:22042]
normxgstsg5freqs10 = normxgsts[2000:2200,19794:22042]
normxgstsg5freqs11 = normxgsts[1800:2000,19794:22042]
#Transformación matrices de NumPy en vectores por columna
lista_F1G5 = normxgstsg5freqs1.flatten(order='F')
lista_F2G5 = normxgstsg5freqs2.flatten(order='F')
lista_F3G5 = normxgstsg5freqs3.flatten(order='F')
lista_F4G5 = normxgstsg5freqs4.flatten(order='F')
lista_F5G5 = normxgstsg5freqs5.flatten(order='F')
lista_F6G5 = normxgstsg5freqs6.flatten(order='F')
lista_F7G5 = normxgstsg5freqs7.flatten(order='F')
lista_F8G5 = normxgstsg5freqs8.flatten(order='F')
lista_F9G5 = normxgstsg5freqs9.flatten(order='F')
lista_F10G5 = normxgstsg5freqs10.flatten(order='F')
lista_F11G5 = normxgstsg5freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G5 = mean(lista_F1G5)
Promedio_lista_F2G5 = mean(lista_F2G5)
Promedio_lista_F3G5 = mean(lista_F3G5)
Promedio_lista_F4G5 = mean(lista_F4G5)
Promedio_lista_F5G5 = mean(lista_F5G5)
Promedio_lista_F6G5 = mean(lista_F6G5)
Promedio_lista_F7G5 = mean(lista_F7G5)
Promedio_lista_F8G5 = mean(lista_F8G5)
Promedio_lista_F9G5 = mean(lista_F9G5)
Promedio_lista_F10G5 = mean(lista_F10G5)
Promedio_lista_F11G5 = mean(lista_F11G5)
#Lista de Promedios
PromediosG5 = []
PromediosG5.append(Promedio_lista_F1G5)
PromediosG5.append(Promedio_lista_F2G5)
PromediosG5.append(Promedio_lista_F3G5)
PromediosG5.append(Promedio_lista_F4G5)
PromediosG5.append(Promedio_lista_F5G5)
PromediosG5.append(Promedio_lista_F6G5)
PromediosG5.append(Promedio_lista_F7G5)
PromediosG5.append(Promedio_lista_F8G5)
PromediosG5.append(Promedio_lista_F9G5)
PromediosG5.append(Promedio_lista_F10G5)
PromediosG5.append(Promedio_lista_F11G5)

#Segmentación G6
normxgstsg6freqs1 = normxgsts[3800:4000,25666:27749]
normxgstsg6freqs2 = normxgsts[3600:3800,25666:27749]
normxgstsg6freqs3 = normxgsts[3400:3600,25666:27749]
normxgstsg6freqs4 = normxgsts[3200:3400,25666:27749]
normxgstsg6freqs5 = normxgsts[3000:3200,25666:27749]
normxgstsg6freqs6 = normxgsts[2800:3000,25666:27749]
normxgstsg6freqs7 = normxgsts[2600:2800,25666:27749]
normxgstsg6freqs8 = normxgsts[2400:2600,25666:27749]
normxgstsg6freqs9 = normxgsts[2200:2400,25666:27749]
normxgstsg6freqs10 = normxgsts[2000:2200,25666:27749]
normxgstsg6freqs11 = normxgsts[1800:2000,25666:27749]
#Transformación matrices de NumPy en vectores por columna
lista_F1G6 = normxgstsg6freqs1.flatten(order='F')
lista_F2G6 = normxgstsg6freqs2.flatten(order='F')
lista_F3G6 = normxgstsg6freqs3.flatten(order='F')
lista_F4G6 = normxgstsg6freqs4.flatten(order='F')
lista_F5G6 = normxgstsg6freqs5.flatten(order='F')
lista_F6G6 = normxgstsg6freqs6.flatten(order='F')
lista_F7G6 = normxgstsg6freqs7.flatten(order='F')
lista_F8G6 = normxgstsg6freqs8.flatten(order='F')
lista_F9G6 = normxgstsg6freqs9.flatten(order='F')
lista_F10G6 = normxgstsg6freqs10.flatten(order='F')
lista_F11G6 = normxgstsg6freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G6 = mean(lista_F1G6)
Promedio_lista_F2G6 = mean(lista_F2G6)
Promedio_lista_F3G6 = mean(lista_F3G6)
Promedio_lista_F4G6 = mean(lista_F4G6)
Promedio_lista_F5G6 = mean(lista_F5G6)
Promedio_lista_F6G6 = mean(lista_F6G6)
Promedio_lista_F7G6 = mean(lista_F7G6)
Promedio_lista_F8G6 = mean(lista_F8G6)
Promedio_lista_F9G6 = mean(lista_F9G6)
Promedio_lista_F10G6 = mean(lista_F10G6)
Promedio_lista_F11G6 = mean(lista_F11G6)
#Lista de Promedios
PromediosG6 = []
PromediosG6.append(Promedio_lista_F1G6)
PromediosG6.append(Promedio_lista_F2G6)
PromediosG6.append(Promedio_lista_F3G6)
PromediosG6.append(Promedio_lista_F4G6)
PromediosG6.append(Promedio_lista_F5G6)
PromediosG6.append(Promedio_lista_F6G6)
PromediosG6.append(Promedio_lista_F7G6)
PromediosG6.append(Promedio_lista_F8G6)
PromediosG6.append(Promedio_lista_F9G6)
PromediosG6.append(Promedio_lista_F10G6)
PromediosG6.append(Promedio_lista_F11G6)

#Segmentación G7
normxgstsg7freqs1 = normxgsts[3800:4000,30557:32233]
normxgstsg7freqs2 = normxgsts[3600:3800,30557:32233]
normxgstsg7freqs3 = normxgsts[3400:3600,30557:32233]
normxgstsg7freqs4 = normxgsts[3200:3400,30557:32233]
normxgstsg7freqs5 = normxgsts[3000:3200,30557:32233]
normxgstsg7freqs6 = normxgsts[2800:3000,30557:32233]
normxgstsg7freqs7 = normxgsts[2600:2800,30557:32233]
normxgstsg7freqs8 = normxgsts[2400:2600,30557:32233]
normxgstsg7freqs9 = normxgsts[2200:2400,30557:32233]
normxgstsg7freqs10 = normxgsts[2000:2200,30557:32233]
normxgstsg7freqs11 = normxgsts[1800:2000,30557:32233]
#Transformación matrices de NumPy en vectores por columna
lista_F1G7 = normxgstsg7freqs1.flatten(order='F')
lista_F2G7 = normxgstsg7freqs2.flatten(order='F')
lista_F3G7 = normxgstsg7freqs3.flatten(order='F')
lista_F4G7 = normxgstsg7freqs4.flatten(order='F')
lista_F5G7 = normxgstsg7freqs5.flatten(order='F')
lista_F6G7 = normxgstsg7freqs6.flatten(order='F')
lista_F7G7 = normxgstsg7freqs7.flatten(order='F')
lista_F8G7 = normxgstsg7freqs8.flatten(order='F')
lista_F9G7 = normxgstsg7freqs9.flatten(order='F')
lista_F10G7 = normxgstsg7freqs10.flatten(order='F')
lista_F11G7 = normxgstsg7freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G7 = mean(lista_F1G7)
Promedio_lista_F2G7 = mean(lista_F2G7)
Promedio_lista_F3G7 = mean(lista_F3G7)
Promedio_lista_F4G7 = mean(lista_F4G7)
Promedio_lista_F5G7 = mean(lista_F5G7)
Promedio_lista_F6G7 = mean(lista_F6G7)
Promedio_lista_F7G7 = mean(lista_F7G7)
Promedio_lista_F8G7 = mean(lista_F8G7)
Promedio_lista_F9G7 = mean(lista_F9G7)
Promedio_lista_F10G7 = mean(lista_F10G7)
Promedio_lista_F11G7 = mean(lista_F11G7)
#Lista de Promedios
PromediosG7 = []
PromediosG7.append(Promedio_lista_F1G7)
PromediosG7.append(Promedio_lista_F2G7)
PromediosG7.append(Promedio_lista_F3G7)
PromediosG7.append(Promedio_lista_F4G7)
PromediosG7.append(Promedio_lista_F5G7)
PromediosG7.append(Promedio_lista_F6G7)
PromediosG7.append(Promedio_lista_F7G7)
PromediosG7.append(Promedio_lista_F8G7)
PromediosG7.append(Promedio_lista_F9G7)
PromediosG7.append(Promedio_lista_F10G7)
PromediosG7.append(Promedio_lista_F11G7)

#Segmentación G8
normxgstsg8freqs1 = normxgsts[3800:4000,34108:35929]
normxgstsg8freqs2 = normxgsts[3600:3800,34108:35929]
normxgstsg8freqs3 = normxgsts[3400:3600,34108:35929]
normxgstsg8freqs4 = normxgsts[3200:3400,34108:35929]
normxgstsg8freqs5 = normxgsts[3000:3200,34108:35929]
normxgstsg8freqs6 = normxgsts[2800:3000,34108:35929]
normxgstsg8freqs7 = normxgsts[2600:2800,34108:35929]
normxgstsg8freqs8 = normxgsts[2400:2600,34108:35929]
normxgstsg8freqs9 = normxgsts[2200:2400,34108:35929]
normxgstsg8freqs10 = normxgsts[2000:2200,34108:35929]
normxgstsg8freqs11 = normxgsts[1800:2000,34108:35929]
#Transformación matrices de NumPy en vectores por columna
lista_F1G8 = normxgstsg8freqs1.flatten(order='F')
lista_F2G8 = normxgstsg8freqs2.flatten(order='F')
lista_F3G8 = normxgstsg8freqs3.flatten(order='F')
lista_F4G8 = normxgstsg8freqs4.flatten(order='F')
lista_F5G8 = normxgstsg8freqs5.flatten(order='F')
lista_F6G8 = normxgstsg8freqs6.flatten(order='F')
lista_F7G8 = normxgstsg8freqs7.flatten(order='F')
lista_F8G8 = normxgstsg8freqs8.flatten(order='F')
lista_F9G8 = normxgstsg8freqs9.flatten(order='F')
lista_F10G8 = normxgstsg8freqs10.flatten(order='F')
lista_F11G8 = normxgstsg8freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G8 = mean(lista_F1G8)
Promedio_lista_F2G8 = mean(lista_F2G8)
Promedio_lista_F3G8 = mean(lista_F3G8)
Promedio_lista_F4G8 = mean(lista_F4G8)
Promedio_lista_F5G8 = mean(lista_F5G8)
Promedio_lista_F6G8 = mean(lista_F6G8)
Promedio_lista_F7G8 = mean(lista_F7G8)
Promedio_lista_F8G8 = mean(lista_F8G8)
Promedio_lista_F9G8 = mean(lista_F9G8)
Promedio_lista_F10G8 = mean(lista_F10G8)
Promedio_lista_F11G8 = mean(lista_F11G8)
#Lista de Promedios
PromediosG8 = []
PromediosG8.append(Promedio_lista_F1G8)
PromediosG8.append(Promedio_lista_F2G8)
PromediosG8.append(Promedio_lista_F3G8)
PromediosG8.append(Promedio_lista_F4G8)
PromediosG8.append(Promedio_lista_F5G8)
PromediosG8.append(Promedio_lista_F6G8)
PromediosG8.append(Promedio_lista_F7G8)
PromediosG8.append(Promedio_lista_F8G8)
PromediosG8.append(Promedio_lista_F9G8)
PromediosG8.append(Promedio_lista_F10G8)
PromediosG8.append(Promedio_lista_F11G8)

#Segmentación G9
normxgstsg9freqs1 = normxgsts[3800:4000,39641:41660]
normxgstsg9freqs2 = normxgsts[3600:3800,39641:41660]
normxgstsg9freqs3 = normxgsts[3400:3600,39641:41660]
normxgstsg9freqs4 = normxgsts[3200:3400,39641:41660]
normxgstsg9freqs5 = normxgsts[3000:3200,39641:41660]
normxgstsg9freqs6 = normxgsts[2800:3000,39641:41660]
normxgstsg9freqs7 = normxgsts[2600:2800,39641:41660]
normxgstsg9freqs8 = normxgsts[2400:2600,39641:41660]
normxgstsg9freqs9 = normxgsts[2200:2400,39641:41660]
normxgstsg9freqs10 = normxgsts[2000:2200,39641:41660]
normxgstsg9freqs11 = normxgsts[1800:2000,39641:41660]
#Transformación matrices de NumPy en vectores por columna
lista_F1G9 = normxgstsg9freqs1.flatten(order='F')
lista_F2G9 = normxgstsg9freqs2.flatten(order='F')
lista_F3G9 = normxgstsg9freqs3.flatten(order='F')
lista_F4G9 = normxgstsg9freqs4.flatten(order='F')
lista_F5G9 = normxgstsg9freqs5.flatten(order='F')
lista_F6G9 = normxgstsg9freqs6.flatten(order='F')
lista_F7G9 = normxgstsg9freqs7.flatten(order='F')
lista_F8G9 = normxgstsg9freqs8.flatten(order='F')
lista_F9G9 = normxgstsg9freqs9.flatten(order='F')
lista_F10G9 = normxgstsg9freqs10.flatten(order='F')
lista_F11G9 = normxgstsg9freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G9 = mean(lista_F1G9)
Promedio_lista_F2G9 = mean(lista_F2G9)
Promedio_lista_F3G9 = mean(lista_F3G9)
Promedio_lista_F4G9 = mean(lista_F4G9)
Promedio_lista_F5G9 = mean(lista_F5G9)
Promedio_lista_F6G9 = mean(lista_F6G9)
Promedio_lista_F7G9 = mean(lista_F7G9)
Promedio_lista_F8G9 = mean(lista_F8G9)
Promedio_lista_F9G9 = mean(lista_F9G9)
Promedio_lista_F10G9 = mean(lista_F10G9)
Promedio_lista_F11G9 = mean(lista_F11G9)
#Lista de Promedios
PromediosG9 = []
PromediosG9.append(Promedio_lista_F1G9)
PromediosG9.append(Promedio_lista_F2G9)
PromediosG9.append(Promedio_lista_F3G9)
PromediosG9.append(Promedio_lista_F4G9)
PromediosG9.append(Promedio_lista_F5G9)
PromediosG9.append(Promedio_lista_F6G9)
PromediosG9.append(Promedio_lista_F7G9)
PromediosG9.append(Promedio_lista_F8G9)
PromediosG9.append(Promedio_lista_F9G9)
PromediosG9.append(Promedio_lista_F10G9)
PromediosG9.append(Promedio_lista_F11G9)

#Segmentación G10
normxgstsg10freqs1 = normxgsts[3800:4000,45362:47548]
normxgstsg10freqs2 = normxgsts[3600:3800,45362:47548]
normxgstsg10freqs3 = normxgsts[3400:3600,45362:47548]
normxgstsg10freqs4 = normxgsts[3200:3400,45362:47548]
normxgstsg10freqs5 = normxgsts[3000:3200,45362:47548]
normxgstsg10freqs6 = normxgsts[2800:3000,45362:47548]
normxgstsg10freqs7 = normxgsts[2600:2800,45362:47548]
normxgstsg10freqs8 = normxgsts[2400:2600,45362:47548]
normxgstsg10freqs9 = normxgsts[2200:2400,45362:47548]
normxgstsg10freqs10 = normxgsts[2000:2200,45362:47548]
normxgstsg10freqs11 = normxgsts[1800:2000,45362:47548]
#Transformación matrices de NumPy en vectores por columna
lista_F1G10 = normxgstsg10freqs1.flatten(order='F')
lista_F2G10 = normxgstsg10freqs2.flatten(order='F')
lista_F3G10 = normxgstsg10freqs3.flatten(order='F')
lista_F4G10 = normxgstsg10freqs4.flatten(order='F')
lista_F5G10 = normxgstsg10freqs5.flatten(order='F')
lista_F6G10 = normxgstsg10freqs6.flatten(order='F')
lista_F7G10 = normxgstsg10freqs7.flatten(order='F')
lista_F8G10 = normxgstsg10freqs8.flatten(order='F')
lista_F9G10 = normxgstsg10freqs9.flatten(order='F')
lista_F10G10 = normxgstsg10freqs10.flatten(order='F')
lista_F11G10 = normxgstsg10freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G10 = mean(lista_F1G10)
Promedio_lista_F2G10 = mean(lista_F2G10)
Promedio_lista_F3G10 = mean(lista_F3G10)
Promedio_lista_F4G10 = mean(lista_F4G10)
Promedio_lista_F5G10 = mean(lista_F5G10)
Promedio_lista_F6G10 = mean(lista_F6G10)
Promedio_lista_F7G10 = mean(lista_F7G10)
Promedio_lista_F8G10 = mean(lista_F8G10)
Promedio_lista_F9G10 = mean(lista_F9G10)
Promedio_lista_F10G10 = mean(lista_F10G10)
Promedio_lista_F11G10 = mean(lista_F11G10)
#Lista de Promedios
PromediosG10 = []
PromediosG10.append(Promedio_lista_F1G10)
PromediosG10.append(Promedio_lista_F2G10)
PromediosG10.append(Promedio_lista_F3G10)
PromediosG10.append(Promedio_lista_F4G10)
PromediosG10.append(Promedio_lista_F5G10)
PromediosG10.append(Promedio_lista_F6G10)
PromediosG10.append(Promedio_lista_F7G10)
PromediosG10.append(Promedio_lista_F8G10)
PromediosG10.append(Promedio_lista_F9G10)
PromediosG10.append(Promedio_lista_F10G10)
PromediosG10.append(Promedio_lista_F11G10)

#Segmentación G11
normxgstsg11freqs1 = normxgsts[3800:4000,50807:52752]
normxgstsg11freqs2 = normxgsts[3600:3800,50807:52752]
normxgstsg11freqs3 = normxgsts[3400:3600,50807:52752]
normxgstsg11freqs4 = normxgsts[3200:3400,50807:52752]
normxgstsg11freqs5 = normxgsts[3000:3200,50807:52752]
normxgstsg11freqs6 = normxgsts[2800:3000,50807:52752]
normxgstsg11freqs7 = normxgsts[2600:2800,50807:52752]
normxgstsg11freqs8 = normxgsts[2400:2600,50807:52752]
normxgstsg11freqs9 = normxgsts[2200:2400,50807:52752]
normxgstsg11freqs10 = normxgsts[2000:2200,50807:52752]
normxgstsg11freqs11 = normxgsts[1800:2000,50807:52752]
#Transformación matrices de NumPy en vectores por columna
lista_F1G11 = normxgstsg11freqs1.flatten(order='F')
lista_F2G11 = normxgstsg11freqs2.flatten(order='F')
lista_F3G11 = normxgstsg11freqs3.flatten(order='F')
lista_F4G11 = normxgstsg11freqs4.flatten(order='F')
lista_F5G11 = normxgstsg11freqs5.flatten(order='F')
lista_F6G11 = normxgstsg11freqs6.flatten(order='F')
lista_F7G11 = normxgstsg11freqs7.flatten(order='F')
lista_F8G11 = normxgstsg11freqs8.flatten(order='F')
lista_F9G11 = normxgstsg11freqs9.flatten(order='F')
lista_F10G11 = normxgstsg11freqs10.flatten(order='F')
lista_F11G11 = normxgstsg11freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G11 = mean(lista_F1G11)
Promedio_lista_F2G11 = mean(lista_F2G11)
Promedio_lista_F3G11 = mean(lista_F3G11)
Promedio_lista_F4G11 = mean(lista_F4G11)
Promedio_lista_F5G11 = mean(lista_F5G11)
Promedio_lista_F6G11 = mean(lista_F6G11)
Promedio_lista_F7G11 = mean(lista_F7G11)
Promedio_lista_F8G11 = mean(lista_F8G11)
Promedio_lista_F9G11 = mean(lista_F9G11)
Promedio_lista_F10G11 = mean(lista_F10G11)
Promedio_lista_F11G11 = mean(lista_F11G11)
#Lista de Promedios
PromediosG11 = []
PromediosG11.append(Promedio_lista_F1G11)
PromediosG11.append(Promedio_lista_F2G11)
PromediosG11.append(Promedio_lista_F3G11)
PromediosG11.append(Promedio_lista_F4G11)
PromediosG11.append(Promedio_lista_F5G11)
PromediosG11.append(Promedio_lista_F6G11)
PromediosG11.append(Promedio_lista_F7G11)
PromediosG11.append(Promedio_lista_F8G11)
PromediosG11.append(Promedio_lista_F9G11)
PromediosG11.append(Promedio_lista_F10G11)
PromediosG11.append(Promedio_lista_F11G11)

#Segmentación G12
normxgstsg12freqs1 = normxgsts[3800:4000,56003:57709]
normxgstsg12freqs2 = normxgsts[3600:3800,56003:57709]
normxgstsg12freqs3 = normxgsts[3400:3600,56003:57709]
normxgstsg12freqs4 = normxgsts[3200:3400,56003:57709]
normxgstsg12freqs5 = normxgsts[3000:3200,56003:57709]
normxgstsg12freqs6 = normxgsts[2800:3000,56003:57709]
normxgstsg12freqs7 = normxgsts[2600:2800,56003:57709]
normxgstsg12freqs8 = normxgsts[2400:2600,56003:57709]
normxgstsg12freqs9 = normxgsts[2200:2400,56003:57709]
normxgstsg12freqs10 = normxgsts[2000:2200,56003:57709]
normxgstsg12freqs11 = normxgsts[1800:2000,56003:57709]
#Transformación matrices de NumPy en vectores por columna
lista_F1G12 = normxgstsg12freqs1.flatten(order='F')
lista_F2G12 = normxgstsg12freqs2.flatten(order='F')
lista_F3G12 = normxgstsg12freqs3.flatten(order='F')
lista_F4G12 = normxgstsg12freqs4.flatten(order='F')
lista_F5G12 = normxgstsg12freqs5.flatten(order='F')
lista_F6G12 = normxgstsg12freqs6.flatten(order='F')
lista_F7G12 = normxgstsg12freqs7.flatten(order='F')
lista_F8G12 = normxgstsg12freqs8.flatten(order='F')
lista_F9G12 = normxgstsg12freqs9.flatten(order='F')
lista_F10G12 = normxgstsg12freqs10.flatten(order='F')
lista_F11G12 = normxgstsg12freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G12 = mean(lista_F1G12)
Promedio_lista_F2G12 = mean(lista_F2G12)
Promedio_lista_F3G12 = mean(lista_F3G12)
Promedio_lista_F4G12 = mean(lista_F4G12)
Promedio_lista_F5G12 = mean(lista_F5G12)
Promedio_lista_F6G12 = mean(lista_F6G12)
Promedio_lista_F7G12 = mean(lista_F7G12)
Promedio_lista_F8G12 = mean(lista_F8G12)
Promedio_lista_F9G12 = mean(lista_F9G12)
Promedio_lista_F10G12 = mean(lista_F10G12)
Promedio_lista_F11G12 = mean(lista_F11G12)
#Lista de Promedios
PromediosG12 = []
PromediosG12.append(Promedio_lista_F1G12)
PromediosG12.append(Promedio_lista_F2G12)
PromediosG12.append(Promedio_lista_F3G12)
PromediosG12.append(Promedio_lista_F4G12)
PromediosG12.append(Promedio_lista_F5G12)
PromediosG12.append(Promedio_lista_F6G12)
PromediosG12.append(Promedio_lista_F7G12)
PromediosG12.append(Promedio_lista_F8G12)
PromediosG12.append(Promedio_lista_F9G12)
PromediosG12.append(Promedio_lista_F10G12)
PromediosG12.append(Promedio_lista_F11G12)

Promedios = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6, 
             PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]
PromediosFrecuencias = np.array(Promedios); 
PromediosFrecuencias = np.transpose(PromediosFrecuencias)

DF = pd.DataFrame(PromediosFrecuencias) 
#DF.to_csv("Promedios Paciente 3 Canal 2.csv")

PromediosG1 = np.array(PromediosG1)
PromediosG1 = np.transpose(PromediosG1)

PromediosG2 = np.array(PromediosG2)
PromediosG2 = np.transpose(PromediosG2)

PromediosG3 = np.array(PromediosG3)
PromediosG3 = np.transpose(PromediosG3)

PromediosG4 = np.array(PromediosG4)
PromediosG4 = np.transpose(PromediosG4)

PromediosG5 = np.array(PromediosG5)
PromediosG5 = np.transpose(PromediosG5)

PromediosG6 = np.array(PromediosG6)
PromediosG6 = np.transpose(PromediosG6)

PromediosG7 = np.array(PromediosG7)
PromediosG7 = np.transpose(PromediosG7)

PromediosG8 = np.array(PromediosG8)
PromediosG8 = np.transpose(PromediosG8)

PromediosG9 = np.array(PromediosG9)
PromediosG9 = np.transpose(PromediosG9)

PromediosG10 = np.array(PromediosG10)
PromediosG10 = np.transpose(PromediosG10)

PromediosG11 = np.array(PromediosG11)
PromediosG11 = np.transpose(PromediosG11)

PromediosG12 = np.array(PromediosG12)
PromediosG12 = np.transpose(PromediosG12)

data = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6,
        PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]

#Secuencia Consecutiva de Matrices 
SPromediosG1 = list(itertools.chain(PromediosG1, PromediosG7))
SPromediosG1 = np.array(SPromediosG1)
SPromediosG2 = list(itertools.chain(PromediosG2, PromediosG8))
SPromediosG2 = np.array(SPromediosG2)
SPromediosG3 = list(itertools.chain(PromediosG3, PromediosG9))
SPromediosG3 = np.array(SPromediosG3)
SPromediosG4 = list(itertools.chain(PromediosG4, PromediosG10))
SPromediosG4 = np.array(SPromediosG4)
SPromediosG5 = list(itertools.chain(PromediosG5, PromediosG11))
SPromediosG5 = np.array(SPromediosG5)
SPromediosG6 = list(itertools.chain(PromediosG6, PromediosG12))
SPromediosG6 = np.array(SPromediosG6)
SPromedios = [SPromediosG1, SPromediosG2, SPromediosG3, SPromediosG4, SPromediosG5, SPromediosG6]

SPromediosU0 = list(itertools.chain(SPromedios[0], SPromedios[1]))
SPromediosU1 = list(itertools.chain(SPromediosU0, SPromedios[2]))
SPromediosU2 = list(itertools.chain(SPromediosU1, SPromedios[3]))
SPromediosU3 = list(itertools.chain(SPromediosU2, SPromedios[4]))
SPromediosU4 = list(itertools.chain(SPromediosU3, SPromedios[5]))
SPromediosU = np.array(SPromediosU4)

#Histograma
plt.figure(figsize =(16, 10))
plt.hist(SPromediosU)
plt.grid()
plt.title('Histograma de la Secuencia Consecutiva de Promedios')
plt.xlabel('Valor de la variable')
plt.ylabel('Conteo')
plt.savefig('Histograma Promedios Gesto 1 P3C6.png', dpi = 201)
plt.show()


#scipy stats Wilcoxon 
WCresultados = []
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[1], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[2], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)


stat, p_value = wilcoxon(SPromedios[3], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[4], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[5], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)

WCresultados = np.array(WCresultados)
WCresultados = np.reshape(WCresultados,[6,6])

DF = pd.DataFrame(WCresultados) 
DF.to_csv("Wilcoxon Resultados entre gestos distintos P3C2.csv")


emg_funcional = np.array(emgpacientes03[1])
#Lectura archivo 
samplerate = 1e3 #Frecuencia de muestreo
length = emg_funcional.shape[0] / samplerate #Duración total Electromiografía en segundos
#print("Duración Total = {length}s")

#Parametros CWT 
num = 4000 #Número de sampleo generado
wavelet = 'cmor1.0-1.5' # Bugs Fixed https://pywavelets.readthedocs.io/en/latest/release.1.1.0.html 
maxesc = np.log2(1500)
widths = np.logspace(0,maxesc, num=num, base=2,dtype='int')
#print("Escalas Usadas: ", widths)
dt = 1/samplerate # diferencia de intervalos de tiempo
frequencies = pywt.scale2frequency(wavelet, widths) / dt # Obtención de frecuencias correspondientes a escalas Modificar aquí precisi´´on
#print("Freqs asociadas a Escalas: ", frequencies)
# Calculo transformación continua de ondículas de los anchos de matriz numpy
wavelet_coeffs, freqs = pywt.cwt(emg_funcional, widths, wavelet=wavelet, sampling_period=samplerate, method='fft')
#print("Tamaño de wavelet transform: ", wavelet_coeffs.shape)
#print("Freqs asociadas a wavelet transform: ", freqs)

#Valores Absolutos de la CWT
abscwt = abs(wavelet_coeffs) # Devolución de ABS de los números complejos es la magnitud
gesto_1 = abscwt[:,763:2242]
gesto_2 = abscwt[:,4045:5628]
gesto_3 = abscwt[:,9070:11027]
gesto_4 = abscwt[:,14568:16474]
gesto_5 = abscwt[:,19795:22041]
gesto_6 = abscwt[:,25667:27748]
gesto_7 = abscwt[:,30558:32232]
gesto_8 = abscwt[:,34109:35928] 
gesto_9 = abscwt[:,39642:41659] 
gesto_10 = abscwt[:,45363:47547] 
gesto_11 = abscwt[:,50808:52751] 
gesto_12 = abscwt[:,56004:57708]

normxgsts = abscwt
#División de segmentos desde 0 a 57,69 Hz

#Segmentación 1
normxgstsg1freqs1 = normxgsts[3800:4000,762:2243]
normxgstsg1freqs2 = normxgsts[3600:3800,762:2243]
normxgstsg1freqs3 = normxgsts[3400:3600,762:2243]
normxgstsg1freqs4 = normxgsts[3200:3400,762:2243]
normxgstsg1freqs5 = normxgsts[3000:3200,762:2243]
normxgstsg1freqs6 = normxgsts[2800:3000,762:2243]
normxgstsg1freqs7 = normxgsts[2600:2800,762:2243]
normxgstsg1freqs8 = normxgsts[2400:2600,762:2243]
normxgstsg1freqs9 = normxgsts[2200:2400,762:2243]
normxgstsg1freqs10 = normxgsts[2000:2200,762:2243]
normxgstsg1freqs11 = normxgsts[1800:2000,762:2243]
#Transformación matrices de NumPy en vectores por columna
lista_F1G1 = normxgstsg1freqs1.flatten(order='F')
lista_F2G1 = normxgstsg1freqs2.flatten(order='F')
lista_F3G1 = normxgstsg1freqs3.flatten(order='F')
lista_F4G1 = normxgstsg1freqs4.flatten(order='F')
lista_F5G1 = normxgstsg1freqs5.flatten(order='F')
lista_F6G1 = normxgstsg1freqs6.flatten(order='F')
lista_F7G1 = normxgstsg1freqs7.flatten(order='F')
lista_F8G1 = normxgstsg1freqs8.flatten(order='F')
lista_F9G1 = normxgstsg1freqs9.flatten(order='F')
lista_F10G1 = normxgstsg1freqs10.flatten(order='F')
lista_F11G1 = normxgstsg1freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G1 = mean(lista_F1G1)
Promedio_lista_F2G1 = mean(lista_F2G1)
Promedio_lista_F3G1 = mean(lista_F3G1)
Promedio_lista_F4G1 = mean(lista_F4G1)
Promedio_lista_F5G1 = mean(lista_F5G1)
Promedio_lista_F6G1 = mean(lista_F6G1)
Promedio_lista_F7G1 = mean(lista_F7G1)
Promedio_lista_F8G1 = mean(lista_F8G1)
Promedio_lista_F9G1 = mean(lista_F9G1)
Promedio_lista_F10G1 = mean(lista_F10G1)
Promedio_lista_F11G1 = mean(lista_F11G1)
PromediosG1 = []
PromediosG1.append(Promedio_lista_F1G1)
PromediosG1.append(Promedio_lista_F2G1)
PromediosG1.append(Promedio_lista_F3G1)
PromediosG1.append(Promedio_lista_F4G1)
PromediosG1.append(Promedio_lista_F5G1)
PromediosG1.append(Promedio_lista_F6G1)
PromediosG1.append(Promedio_lista_F7G1)
PromediosG1.append(Promedio_lista_F8G1)
PromediosG1.append(Promedio_lista_F9G1)
PromediosG1.append(Promedio_lista_F10G1)
PromediosG1.append(Promedio_lista_F11G1)

#Segmentación G2
normxgstsg2freqs1 = normxgsts[3800:4000,4044:5629]
normxgstsg2freqs2 = normxgsts[3600:3800,4044:5629]
normxgstsg2freqs3 = normxgsts[3400:3600,4044:5629]
normxgstsg2freqs4 = normxgsts[3200:3400,4044:5629]
normxgstsg2freqs5 = normxgsts[3000:3200,4044:5629]
normxgstsg2freqs6 = normxgsts[2800:3000,4044:5629]
normxgstsg2freqs7 = normxgsts[2600:2800,4044:5629]
normxgstsg2freqs8 = normxgsts[2400:2600,4044:5629]
normxgstsg2freqs9 = normxgsts[2200:2400,4044:5629]
normxgstsg2freqs10 = normxgsts[2000:2200,4044:5629]
normxgstsg2freqs11 = normxgsts[1800:2000,4044:5629]
#Transformación matrices de NumPy en vectores por columna
lista_F1G2 = normxgstsg2freqs1.flatten(order='F')
lista_F2G2 = normxgstsg2freqs2.flatten(order='F')
lista_F3G2 = normxgstsg2freqs3.flatten(order='F')
lista_F4G2 = normxgstsg2freqs4.flatten(order='F')
lista_F5G2 = normxgstsg2freqs5.flatten(order='F')
lista_F6G2 = normxgstsg2freqs6.flatten(order='F')
lista_F7G2 = normxgstsg2freqs7.flatten(order='F')
lista_F8G2 = normxgstsg2freqs8.flatten(order='F')
lista_F9G2 = normxgstsg2freqs9.flatten(order='F')
lista_F10G2 = normxgstsg2freqs10.flatten(order='F')
lista_F11G2 = normxgstsg2freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G2 = mean(lista_F1G2)
Promedio_lista_F2G2 = mean(lista_F2G2)
Promedio_lista_F3G2 = mean(lista_F3G2)
Promedio_lista_F4G2 = mean(lista_F4G2)
Promedio_lista_F5G2 = mean(lista_F5G2)
Promedio_lista_F6G2 = mean(lista_F6G2)
Promedio_lista_F7G2 = mean(lista_F7G2)
Promedio_lista_F8G2 = mean(lista_F8G2)
Promedio_lista_F9G2 = mean(lista_F9G2)
Promedio_lista_F10G2 = mean(lista_F10G2)
Promedio_lista_F11G2 = mean(lista_F11G2)
#Lista de Promedios
PromediosG2 = []
PromediosG2.append(Promedio_lista_F1G2)
PromediosG2.append(Promedio_lista_F2G2)
PromediosG2.append(Promedio_lista_F3G2)
PromediosG2.append(Promedio_lista_F4G2)
PromediosG2.append(Promedio_lista_F5G2)
PromediosG2.append(Promedio_lista_F6G2)
PromediosG2.append(Promedio_lista_F7G2)
PromediosG2.append(Promedio_lista_F8G2)
PromediosG2.append(Promedio_lista_F9G2)
PromediosG2.append(Promedio_lista_F10G2)
PromediosG2.append(Promedio_lista_F11G2)
                               
#Segmentación G3
normxgstsg3freqs1 = normxgsts[3800:4000,9069:11028]
normxgstsg3freqs2 = normxgsts[3600:3800,9069:11028]
normxgstsg3freqs3 = normxgsts[3400:3600,9069:11028]
normxgstsg3freqs4 = normxgsts[3200:3400,9069:11028]
normxgstsg3freqs5 = normxgsts[3000:3200,9069:11028]
normxgstsg3freqs6 = normxgsts[2800:3000,9069:11028]
normxgstsg3freqs7 = normxgsts[2600:2800,9069:11028]
normxgstsg3freqs8 = normxgsts[2400:2600,9069:11028]
normxgstsg3freqs9 = normxgsts[2200:2400,9069:11028]
normxgstsg3freqs10 = normxgsts[2000:2200,9069:11028]
normxgstsg3freqs11 = normxgsts[1800:2000,9069:11028]
#Transformación matrices de NumPy en vectores por columna
lista_F1G3 = normxgstsg3freqs1.flatten(order='F')
lista_F2G3 = normxgstsg3freqs2.flatten(order='F')
lista_F3G3 = normxgstsg3freqs3.flatten(order='F')
lista_F4G3 = normxgstsg3freqs4.flatten(order='F')
lista_F5G3 = normxgstsg3freqs5.flatten(order='F')
lista_F6G3 = normxgstsg3freqs6.flatten(order='F')
lista_F7G3 = normxgstsg3freqs7.flatten(order='F')
lista_F8G3 = normxgstsg3freqs8.flatten(order='F')
lista_F9G3 = normxgstsg3freqs9.flatten(order='F')
lista_F10G3 = normxgstsg3freqs10.flatten(order='F')
lista_F11G3 = normxgstsg3freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G3 = mean(lista_F1G3)
Promedio_lista_F2G3 = mean(lista_F2G3)
Promedio_lista_F3G3 = mean(lista_F3G3)
Promedio_lista_F4G3 = mean(lista_F4G3)
Promedio_lista_F5G3 = mean(lista_F5G3)
Promedio_lista_F6G3 = mean(lista_F6G3)
Promedio_lista_F7G3 = mean(lista_F7G3)
Promedio_lista_F8G3 = mean(lista_F8G3)
Promedio_lista_F9G3 = mean(lista_F9G3)
Promedio_lista_F10G3 = mean(lista_F10G3)
Promedio_lista_F11G3 = mean(lista_F11G3)
#Lista de Promedios
PromediosG3 = []
PromediosG3.append(Promedio_lista_F1G3)
PromediosG3.append(Promedio_lista_F2G3)
PromediosG3.append(Promedio_lista_F3G3)
PromediosG3.append(Promedio_lista_F4G3)
PromediosG3.append(Promedio_lista_F5G3)
PromediosG3.append(Promedio_lista_F6G3)
PromediosG3.append(Promedio_lista_F7G3)
PromediosG3.append(Promedio_lista_F8G3)
PromediosG3.append(Promedio_lista_F9G3)
PromediosG3.append(Promedio_lista_F10G3)
PromediosG3.append(Promedio_lista_F11G3)   

#Segmentación G4
normxgstsg4freqs1 = normxgsts[3800:4000,14567:16475]
normxgstsg4freqs2 = normxgsts[3600:3800,14567:16475]
normxgstsg4freqs3 = normxgsts[3400:3600,14567:16475]
normxgstsg4freqs4 = normxgsts[3200:3400,14567:16475]
normxgstsg4freqs5 = normxgsts[3000:3200,14567:16475]
normxgstsg4freqs6 = normxgsts[2800:3000,14567:16475]
normxgstsg4freqs7 = normxgsts[2600:2800,14567:16475]
normxgstsg4freqs8 = normxgsts[2400:2600,14567:16475]
normxgstsg4freqs9 = normxgsts[2200:2400,14567:16475]
normxgstsg4freqs10 = normxgsts[2000:2200,14567:16475]
normxgstsg4freqs11 = normxgsts[1800:2000,14567:16475]
#Transformación matrices de NumPy en vectores por columna
lista_F1G4 = normxgstsg4freqs1.flatten(order='F')
lista_F2G4 = normxgstsg4freqs2.flatten(order='F')
lista_F3G4 = normxgstsg4freqs3.flatten(order='F')
lista_F4G4 = normxgstsg4freqs4.flatten(order='F')
lista_F5G4 = normxgstsg4freqs5.flatten(order='F')
lista_F6G4 = normxgstsg4freqs6.flatten(order='F')
lista_F7G4 = normxgstsg4freqs7.flatten(order='F')
lista_F8G4 = normxgstsg4freqs8.flatten(order='F')
lista_F9G4 = normxgstsg4freqs9.flatten(order='F')
lista_F10G4 = normxgstsg4freqs10.flatten(order='F')
lista_F11G4 = normxgstsg4freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G4 = mean(lista_F1G4)
Promedio_lista_F2G4 = mean(lista_F2G4)
Promedio_lista_F3G4 = mean(lista_F3G4)
Promedio_lista_F4G4 = mean(lista_F4G4)
Promedio_lista_F5G4 = mean(lista_F5G4)
Promedio_lista_F6G4 = mean(lista_F6G4)
Promedio_lista_F7G4 = mean(lista_F7G4)
Promedio_lista_F8G4 = mean(lista_F8G4)
Promedio_lista_F9G4 = mean(lista_F9G4)
Promedio_lista_F10G4 = mean(lista_F10G4)
Promedio_lista_F11G4 = mean(lista_F11G4)
#Lista de Promedios
PromediosG4 = []
PromediosG4.append(Promedio_lista_F1G4)
PromediosG4.append(Promedio_lista_F2G4)
PromediosG4.append(Promedio_lista_F3G4)
PromediosG4.append(Promedio_lista_F4G4)
PromediosG4.append(Promedio_lista_F5G4)
PromediosG4.append(Promedio_lista_F6G4)
PromediosG4.append(Promedio_lista_F7G4)
PromediosG4.append(Promedio_lista_F8G4)
PromediosG4.append(Promedio_lista_F9G4)
PromediosG4.append(Promedio_lista_F10G4)
PromediosG4.append(Promedio_lista_F11G4)    

#Segmentación G5
normxgstsg5freqs1 = normxgsts[3800:4000,19794:22042]
normxgstsg5freqs2 = normxgsts[3600:3800,19794:22042]
normxgstsg5freqs3 = normxgsts[3400:3600,19794:22042]
normxgstsg5freqs4 = normxgsts[3200:3400,19794:22042]
normxgstsg5freqs5 = normxgsts[3000:3200,19794:22042]
normxgstsg5freqs6 = normxgsts[2800:3000,19794:22042]
normxgstsg5freqs7 = normxgsts[2600:2800,19794:22042]
normxgstsg5freqs8 = normxgsts[2400:2600,19794:22042]
normxgstsg5freqs9 = normxgsts[2200:2400,19794:22042]
normxgstsg5freqs10 = normxgsts[2000:2200,19794:22042]
normxgstsg5freqs11 = normxgsts[1800:2000,19794:22042]
#Transformación matrices de NumPy en vectores por columna
lista_F1G5 = normxgstsg5freqs1.flatten(order='F')
lista_F2G5 = normxgstsg5freqs2.flatten(order='F')
lista_F3G5 = normxgstsg5freqs3.flatten(order='F')
lista_F4G5 = normxgstsg5freqs4.flatten(order='F')
lista_F5G5 = normxgstsg5freqs5.flatten(order='F')
lista_F6G5 = normxgstsg5freqs6.flatten(order='F')
lista_F7G5 = normxgstsg5freqs7.flatten(order='F')
lista_F8G5 = normxgstsg5freqs8.flatten(order='F')
lista_F9G5 = normxgstsg5freqs9.flatten(order='F')
lista_F10G5 = normxgstsg5freqs10.flatten(order='F')
lista_F11G5 = normxgstsg5freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G5 = mean(lista_F1G5)
Promedio_lista_F2G5 = mean(lista_F2G5)
Promedio_lista_F3G5 = mean(lista_F3G5)
Promedio_lista_F4G5 = mean(lista_F4G5)
Promedio_lista_F5G5 = mean(lista_F5G5)
Promedio_lista_F6G5 = mean(lista_F6G5)
Promedio_lista_F7G5 = mean(lista_F7G5)
Promedio_lista_F8G5 = mean(lista_F8G5)
Promedio_lista_F9G5 = mean(lista_F9G5)
Promedio_lista_F10G5 = mean(lista_F10G5)
Promedio_lista_F11G5 = mean(lista_F11G5)
#Lista de Promedios
PromediosG5 = []
PromediosG5.append(Promedio_lista_F1G5)
PromediosG5.append(Promedio_lista_F2G5)
PromediosG5.append(Promedio_lista_F3G5)
PromediosG5.append(Promedio_lista_F4G5)
PromediosG5.append(Promedio_lista_F5G5)
PromediosG5.append(Promedio_lista_F6G5)
PromediosG5.append(Promedio_lista_F7G5)
PromediosG5.append(Promedio_lista_F8G5)
PromediosG5.append(Promedio_lista_F9G5)
PromediosG5.append(Promedio_lista_F10G5)
PromediosG5.append(Promedio_lista_F11G5)

#Segmentación G6
normxgstsg6freqs1 = normxgsts[3800:4000,25666:27749]
normxgstsg6freqs2 = normxgsts[3600:3800,25666:27749]
normxgstsg6freqs3 = normxgsts[3400:3600,25666:27749]
normxgstsg6freqs4 = normxgsts[3200:3400,25666:27749]
normxgstsg6freqs5 = normxgsts[3000:3200,25666:27749]
normxgstsg6freqs6 = normxgsts[2800:3000,25666:27749]
normxgstsg6freqs7 = normxgsts[2600:2800,25666:27749]
normxgstsg6freqs8 = normxgsts[2400:2600,25666:27749]
normxgstsg6freqs9 = normxgsts[2200:2400,25666:27749]
normxgstsg6freqs10 = normxgsts[2000:2200,25666:27749]
normxgstsg6freqs11 = normxgsts[1800:2000,25666:27749]
#Transformación matrices de NumPy en vectores por columna
lista_F1G6 = normxgstsg6freqs1.flatten(order='F')
lista_F2G6 = normxgstsg6freqs2.flatten(order='F')
lista_F3G6 = normxgstsg6freqs3.flatten(order='F')
lista_F4G6 = normxgstsg6freqs4.flatten(order='F')
lista_F5G6 = normxgstsg6freqs5.flatten(order='F')
lista_F6G6 = normxgstsg6freqs6.flatten(order='F')
lista_F7G6 = normxgstsg6freqs7.flatten(order='F')
lista_F8G6 = normxgstsg6freqs8.flatten(order='F')
lista_F9G6 = normxgstsg6freqs9.flatten(order='F')
lista_F10G6 = normxgstsg6freqs10.flatten(order='F')
lista_F11G6 = normxgstsg6freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G6 = mean(lista_F1G6)
Promedio_lista_F2G6 = mean(lista_F2G6)
Promedio_lista_F3G6 = mean(lista_F3G6)
Promedio_lista_F4G6 = mean(lista_F4G6)
Promedio_lista_F5G6 = mean(lista_F5G6)
Promedio_lista_F6G6 = mean(lista_F6G6)
Promedio_lista_F7G6 = mean(lista_F7G6)
Promedio_lista_F8G6 = mean(lista_F8G6)
Promedio_lista_F9G6 = mean(lista_F9G6)
Promedio_lista_F10G6 = mean(lista_F10G6)
Promedio_lista_F11G6 = mean(lista_F11G6)
#Lista de Promedios
PromediosG6 = []
PromediosG6.append(Promedio_lista_F1G6)
PromediosG6.append(Promedio_lista_F2G6)
PromediosG6.append(Promedio_lista_F3G6)
PromediosG6.append(Promedio_lista_F4G6)
PromediosG6.append(Promedio_lista_F5G6)
PromediosG6.append(Promedio_lista_F6G6)
PromediosG6.append(Promedio_lista_F7G6)
PromediosG6.append(Promedio_lista_F8G6)
PromediosG6.append(Promedio_lista_F9G6)
PromediosG6.append(Promedio_lista_F10G6)
PromediosG6.append(Promedio_lista_F11G6)

#Segmentación G7
normxgstsg7freqs1 = normxgsts[3800:4000,30557:32233]
normxgstsg7freqs2 = normxgsts[3600:3800,30557:32233]
normxgstsg7freqs3 = normxgsts[3400:3600,30557:32233]
normxgstsg7freqs4 = normxgsts[3200:3400,30557:32233]
normxgstsg7freqs5 = normxgsts[3000:3200,30557:32233]
normxgstsg7freqs6 = normxgsts[2800:3000,30557:32233]
normxgstsg7freqs7 = normxgsts[2600:2800,30557:32233]
normxgstsg7freqs8 = normxgsts[2400:2600,30557:32233]
normxgstsg7freqs9 = normxgsts[2200:2400,30557:32233]
normxgstsg7freqs10 = normxgsts[2000:2200,30557:32233]
normxgstsg7freqs11 = normxgsts[1800:2000,30557:32233]
#Transformación matrices de NumPy en vectores por columna
lista_F1G7 = normxgstsg7freqs1.flatten(order='F')
lista_F2G7 = normxgstsg7freqs2.flatten(order='F')
lista_F3G7 = normxgstsg7freqs3.flatten(order='F')
lista_F4G7 = normxgstsg7freqs4.flatten(order='F')
lista_F5G7 = normxgstsg7freqs5.flatten(order='F')
lista_F6G7 = normxgstsg7freqs6.flatten(order='F')
lista_F7G7 = normxgstsg7freqs7.flatten(order='F')
lista_F8G7 = normxgstsg7freqs8.flatten(order='F')
lista_F9G7 = normxgstsg7freqs9.flatten(order='F')
lista_F10G7 = normxgstsg7freqs10.flatten(order='F')
lista_F11G7 = normxgstsg7freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G7 = mean(lista_F1G7)
Promedio_lista_F2G7 = mean(lista_F2G7)
Promedio_lista_F3G7 = mean(lista_F3G7)
Promedio_lista_F4G7 = mean(lista_F4G7)
Promedio_lista_F5G7 = mean(lista_F5G7)
Promedio_lista_F6G7 = mean(lista_F6G7)
Promedio_lista_F7G7 = mean(lista_F7G7)
Promedio_lista_F8G7 = mean(lista_F8G7)
Promedio_lista_F9G7 = mean(lista_F9G7)
Promedio_lista_F10G7 = mean(lista_F10G7)
Promedio_lista_F11G7 = mean(lista_F11G7)
#Lista de Promedios
PromediosG7 = []
PromediosG7.append(Promedio_lista_F1G7)
PromediosG7.append(Promedio_lista_F2G7)
PromediosG7.append(Promedio_lista_F3G7)
PromediosG7.append(Promedio_lista_F4G7)
PromediosG7.append(Promedio_lista_F5G7)
PromediosG7.append(Promedio_lista_F6G7)
PromediosG7.append(Promedio_lista_F7G7)
PromediosG7.append(Promedio_lista_F8G7)
PromediosG7.append(Promedio_lista_F9G7)
PromediosG7.append(Promedio_lista_F10G7)
PromediosG7.append(Promedio_lista_F11G7)

#Segmentación G8
normxgstsg8freqs1 = normxgsts[3800:4000,34108:35929]
normxgstsg8freqs2 = normxgsts[3600:3800,34108:35929]
normxgstsg8freqs3 = normxgsts[3400:3600,34108:35929]
normxgstsg8freqs4 = normxgsts[3200:3400,34108:35929]
normxgstsg8freqs5 = normxgsts[3000:3200,34108:35929]
normxgstsg8freqs6 = normxgsts[2800:3000,34108:35929]
normxgstsg8freqs7 = normxgsts[2600:2800,34108:35929]
normxgstsg8freqs8 = normxgsts[2400:2600,34108:35929]
normxgstsg8freqs9 = normxgsts[2200:2400,34108:35929]
normxgstsg8freqs10 = normxgsts[2000:2200,34108:35929]
normxgstsg8freqs11 = normxgsts[1800:2000,34108:35929]
#Transformación matrices de NumPy en vectores por columna
lista_F1G8 = normxgstsg8freqs1.flatten(order='F')
lista_F2G8 = normxgstsg8freqs2.flatten(order='F')
lista_F3G8 = normxgstsg8freqs3.flatten(order='F')
lista_F4G8 = normxgstsg8freqs4.flatten(order='F')
lista_F5G8 = normxgstsg8freqs5.flatten(order='F')
lista_F6G8 = normxgstsg8freqs6.flatten(order='F')
lista_F7G8 = normxgstsg8freqs7.flatten(order='F')
lista_F8G8 = normxgstsg8freqs8.flatten(order='F')
lista_F9G8 = normxgstsg8freqs9.flatten(order='F')
lista_F10G8 = normxgstsg8freqs10.flatten(order='F')
lista_F11G8 = normxgstsg8freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G8 = mean(lista_F1G8)
Promedio_lista_F2G8 = mean(lista_F2G8)
Promedio_lista_F3G8 = mean(lista_F3G8)
Promedio_lista_F4G8 = mean(lista_F4G8)
Promedio_lista_F5G8 = mean(lista_F5G8)
Promedio_lista_F6G8 = mean(lista_F6G8)
Promedio_lista_F7G8 = mean(lista_F7G8)
Promedio_lista_F8G8 = mean(lista_F8G8)
Promedio_lista_F9G8 = mean(lista_F9G8)
Promedio_lista_F10G8 = mean(lista_F10G8)
Promedio_lista_F11G8 = mean(lista_F11G8)
#Lista de Promedios
PromediosG8 = []
PromediosG8.append(Promedio_lista_F1G8)
PromediosG8.append(Promedio_lista_F2G8)
PromediosG8.append(Promedio_lista_F3G8)
PromediosG8.append(Promedio_lista_F4G8)
PromediosG8.append(Promedio_lista_F5G8)
PromediosG8.append(Promedio_lista_F6G8)
PromediosG8.append(Promedio_lista_F7G8)
PromediosG8.append(Promedio_lista_F8G8)
PromediosG8.append(Promedio_lista_F9G8)
PromediosG8.append(Promedio_lista_F10G8)
PromediosG8.append(Promedio_lista_F11G8)

#Segmentación G9
normxgstsg9freqs1 = normxgsts[3800:4000,39641:41660]
normxgstsg9freqs2 = normxgsts[3600:3800,39641:41660]
normxgstsg9freqs3 = normxgsts[3400:3600,39641:41660]
normxgstsg9freqs4 = normxgsts[3200:3400,39641:41660]
normxgstsg9freqs5 = normxgsts[3000:3200,39641:41660]
normxgstsg9freqs6 = normxgsts[2800:3000,39641:41660]
normxgstsg9freqs7 = normxgsts[2600:2800,39641:41660]
normxgstsg9freqs8 = normxgsts[2400:2600,39641:41660]
normxgstsg9freqs9 = normxgsts[2200:2400,39641:41660]
normxgstsg9freqs10 = normxgsts[2000:2200,39641:41660]
normxgstsg9freqs11 = normxgsts[1800:2000,39641:41660]
#Transformación matrices de NumPy en vectores por columna
lista_F1G9 = normxgstsg9freqs1.flatten(order='F')
lista_F2G9 = normxgstsg9freqs2.flatten(order='F')
lista_F3G9 = normxgstsg9freqs3.flatten(order='F')
lista_F4G9 = normxgstsg9freqs4.flatten(order='F')
lista_F5G9 = normxgstsg9freqs5.flatten(order='F')
lista_F6G9 = normxgstsg9freqs6.flatten(order='F')
lista_F7G9 = normxgstsg9freqs7.flatten(order='F')
lista_F8G9 = normxgstsg9freqs8.flatten(order='F')
lista_F9G9 = normxgstsg9freqs9.flatten(order='F')
lista_F10G9 = normxgstsg9freqs10.flatten(order='F')
lista_F11G9 = normxgstsg9freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G9 = mean(lista_F1G9)
Promedio_lista_F2G9 = mean(lista_F2G9)
Promedio_lista_F3G9 = mean(lista_F3G9)
Promedio_lista_F4G9 = mean(lista_F4G9)
Promedio_lista_F5G9 = mean(lista_F5G9)
Promedio_lista_F6G9 = mean(lista_F6G9)
Promedio_lista_F7G9 = mean(lista_F7G9)
Promedio_lista_F8G9 = mean(lista_F8G9)
Promedio_lista_F9G9 = mean(lista_F9G9)
Promedio_lista_F10G9 = mean(lista_F10G9)
Promedio_lista_F11G9 = mean(lista_F11G9)
#Lista de Promedios
PromediosG9 = []
PromediosG9.append(Promedio_lista_F1G9)
PromediosG9.append(Promedio_lista_F2G9)
PromediosG9.append(Promedio_lista_F3G9)
PromediosG9.append(Promedio_lista_F4G9)
PromediosG9.append(Promedio_lista_F5G9)
PromediosG9.append(Promedio_lista_F6G9)
PromediosG9.append(Promedio_lista_F7G9)
PromediosG9.append(Promedio_lista_F8G9)
PromediosG9.append(Promedio_lista_F9G9)
PromediosG9.append(Promedio_lista_F10G9)
PromediosG9.append(Promedio_lista_F11G9)

#Segmentación G10
normxgstsg10freqs1 = normxgsts[3800:4000,45362:47548]
normxgstsg10freqs2 = normxgsts[3600:3800,45362:47548]
normxgstsg10freqs3 = normxgsts[3400:3600,45362:47548]
normxgstsg10freqs4 = normxgsts[3200:3400,45362:47548]
normxgstsg10freqs5 = normxgsts[3000:3200,45362:47548]
normxgstsg10freqs6 = normxgsts[2800:3000,45362:47548]
normxgstsg10freqs7 = normxgsts[2600:2800,45362:47548]
normxgstsg10freqs8 = normxgsts[2400:2600,45362:47548]
normxgstsg10freqs9 = normxgsts[2200:2400,45362:47548]
normxgstsg10freqs10 = normxgsts[2000:2200,45362:47548]
normxgstsg10freqs11 = normxgsts[1800:2000,45362:47548]
#Transformación matrices de NumPy en vectores por columna
lista_F1G10 = normxgstsg10freqs1.flatten(order='F')
lista_F2G10 = normxgstsg10freqs2.flatten(order='F')
lista_F3G10 = normxgstsg10freqs3.flatten(order='F')
lista_F4G10 = normxgstsg10freqs4.flatten(order='F')
lista_F5G10 = normxgstsg10freqs5.flatten(order='F')
lista_F6G10 = normxgstsg10freqs6.flatten(order='F')
lista_F7G10 = normxgstsg10freqs7.flatten(order='F')
lista_F8G10 = normxgstsg10freqs8.flatten(order='F')
lista_F9G10 = normxgstsg10freqs9.flatten(order='F')
lista_F10G10 = normxgstsg10freqs10.flatten(order='F')
lista_F11G10 = normxgstsg10freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G10 = mean(lista_F1G10)
Promedio_lista_F2G10 = mean(lista_F2G10)
Promedio_lista_F3G10 = mean(lista_F3G10)
Promedio_lista_F4G10 = mean(lista_F4G10)
Promedio_lista_F5G10 = mean(lista_F5G10)
Promedio_lista_F6G10 = mean(lista_F6G10)
Promedio_lista_F7G10 = mean(lista_F7G10)
Promedio_lista_F8G10 = mean(lista_F8G10)
Promedio_lista_F9G10 = mean(lista_F9G10)
Promedio_lista_F10G10 = mean(lista_F10G10)
Promedio_lista_F11G10 = mean(lista_F11G10)
#Lista de Promedios
PromediosG10 = []
PromediosG10.append(Promedio_lista_F1G10)
PromediosG10.append(Promedio_lista_F2G10)
PromediosG10.append(Promedio_lista_F3G10)
PromediosG10.append(Promedio_lista_F4G10)
PromediosG10.append(Promedio_lista_F5G10)
PromediosG10.append(Promedio_lista_F6G10)
PromediosG10.append(Promedio_lista_F7G10)
PromediosG10.append(Promedio_lista_F8G10)
PromediosG10.append(Promedio_lista_F9G10)
PromediosG10.append(Promedio_lista_F10G10)
PromediosG10.append(Promedio_lista_F11G10)

#Segmentación G11
normxgstsg11freqs1 = normxgsts[3800:4000,50807:52752]
normxgstsg11freqs2 = normxgsts[3600:3800,50807:52752]
normxgstsg11freqs3 = normxgsts[3400:3600,50807:52752]
normxgstsg11freqs4 = normxgsts[3200:3400,50807:52752]
normxgstsg11freqs5 = normxgsts[3000:3200,50807:52752]
normxgstsg11freqs6 = normxgsts[2800:3000,50807:52752]
normxgstsg11freqs7 = normxgsts[2600:2800,50807:52752]
normxgstsg11freqs8 = normxgsts[2400:2600,50807:52752]
normxgstsg11freqs9 = normxgsts[2200:2400,50807:52752]
normxgstsg11freqs10 = normxgsts[2000:2200,50807:52752]
normxgstsg11freqs11 = normxgsts[1800:2000,50807:52752]
#Transformación matrices de NumPy en vectores por columna
lista_F1G11 = normxgstsg11freqs1.flatten(order='F')
lista_F2G11 = normxgstsg11freqs2.flatten(order='F')
lista_F3G11 = normxgstsg11freqs3.flatten(order='F')
lista_F4G11 = normxgstsg11freqs4.flatten(order='F')
lista_F5G11 = normxgstsg11freqs5.flatten(order='F')
lista_F6G11 = normxgstsg11freqs6.flatten(order='F')
lista_F7G11 = normxgstsg11freqs7.flatten(order='F')
lista_F8G11 = normxgstsg11freqs8.flatten(order='F')
lista_F9G11 = normxgstsg11freqs9.flatten(order='F')
lista_F10G11 = normxgstsg11freqs10.flatten(order='F')
lista_F11G11 = normxgstsg11freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G11 = mean(lista_F1G11)
Promedio_lista_F2G11 = mean(lista_F2G11)
Promedio_lista_F3G11 = mean(lista_F3G11)
Promedio_lista_F4G11 = mean(lista_F4G11)
Promedio_lista_F5G11 = mean(lista_F5G11)
Promedio_lista_F6G11 = mean(lista_F6G11)
Promedio_lista_F7G11 = mean(lista_F7G11)
Promedio_lista_F8G11 = mean(lista_F8G11)
Promedio_lista_F9G11 = mean(lista_F9G11)
Promedio_lista_F10G11 = mean(lista_F10G11)
Promedio_lista_F11G11 = mean(lista_F11G11)
#Lista de Promedios
PromediosG11 = []
PromediosG11.append(Promedio_lista_F1G11)
PromediosG11.append(Promedio_lista_F2G11)
PromediosG11.append(Promedio_lista_F3G11)
PromediosG11.append(Promedio_lista_F4G11)
PromediosG11.append(Promedio_lista_F5G11)
PromediosG11.append(Promedio_lista_F6G11)
PromediosG11.append(Promedio_lista_F7G11)
PromediosG11.append(Promedio_lista_F8G11)
PromediosG11.append(Promedio_lista_F9G11)
PromediosG11.append(Promedio_lista_F10G11)
PromediosG11.append(Promedio_lista_F11G11)

#Segmentación G12
normxgstsg12freqs1 = normxgsts[3800:4000,56003:57709]
normxgstsg12freqs2 = normxgsts[3600:3800,56003:57709]
normxgstsg12freqs3 = normxgsts[3400:3600,56003:57709]
normxgstsg12freqs4 = normxgsts[3200:3400,56003:57709]
normxgstsg12freqs5 = normxgsts[3000:3200,56003:57709]
normxgstsg12freqs6 = normxgsts[2800:3000,56003:57709]
normxgstsg12freqs7 = normxgsts[2600:2800,56003:57709]
normxgstsg12freqs8 = normxgsts[2400:2600,56003:57709]
normxgstsg12freqs9 = normxgsts[2200:2400,56003:57709]
normxgstsg12freqs10 = normxgsts[2000:2200,56003:57709]
normxgstsg12freqs11 = normxgsts[1800:2000,56003:57709]
#Transformación matrices de NumPy en vectores por columna
lista_F1G12 = normxgstsg12freqs1.flatten(order='F')
lista_F2G12 = normxgstsg12freqs2.flatten(order='F')
lista_F3G12 = normxgstsg12freqs3.flatten(order='F')
lista_F4G12 = normxgstsg12freqs4.flatten(order='F')
lista_F5G12 = normxgstsg12freqs5.flatten(order='F')
lista_F6G12 = normxgstsg12freqs6.flatten(order='F')
lista_F7G12 = normxgstsg12freqs7.flatten(order='F')
lista_F8G12 = normxgstsg12freqs8.flatten(order='F')
lista_F9G12 = normxgstsg12freqs9.flatten(order='F')
lista_F10G12 = normxgstsg12freqs10.flatten(order='F')
lista_F11G12 = normxgstsg12freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G12 = mean(lista_F1G12)
Promedio_lista_F2G12 = mean(lista_F2G12)
Promedio_lista_F3G12 = mean(lista_F3G12)
Promedio_lista_F4G12 = mean(lista_F4G12)
Promedio_lista_F5G12 = mean(lista_F5G12)
Promedio_lista_F6G12 = mean(lista_F6G12)
Promedio_lista_F7G12 = mean(lista_F7G12)
Promedio_lista_F8G12 = mean(lista_F8G12)
Promedio_lista_F9G12 = mean(lista_F9G12)
Promedio_lista_F10G12 = mean(lista_F10G12)
Promedio_lista_F11G12 = mean(lista_F11G12)
#Lista de Promedios
PromediosG12 = []
PromediosG12.append(Promedio_lista_F1G12)
PromediosG12.append(Promedio_lista_F2G12)
PromediosG12.append(Promedio_lista_F3G12)
PromediosG12.append(Promedio_lista_F4G12)
PromediosG12.append(Promedio_lista_F5G12)
PromediosG12.append(Promedio_lista_F6G12)
PromediosG12.append(Promedio_lista_F7G12)
PromediosG12.append(Promedio_lista_F8G12)
PromediosG12.append(Promedio_lista_F9G12)
PromediosG12.append(Promedio_lista_F10G12)
PromediosG12.append(Promedio_lista_F11G12)

Promedios = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6, 
             PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]
PromediosFrecuencias = np.array(Promedios); 
PromediosFrecuencias = np.transpose(PromediosFrecuencias)

DF = pd.DataFrame(PromediosFrecuencias)
#DF.to_csv("Promedios Paciente 3 Canal 5.csv")

PromediosG1 = np.array(PromediosG1)
PromediosG1 = np.transpose(PromediosG1)

PromediosG2 = np.array(PromediosG2)
PromediosG2 = np.transpose(PromediosG2)

PromediosG3 = np.array(PromediosG3)
PromediosG3 = np.transpose(PromediosG3)

PromediosG4 = np.array(PromediosG4)
PromediosG4 = np.transpose(PromediosG4)

PromediosG5 = np.array(PromediosG5)
PromediosG5 = np.transpose(PromediosG5)

PromediosG6 = np.array(PromediosG6)
PromediosG6 = np.transpose(PromediosG6)

PromediosG7 = np.array(PromediosG7)
PromediosG7 = np.transpose(PromediosG7)

PromediosG8 = np.array(PromediosG8)
PromediosG8 = np.transpose(PromediosG8)

PromediosG9 = np.array(PromediosG9)
PromediosG9 = np.transpose(PromediosG9)

PromediosG10 = np.array(PromediosG10)
PromediosG10 = np.transpose(PromediosG10)

PromediosG11 = np.array(PromediosG11)
PromediosG11 = np.transpose(PromediosG11)

PromediosG12 = np.array(PromediosG12)
PromediosG12 = np.transpose(PromediosG12)

data = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6,
        PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]

#Secuencia Consecutiva de Matrices 
SPromediosG1 = list(itertools.chain(PromediosG1, PromediosG7))
SPromediosG1 = np.array(SPromediosG1)
SPromediosG2 = list(itertools.chain(PromediosG2, PromediosG8))
SPromediosG2 = np.array(SPromediosG2)
SPromediosG3 = list(itertools.chain(PromediosG3, PromediosG9))
SPromediosG3 = np.array(SPromediosG3)
SPromediosG4 = list(itertools.chain(PromediosG4, PromediosG10))
SPromediosG4 = np.array(SPromediosG4)
SPromediosG5 = list(itertools.chain(PromediosG5, PromediosG11))
SPromediosG5 = np.array(SPromediosG5)
SPromediosG6 = list(itertools.chain(PromediosG6, PromediosG12))
SPromediosG6 = np.array(SPromediosG6)
SPromedios = [SPromediosG1, SPromediosG2, SPromediosG3, SPromediosG4, SPromediosG5, SPromediosG6]

SPromediosU0 = list(itertools.chain(SPromedios[0], SPromedios[1]))
SPromediosU1 = list(itertools.chain(SPromediosU0, SPromedios[2]))
SPromediosU2 = list(itertools.chain(SPromediosU1, SPromedios[3]))
SPromediosU3 = list(itertools.chain(SPromediosU2, SPromedios[4]))
SPromediosU4 = list(itertools.chain(SPromediosU3, SPromedios[5]))
SPromediosU = np.array(SPromediosU4)

#Histograma
plt.figure(figsize =(16, 10))
plt.hist(SPromediosU)
plt.grid()
plt.title('Histograma de la Secuencia Consecutiva de Promedios')
plt.xlabel('Valor de la variable')
plt.ylabel('Conteo')
plt.savefig('Histograma Promedios Gesto 1 P3C6.png', dpi = 201)
plt.show()


#scipy stats Wilcoxon 
WCresultados = []
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[1], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[2], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)


stat, p_value = wilcoxon(SPromedios[3], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[4], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[5], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)

WCresultados = np.array(WCresultados)
WCresultados = np.reshape(WCresultados,[6,6])

DF = pd.DataFrame(WCresultados) 
DF.to_csv("Wilcoxon Resultados entre gestos distintos P3C5.csv")

emg_funcional = np.array(emgpacientes03[2])
#Lectura archivo 
samplerate = 1e3 #Frecuencia de muestreo
length = emg_funcional.shape[0] / samplerate #Duración total Electromiografía en segundos
#print("Duración Total = {length}s")

#Parametros CWT 
num = 4000 #Número de sampleo generado
wavelet = 'cmor1.0-1.5' # Bugs Fixed https://pywavelets.readthedocs.io/en/latest/release.1.1.0.html 
maxesc = np.log2(1500)
widths = np.logspace(0,maxesc, num=num, base=2,dtype='int')
#print("Escalas Usadas: ", widths)
dt = 1/samplerate # diferencia de intervalos de tiempo
frequencies = pywt.scale2frequency(wavelet, widths) / dt # Obtención de frecuencias correspondientes a escalas Modificar aquí precisi´´on
#print("Freqs asociadas a Escalas: ", frequencies)
# Calculo transformación continua de ondículas de los anchos de matriz numpy
wavelet_coeffs, freqs = pywt.cwt(emg_funcional, widths, wavelet=wavelet, sampling_period=samplerate, method='fft')
#print("Tamaño de wavelet transform: ", wavelet_coeffs.shape)
#print("Freqs asociadas a wavelet transform: ", freqs)

#Valores Absolutos de la CWT
abscwt = abs(wavelet_coeffs) # Devolución de ABS de los números complejos es la magnitud
gesto_1 = abscwt[:,763:2242]
gesto_2 = abscwt[:,4045:5628]
gesto_3 = abscwt[:,9070:11027]
gesto_4 = abscwt[:,14568:16474]
gesto_5 = abscwt[:,19795:22041]
gesto_6 = abscwt[:,25667:27748]
gesto_7 = abscwt[:,30558:32232]
gesto_8 = abscwt[:,34109:35928] 
gesto_9 = abscwt[:,39642:41659] 
gesto_10 = abscwt[:,45363:47547] 
gesto_11 = abscwt[:,50808:52751] 
gesto_12 = abscwt[:,56004:57708]

normxgsts = abscwt
#División de segmentos desde 0 a 57,69 Hz

#Segmentación 1
normxgstsg1freqs1 = normxgsts[3800:4000,762:2243]
normxgstsg1freqs2 = normxgsts[3600:3800,762:2243]
normxgstsg1freqs3 = normxgsts[3400:3600,762:2243]
normxgstsg1freqs4 = normxgsts[3200:3400,762:2243]
normxgstsg1freqs5 = normxgsts[3000:3200,762:2243]
normxgstsg1freqs6 = normxgsts[2800:3000,762:2243]
normxgstsg1freqs7 = normxgsts[2600:2800,762:2243]
normxgstsg1freqs8 = normxgsts[2400:2600,762:2243]
normxgstsg1freqs9 = normxgsts[2200:2400,762:2243]
normxgstsg1freqs10 = normxgsts[2000:2200,762:2243]
normxgstsg1freqs11 = normxgsts[1800:2000,762:2243]
#Transformación matrices de NumPy en vectores por columna
lista_F1G1 = normxgstsg1freqs1.flatten(order='F')
lista_F2G1 = normxgstsg1freqs2.flatten(order='F')
lista_F3G1 = normxgstsg1freqs3.flatten(order='F')
lista_F4G1 = normxgstsg1freqs4.flatten(order='F')
lista_F5G1 = normxgstsg1freqs5.flatten(order='F')
lista_F6G1 = normxgstsg1freqs6.flatten(order='F')
lista_F7G1 = normxgstsg1freqs7.flatten(order='F')
lista_F8G1 = normxgstsg1freqs8.flatten(order='F')
lista_F9G1 = normxgstsg1freqs9.flatten(order='F')
lista_F10G1 = normxgstsg1freqs10.flatten(order='F')
lista_F11G1 = normxgstsg1freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G1 = mean(lista_F1G1)
Promedio_lista_F2G1 = mean(lista_F2G1)
Promedio_lista_F3G1 = mean(lista_F3G1)
Promedio_lista_F4G1 = mean(lista_F4G1)
Promedio_lista_F5G1 = mean(lista_F5G1)
Promedio_lista_F6G1 = mean(lista_F6G1)
Promedio_lista_F7G1 = mean(lista_F7G1)
Promedio_lista_F8G1 = mean(lista_F8G1)
Promedio_lista_F9G1 = mean(lista_F9G1)
Promedio_lista_F10G1 = mean(lista_F10G1)
Promedio_lista_F11G1 = mean(lista_F11G1)
PromediosG1 = []
PromediosG1.append(Promedio_lista_F1G1)
PromediosG1.append(Promedio_lista_F2G1)
PromediosG1.append(Promedio_lista_F3G1)
PromediosG1.append(Promedio_lista_F4G1)
PromediosG1.append(Promedio_lista_F5G1)
PromediosG1.append(Promedio_lista_F6G1)
PromediosG1.append(Promedio_lista_F7G1)
PromediosG1.append(Promedio_lista_F8G1)
PromediosG1.append(Promedio_lista_F9G1)
PromediosG1.append(Promedio_lista_F10G1)
PromediosG1.append(Promedio_lista_F11G1)


#Segmentación G2
normxgstsg2freqs1 = normxgsts[3800:4000,4044:5629]
normxgstsg2freqs2 = normxgsts[3600:3800,4044:5629]
normxgstsg2freqs3 = normxgsts[3400:3600,4044:5629]
normxgstsg2freqs4 = normxgsts[3200:3400,4044:5629]
normxgstsg2freqs5 = normxgsts[3000:3200,4044:5629]
normxgstsg2freqs6 = normxgsts[2800:3000,4044:5629]
normxgstsg2freqs7 = normxgsts[2600:2800,4044:5629]
normxgstsg2freqs8 = normxgsts[2400:2600,4044:5629]
normxgstsg2freqs9 = normxgsts[2200:2400,4044:5629]
normxgstsg2freqs10 = normxgsts[2000:2200,4044:5629]
normxgstsg2freqs11 = normxgsts[1800:2000,4044:5629]
#Transformación matrices de NumPy en vectores por columna
lista_F1G2 = normxgstsg2freqs1.flatten(order='F')
lista_F2G2 = normxgstsg2freqs2.flatten(order='F')
lista_F3G2 = normxgstsg2freqs3.flatten(order='F')
lista_F4G2 = normxgstsg2freqs4.flatten(order='F')
lista_F5G2 = normxgstsg2freqs5.flatten(order='F')
lista_F6G2 = normxgstsg2freqs6.flatten(order='F')
lista_F7G2 = normxgstsg2freqs7.flatten(order='F')
lista_F8G2 = normxgstsg2freqs8.flatten(order='F')
lista_F9G2 = normxgstsg2freqs9.flatten(order='F')
lista_F10G2 = normxgstsg2freqs10.flatten(order='F')
lista_F11G2 = normxgstsg2freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G2 = mean(lista_F1G2)
Promedio_lista_F2G2 = mean(lista_F2G2)
Promedio_lista_F3G2 = mean(lista_F3G2)
Promedio_lista_F4G2 = mean(lista_F4G2)
Promedio_lista_F5G2 = mean(lista_F5G2)
Promedio_lista_F6G2 = mean(lista_F6G2)
Promedio_lista_F7G2 = mean(lista_F7G2)
Promedio_lista_F8G2 = mean(lista_F8G2)
Promedio_lista_F9G2 = mean(lista_F9G2)
Promedio_lista_F10G2 = mean(lista_F10G2)
Promedio_lista_F11G2 = mean(lista_F11G2)
#Lista de Promedios
PromediosG2 = []
PromediosG2.append(Promedio_lista_F1G2)
PromediosG2.append(Promedio_lista_F2G2)
PromediosG2.append(Promedio_lista_F3G2)
PromediosG2.append(Promedio_lista_F4G2)
PromediosG2.append(Promedio_lista_F5G2)
PromediosG2.append(Promedio_lista_F6G2)
PromediosG2.append(Promedio_lista_F7G2)
PromediosG2.append(Promedio_lista_F8G2)
PromediosG2.append(Promedio_lista_F9G2)
PromediosG2.append(Promedio_lista_F10G2)
PromediosG2.append(Promedio_lista_F11G2)
                               

#Segmentación G3
normxgstsg3freqs1 = normxgsts[3800:4000,9069:11028]
normxgstsg3freqs2 = normxgsts[3600:3800,9069:11028]
normxgstsg3freqs3 = normxgsts[3400:3600,9069:11028]
normxgstsg3freqs4 = normxgsts[3200:3400,9069:11028]
normxgstsg3freqs5 = normxgsts[3000:3200,9069:11028]
normxgstsg3freqs6 = normxgsts[2800:3000,9069:11028]
normxgstsg3freqs7 = normxgsts[2600:2800,9069:11028]
normxgstsg3freqs8 = normxgsts[2400:2600,9069:11028]
normxgstsg3freqs9 = normxgsts[2200:2400,9069:11028]
normxgstsg3freqs10 = normxgsts[2000:2200,9069:11028]
normxgstsg3freqs11 = normxgsts[1800:2000,9069:11028]
#Transformación matrices de NumPy en vectores por columna
lista_F1G3 = normxgstsg3freqs1.flatten(order='F')
lista_F2G3 = normxgstsg3freqs2.flatten(order='F')
lista_F3G3 = normxgstsg3freqs3.flatten(order='F')
lista_F4G3 = normxgstsg3freqs4.flatten(order='F')
lista_F5G3 = normxgstsg3freqs5.flatten(order='F')
lista_F6G3 = normxgstsg3freqs6.flatten(order='F')
lista_F7G3 = normxgstsg3freqs7.flatten(order='F')
lista_F8G3 = normxgstsg3freqs8.flatten(order='F')
lista_F9G3 = normxgstsg3freqs9.flatten(order='F')
lista_F10G3 = normxgstsg3freqs10.flatten(order='F')
lista_F11G3 = normxgstsg3freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G3 = mean(lista_F1G3)
Promedio_lista_F2G3 = mean(lista_F2G3)
Promedio_lista_F3G3 = mean(lista_F3G3)
Promedio_lista_F4G3 = mean(lista_F4G3)
Promedio_lista_F5G3 = mean(lista_F5G3)
Promedio_lista_F6G3 = mean(lista_F6G3)
Promedio_lista_F7G3 = mean(lista_F7G3)
Promedio_lista_F8G3 = mean(lista_F8G3)
Promedio_lista_F9G3 = mean(lista_F9G3)
Promedio_lista_F10G3 = mean(lista_F10G3)
Promedio_lista_F11G3 = mean(lista_F11G3)
#Lista de Promedios
PromediosG3 = []
PromediosG3.append(Promedio_lista_F1G3)
PromediosG3.append(Promedio_lista_F2G3)
PromediosG3.append(Promedio_lista_F3G3)
PromediosG3.append(Promedio_lista_F4G3)
PromediosG3.append(Promedio_lista_F5G3)
PromediosG3.append(Promedio_lista_F6G3)
PromediosG3.append(Promedio_lista_F7G3)
PromediosG3.append(Promedio_lista_F8G3)
PromediosG3.append(Promedio_lista_F9G3)
PromediosG3.append(Promedio_lista_F10G3)
PromediosG3.append(Promedio_lista_F11G3)   

#Segmentación G4
normxgstsg4freqs1 = normxgsts[3800:4000,14567:16475]
normxgstsg4freqs2 = normxgsts[3600:3800,14567:16475]
normxgstsg4freqs3 = normxgsts[3400:3600,14567:16475]
normxgstsg4freqs4 = normxgsts[3200:3400,14567:16475]
normxgstsg4freqs5 = normxgsts[3000:3200,14567:16475]
normxgstsg4freqs6 = normxgsts[2800:3000,14567:16475]
normxgstsg4freqs7 = normxgsts[2600:2800,14567:16475]
normxgstsg4freqs8 = normxgsts[2400:2600,14567:16475]
normxgstsg4freqs9 = normxgsts[2200:2400,14567:16475]
normxgstsg4freqs10 = normxgsts[2000:2200,14567:16475]
normxgstsg4freqs11 = normxgsts[1800:2000,14567:16475]
#Transformación matrices de NumPy en vectores por columna
lista_F1G4 = normxgstsg4freqs1.flatten(order='F')
lista_F2G4 = normxgstsg4freqs2.flatten(order='F')
lista_F3G4 = normxgstsg4freqs3.flatten(order='F')
lista_F4G4 = normxgstsg4freqs4.flatten(order='F')
lista_F5G4 = normxgstsg4freqs5.flatten(order='F')
lista_F6G4 = normxgstsg4freqs6.flatten(order='F')
lista_F7G4 = normxgstsg4freqs7.flatten(order='F')
lista_F8G4 = normxgstsg4freqs8.flatten(order='F')
lista_F9G4 = normxgstsg4freqs9.flatten(order='F')
lista_F10G4 = normxgstsg4freqs10.flatten(order='F')
lista_F11G4 = normxgstsg4freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G4 = mean(lista_F1G4)
Promedio_lista_F2G4 = mean(lista_F2G4)
Promedio_lista_F3G4 = mean(lista_F3G4)
Promedio_lista_F4G4 = mean(lista_F4G4)
Promedio_lista_F5G4 = mean(lista_F5G4)
Promedio_lista_F6G4 = mean(lista_F6G4)
Promedio_lista_F7G4 = mean(lista_F7G4)
Promedio_lista_F8G4 = mean(lista_F8G4)
Promedio_lista_F9G4 = mean(lista_F9G4)
Promedio_lista_F10G4 = mean(lista_F10G4)
Promedio_lista_F11G4 = mean(lista_F11G4)
#Lista de Promedios
PromediosG4 = []
PromediosG4.append(Promedio_lista_F1G4)
PromediosG4.append(Promedio_lista_F2G4)
PromediosG4.append(Promedio_lista_F3G4)
PromediosG4.append(Promedio_lista_F4G4)
PromediosG4.append(Promedio_lista_F5G4)
PromediosG4.append(Promedio_lista_F6G4)
PromediosG4.append(Promedio_lista_F7G4)
PromediosG4.append(Promedio_lista_F8G4)
PromediosG4.append(Promedio_lista_F9G4)
PromediosG4.append(Promedio_lista_F10G4)
PromediosG4.append(Promedio_lista_F11G4)    

#Segmentación G5
normxgstsg5freqs1 = normxgsts[3800:4000,19794:22042]
normxgstsg5freqs2 = normxgsts[3600:3800,19794:22042]
normxgstsg5freqs3 = normxgsts[3400:3600,19794:22042]
normxgstsg5freqs4 = normxgsts[3200:3400,19794:22042]
normxgstsg5freqs5 = normxgsts[3000:3200,19794:22042]
normxgstsg5freqs6 = normxgsts[2800:3000,19794:22042]
normxgstsg5freqs7 = normxgsts[2600:2800,19794:22042]
normxgstsg5freqs8 = normxgsts[2400:2600,19794:22042]
normxgstsg5freqs9 = normxgsts[2200:2400,19794:22042]
normxgstsg5freqs10 = normxgsts[2000:2200,19794:22042]
normxgstsg5freqs11 = normxgsts[1800:2000,19794:22042]
#Transformación matrices de NumPy en vectores por columna
lista_F1G5 = normxgstsg5freqs1.flatten(order='F')
lista_F2G5 = normxgstsg5freqs2.flatten(order='F')
lista_F3G5 = normxgstsg5freqs3.flatten(order='F')
lista_F4G5 = normxgstsg5freqs4.flatten(order='F')
lista_F5G5 = normxgstsg5freqs5.flatten(order='F')
lista_F6G5 = normxgstsg5freqs6.flatten(order='F')
lista_F7G5 = normxgstsg5freqs7.flatten(order='F')
lista_F8G5 = normxgstsg5freqs8.flatten(order='F')
lista_F9G5 = normxgstsg5freqs9.flatten(order='F')
lista_F10G5 = normxgstsg5freqs10.flatten(order='F')
lista_F11G5 = normxgstsg5freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G5 = mean(lista_F1G5)
Promedio_lista_F2G5 = mean(lista_F2G5)
Promedio_lista_F3G5 = mean(lista_F3G5)
Promedio_lista_F4G5 = mean(lista_F4G5)
Promedio_lista_F5G5 = mean(lista_F5G5)
Promedio_lista_F6G5 = mean(lista_F6G5)
Promedio_lista_F7G5 = mean(lista_F7G5)
Promedio_lista_F8G5 = mean(lista_F8G5)
Promedio_lista_F9G5 = mean(lista_F9G5)
Promedio_lista_F10G5 = mean(lista_F10G5)
Promedio_lista_F11G5 = mean(lista_F11G5)
#Lista de Promedios
PromediosG5 = []
PromediosG5.append(Promedio_lista_F1G5)
PromediosG5.append(Promedio_lista_F2G5)
PromediosG5.append(Promedio_lista_F3G5)
PromediosG5.append(Promedio_lista_F4G5)
PromediosG5.append(Promedio_lista_F5G5)
PromediosG5.append(Promedio_lista_F6G5)
PromediosG5.append(Promedio_lista_F7G5)
PromediosG5.append(Promedio_lista_F8G5)
PromediosG5.append(Promedio_lista_F9G5)
PromediosG5.append(Promedio_lista_F10G5)
PromediosG5.append(Promedio_lista_F11G5)

#Segmentación G6
normxgstsg6freqs1 = normxgsts[3800:4000,25666:27749]
normxgstsg6freqs2 = normxgsts[3600:3800,25666:27749]
normxgstsg6freqs3 = normxgsts[3400:3600,25666:27749]
normxgstsg6freqs4 = normxgsts[3200:3400,25666:27749]
normxgstsg6freqs5 = normxgsts[3000:3200,25666:27749]
normxgstsg6freqs6 = normxgsts[2800:3000,25666:27749]
normxgstsg6freqs7 = normxgsts[2600:2800,25666:27749]
normxgstsg6freqs8 = normxgsts[2400:2600,25666:27749]
normxgstsg6freqs9 = normxgsts[2200:2400,25666:27749]
normxgstsg6freqs10 = normxgsts[2000:2200,25666:27749]
normxgstsg6freqs11 = normxgsts[1800:2000,25666:27749]
#Transformación matrices de NumPy en vectores por columna
lista_F1G6 = normxgstsg6freqs1.flatten(order='F')
lista_F2G6 = normxgstsg6freqs2.flatten(order='F')
lista_F3G6 = normxgstsg6freqs3.flatten(order='F')
lista_F4G6 = normxgstsg6freqs4.flatten(order='F')
lista_F5G6 = normxgstsg6freqs5.flatten(order='F')
lista_F6G6 = normxgstsg6freqs6.flatten(order='F')
lista_F7G6 = normxgstsg6freqs7.flatten(order='F')
lista_F8G6 = normxgstsg6freqs8.flatten(order='F')
lista_F9G6 = normxgstsg6freqs9.flatten(order='F')
lista_F10G6 = normxgstsg6freqs10.flatten(order='F')
lista_F11G6 = normxgstsg6freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G6 = mean(lista_F1G6)
Promedio_lista_F2G6 = mean(lista_F2G6)
Promedio_lista_F3G6 = mean(lista_F3G6)
Promedio_lista_F4G6 = mean(lista_F4G6)
Promedio_lista_F5G6 = mean(lista_F5G6)
Promedio_lista_F6G6 = mean(lista_F6G6)
Promedio_lista_F7G6 = mean(lista_F7G6)
Promedio_lista_F8G6 = mean(lista_F8G6)
Promedio_lista_F9G6 = mean(lista_F9G6)
Promedio_lista_F10G6 = mean(lista_F10G6)
Promedio_lista_F11G6 = mean(lista_F11G6)
#Lista de Promedios
PromediosG6 = []
PromediosG6.append(Promedio_lista_F1G6)
PromediosG6.append(Promedio_lista_F2G6)
PromediosG6.append(Promedio_lista_F3G6)
PromediosG6.append(Promedio_lista_F4G6)
PromediosG6.append(Promedio_lista_F5G6)
PromediosG6.append(Promedio_lista_F6G6)
PromediosG6.append(Promedio_lista_F7G6)
PromediosG6.append(Promedio_lista_F8G6)
PromediosG6.append(Promedio_lista_F9G6)
PromediosG6.append(Promedio_lista_F10G6)
PromediosG6.append(Promedio_lista_F11G6)

#Segmentación G7
normxgstsg7freqs1 = normxgsts[3800:4000,30557:32233]
normxgstsg7freqs2 = normxgsts[3600:3800,30557:32233]
normxgstsg7freqs3 = normxgsts[3400:3600,30557:32233]
normxgstsg7freqs4 = normxgsts[3200:3400,30557:32233]
normxgstsg7freqs5 = normxgsts[3000:3200,30557:32233]
normxgstsg7freqs6 = normxgsts[2800:3000,30557:32233]
normxgstsg7freqs7 = normxgsts[2600:2800,30557:32233]
normxgstsg7freqs8 = normxgsts[2400:2600,30557:32233]
normxgstsg7freqs9 = normxgsts[2200:2400,30557:32233]
normxgstsg7freqs10 = normxgsts[2000:2200,30557:32233]
normxgstsg7freqs11 = normxgsts[1800:2000,30557:32233]
#Transformación matrices de NumPy en vectores por columna
lista_F1G7 = normxgstsg7freqs1.flatten(order='F')
lista_F2G7 = normxgstsg7freqs2.flatten(order='F')
lista_F3G7 = normxgstsg7freqs3.flatten(order='F')
lista_F4G7 = normxgstsg7freqs4.flatten(order='F')
lista_F5G7 = normxgstsg7freqs5.flatten(order='F')
lista_F6G7 = normxgstsg7freqs6.flatten(order='F')
lista_F7G7 = normxgstsg7freqs7.flatten(order='F')
lista_F8G7 = normxgstsg7freqs8.flatten(order='F')
lista_F9G7 = normxgstsg7freqs9.flatten(order='F')
lista_F10G7 = normxgstsg7freqs10.flatten(order='F')
lista_F11G7 = normxgstsg7freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G7 = mean(lista_F1G7)
Promedio_lista_F2G7 = mean(lista_F2G7)
Promedio_lista_F3G7 = mean(lista_F3G7)
Promedio_lista_F4G7 = mean(lista_F4G7)
Promedio_lista_F5G7 = mean(lista_F5G7)
Promedio_lista_F6G7 = mean(lista_F6G7)
Promedio_lista_F7G7 = mean(lista_F7G7)
Promedio_lista_F8G7 = mean(lista_F8G7)
Promedio_lista_F9G7 = mean(lista_F9G7)
Promedio_lista_F10G7 = mean(lista_F10G7)
Promedio_lista_F11G7 = mean(lista_F11G7)
#Lista de Promedios
PromediosG7 = []
PromediosG7.append(Promedio_lista_F1G7)
PromediosG7.append(Promedio_lista_F2G7)
PromediosG7.append(Promedio_lista_F3G7)
PromediosG7.append(Promedio_lista_F4G7)
PromediosG7.append(Promedio_lista_F5G7)
PromediosG7.append(Promedio_lista_F6G7)
PromediosG7.append(Promedio_lista_F7G7)
PromediosG7.append(Promedio_lista_F8G7)
PromediosG7.append(Promedio_lista_F9G7)
PromediosG7.append(Promedio_lista_F10G7)
PromediosG7.append(Promedio_lista_F11G7)

#Segmentación G8
normxgstsg8freqs1 = normxgsts[3800:4000,34108:35929]
normxgstsg8freqs2 = normxgsts[3600:3800,34108:35929]
normxgstsg8freqs3 = normxgsts[3400:3600,34108:35929]
normxgstsg8freqs4 = normxgsts[3200:3400,34108:35929]
normxgstsg8freqs5 = normxgsts[3000:3200,34108:35929]
normxgstsg8freqs6 = normxgsts[2800:3000,34108:35929]
normxgstsg8freqs7 = normxgsts[2600:2800,34108:35929]
normxgstsg8freqs8 = normxgsts[2400:2600,34108:35929]
normxgstsg8freqs9 = normxgsts[2200:2400,34108:35929]
normxgstsg8freqs10 = normxgsts[2000:2200,34108:35929]
normxgstsg8freqs11 = normxgsts[1800:2000,34108:35929]
#Transformación matrices de NumPy en vectores por columna
lista_F1G8 = normxgstsg8freqs1.flatten(order='F')
lista_F2G8 = normxgstsg8freqs2.flatten(order='F')
lista_F3G8 = normxgstsg8freqs3.flatten(order='F')
lista_F4G8 = normxgstsg8freqs4.flatten(order='F')
lista_F5G8 = normxgstsg8freqs5.flatten(order='F')
lista_F6G8 = normxgstsg8freqs6.flatten(order='F')
lista_F7G8 = normxgstsg8freqs7.flatten(order='F')
lista_F8G8 = normxgstsg8freqs8.flatten(order='F')
lista_F9G8 = normxgstsg8freqs9.flatten(order='F')
lista_F10G8 = normxgstsg8freqs10.flatten(order='F')
lista_F11G8 = normxgstsg8freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G8 = mean(lista_F1G8)
Promedio_lista_F2G8 = mean(lista_F2G8)
Promedio_lista_F3G8 = mean(lista_F3G8)
Promedio_lista_F4G8 = mean(lista_F4G8)
Promedio_lista_F5G8 = mean(lista_F5G8)
Promedio_lista_F6G8 = mean(lista_F6G8)
Promedio_lista_F7G8 = mean(lista_F7G8)
Promedio_lista_F8G8 = mean(lista_F8G8)
Promedio_lista_F9G8 = mean(lista_F9G8)
Promedio_lista_F10G8 = mean(lista_F10G8)
Promedio_lista_F11G8 = mean(lista_F11G8)
#Lista de Promedios
PromediosG8 = []
PromediosG8.append(Promedio_lista_F1G8)
PromediosG8.append(Promedio_lista_F2G8)
PromediosG8.append(Promedio_lista_F3G8)
PromediosG8.append(Promedio_lista_F4G8)
PromediosG8.append(Promedio_lista_F5G8)
PromediosG8.append(Promedio_lista_F6G8)
PromediosG8.append(Promedio_lista_F7G8)
PromediosG8.append(Promedio_lista_F8G8)
PromediosG8.append(Promedio_lista_F9G8)
PromediosG8.append(Promedio_lista_F10G8)
PromediosG8.append(Promedio_lista_F11G8)

#Segmentación G9
normxgstsg9freqs1 = normxgsts[3800:4000,39641:41660]
normxgstsg9freqs2 = normxgsts[3600:3800,39641:41660]
normxgstsg9freqs3 = normxgsts[3400:3600,39641:41660]
normxgstsg9freqs4 = normxgsts[3200:3400,39641:41660]
normxgstsg9freqs5 = normxgsts[3000:3200,39641:41660]
normxgstsg9freqs6 = normxgsts[2800:3000,39641:41660]
normxgstsg9freqs7 = normxgsts[2600:2800,39641:41660]
normxgstsg9freqs8 = normxgsts[2400:2600,39641:41660]
normxgstsg9freqs9 = normxgsts[2200:2400,39641:41660]
normxgstsg9freqs10 = normxgsts[2000:2200,39641:41660]
normxgstsg9freqs11 = normxgsts[1800:2000,39641:41660]
#Transformación matrices de NumPy en vectores por columna
lista_F1G9 = normxgstsg9freqs1.flatten(order='F')
lista_F2G9 = normxgstsg9freqs2.flatten(order='F')
lista_F3G9 = normxgstsg9freqs3.flatten(order='F')
lista_F4G9 = normxgstsg9freqs4.flatten(order='F')
lista_F5G9 = normxgstsg9freqs5.flatten(order='F')
lista_F6G9 = normxgstsg9freqs6.flatten(order='F')
lista_F7G9 = normxgstsg9freqs7.flatten(order='F')
lista_F8G9 = normxgstsg9freqs8.flatten(order='F')
lista_F9G9 = normxgstsg9freqs9.flatten(order='F')
lista_F10G9 = normxgstsg9freqs10.flatten(order='F')
lista_F11G9 = normxgstsg9freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G9 = mean(lista_F1G9)
Promedio_lista_F2G9 = mean(lista_F2G9)
Promedio_lista_F3G9 = mean(lista_F3G9)
Promedio_lista_F4G9 = mean(lista_F4G9)
Promedio_lista_F5G9 = mean(lista_F5G9)
Promedio_lista_F6G9 = mean(lista_F6G9)
Promedio_lista_F7G9 = mean(lista_F7G9)
Promedio_lista_F8G9 = mean(lista_F8G9)
Promedio_lista_F9G9 = mean(lista_F9G9)
Promedio_lista_F10G9 = mean(lista_F10G9)
Promedio_lista_F11G9 = mean(lista_F11G9)
#Lista de Promedios
PromediosG9 = []
PromediosG9.append(Promedio_lista_F1G9)
PromediosG9.append(Promedio_lista_F2G9)
PromediosG9.append(Promedio_lista_F3G9)
PromediosG9.append(Promedio_lista_F4G9)
PromediosG9.append(Promedio_lista_F5G9)
PromediosG9.append(Promedio_lista_F6G9)
PromediosG9.append(Promedio_lista_F7G9)
PromediosG9.append(Promedio_lista_F8G9)
PromediosG9.append(Promedio_lista_F9G9)
PromediosG9.append(Promedio_lista_F10G9)
PromediosG9.append(Promedio_lista_F11G9)

#Segmentación G10
normxgstsg10freqs1 = normxgsts[3800:4000,45362:47548]
normxgstsg10freqs2 = normxgsts[3600:3800,45362:47548]
normxgstsg10freqs3 = normxgsts[3400:3600,45362:47548]
normxgstsg10freqs4 = normxgsts[3200:3400,45362:47548]
normxgstsg10freqs5 = normxgsts[3000:3200,45362:47548]
normxgstsg10freqs6 = normxgsts[2800:3000,45362:47548]
normxgstsg10freqs7 = normxgsts[2600:2800,45362:47548]
normxgstsg10freqs8 = normxgsts[2400:2600,45362:47548]
normxgstsg10freqs9 = normxgsts[2200:2400,45362:47548]
normxgstsg10freqs10 = normxgsts[2000:2200,45362:47548]
normxgstsg10freqs11 = normxgsts[1800:2000,45362:47548]
#Transformación matrices de NumPy en vectores por columna
lista_F1G10 = normxgstsg10freqs1.flatten(order='F')
lista_F2G10 = normxgstsg10freqs2.flatten(order='F')
lista_F3G10 = normxgstsg10freqs3.flatten(order='F')
lista_F4G10 = normxgstsg10freqs4.flatten(order='F')
lista_F5G10 = normxgstsg10freqs5.flatten(order='F')
lista_F6G10 = normxgstsg10freqs6.flatten(order='F')
lista_F7G10 = normxgstsg10freqs7.flatten(order='F')
lista_F8G10 = normxgstsg10freqs8.flatten(order='F')
lista_F9G10 = normxgstsg10freqs9.flatten(order='F')
lista_F10G10 = normxgstsg10freqs10.flatten(order='F')
lista_F11G10 = normxgstsg10freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G10 = mean(lista_F1G10)
Promedio_lista_F2G10 = mean(lista_F2G10)
Promedio_lista_F3G10 = mean(lista_F3G10)
Promedio_lista_F4G10 = mean(lista_F4G10)
Promedio_lista_F5G10 = mean(lista_F5G10)
Promedio_lista_F6G10 = mean(lista_F6G10)
Promedio_lista_F7G10 = mean(lista_F7G10)
Promedio_lista_F8G10 = mean(lista_F8G10)
Promedio_lista_F9G10 = mean(lista_F9G10)
Promedio_lista_F10G10 = mean(lista_F10G10)
Promedio_lista_F11G10 = mean(lista_F11G10)
#Lista de Promedios
PromediosG10 = []
PromediosG10.append(Promedio_lista_F1G10)
PromediosG10.append(Promedio_lista_F2G10)
PromediosG10.append(Promedio_lista_F3G10)
PromediosG10.append(Promedio_lista_F4G10)
PromediosG10.append(Promedio_lista_F5G10)
PromediosG10.append(Promedio_lista_F6G10)
PromediosG10.append(Promedio_lista_F7G10)
PromediosG10.append(Promedio_lista_F8G10)
PromediosG10.append(Promedio_lista_F9G10)
PromediosG10.append(Promedio_lista_F10G10)
PromediosG10.append(Promedio_lista_F11G10)

#Segmentación G11
normxgstsg11freqs1 = normxgsts[3800:4000,50807:52752]
normxgstsg11freqs2 = normxgsts[3600:3800,50807:52752]
normxgstsg11freqs3 = normxgsts[3400:3600,50807:52752]
normxgstsg11freqs4 = normxgsts[3200:3400,50807:52752]
normxgstsg11freqs5 = normxgsts[3000:3200,50807:52752]
normxgstsg11freqs6 = normxgsts[2800:3000,50807:52752]
normxgstsg11freqs7 = normxgsts[2600:2800,50807:52752]
normxgstsg11freqs8 = normxgsts[2400:2600,50807:52752]
normxgstsg11freqs9 = normxgsts[2200:2400,50807:52752]
normxgstsg11freqs10 = normxgsts[2000:2200,50807:52752]
normxgstsg11freqs11 = normxgsts[1800:2000,50807:52752]
#Transformación matrices de NumPy en vectores por columna
lista_F1G11 = normxgstsg11freqs1.flatten(order='F')
lista_F2G11 = normxgstsg11freqs2.flatten(order='F')
lista_F3G11 = normxgstsg11freqs3.flatten(order='F')
lista_F4G11 = normxgstsg11freqs4.flatten(order='F')
lista_F5G11 = normxgstsg11freqs5.flatten(order='F')
lista_F6G11 = normxgstsg11freqs6.flatten(order='F')
lista_F7G11 = normxgstsg11freqs7.flatten(order='F')
lista_F8G11 = normxgstsg11freqs8.flatten(order='F')
lista_F9G11 = normxgstsg11freqs9.flatten(order='F')
lista_F10G11 = normxgstsg11freqs10.flatten(order='F')
lista_F11G11 = normxgstsg11freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G11 = mean(lista_F1G11)
Promedio_lista_F2G11 = mean(lista_F2G11)
Promedio_lista_F3G11 = mean(lista_F3G11)
Promedio_lista_F4G11 = mean(lista_F4G11)
Promedio_lista_F5G11 = mean(lista_F5G11)
Promedio_lista_F6G11 = mean(lista_F6G11)
Promedio_lista_F7G11 = mean(lista_F7G11)
Promedio_lista_F8G11 = mean(lista_F8G11)
Promedio_lista_F9G11 = mean(lista_F9G11)
Promedio_lista_F10G11 = mean(lista_F10G11)
Promedio_lista_F11G11 = mean(lista_F11G11)
#Lista de Promedios
PromediosG11 = []
PromediosG11.append(Promedio_lista_F1G11)
PromediosG11.append(Promedio_lista_F2G11)
PromediosG11.append(Promedio_lista_F3G11)
PromediosG11.append(Promedio_lista_F4G11)
PromediosG11.append(Promedio_lista_F5G11)
PromediosG11.append(Promedio_lista_F6G11)
PromediosG11.append(Promedio_lista_F7G11)
PromediosG11.append(Promedio_lista_F8G11)
PromediosG11.append(Promedio_lista_F9G11)
PromediosG11.append(Promedio_lista_F10G11)
PromediosG11.append(Promedio_lista_F11G11)

#Segmentación G12
normxgstsg12freqs1 = normxgsts[3800:4000,56003:57709]
normxgstsg12freqs2 = normxgsts[3600:3800,56003:57709]
normxgstsg12freqs3 = normxgsts[3400:3600,56003:57709]
normxgstsg12freqs4 = normxgsts[3200:3400,56003:57709]
normxgstsg12freqs5 = normxgsts[3000:3200,56003:57709]
normxgstsg12freqs6 = normxgsts[2800:3000,56003:57709]
normxgstsg12freqs7 = normxgsts[2600:2800,56003:57709]
normxgstsg12freqs8 = normxgsts[2400:2600,56003:57709]
normxgstsg12freqs9 = normxgsts[2200:2400,56003:57709]
normxgstsg12freqs10 = normxgsts[2000:2200,56003:57709]
normxgstsg12freqs11 = normxgsts[1800:2000,56003:57709]
#Transformación matrices de NumPy en vectores por columna
lista_F1G12 = normxgstsg12freqs1.flatten(order='F')
lista_F2G12 = normxgstsg12freqs2.flatten(order='F')
lista_F3G12 = normxgstsg12freqs3.flatten(order='F')
lista_F4G12 = normxgstsg12freqs4.flatten(order='F')
lista_F5G12 = normxgstsg12freqs5.flatten(order='F')
lista_F6G12 = normxgstsg12freqs6.flatten(order='F')
lista_F7G12 = normxgstsg12freqs7.flatten(order='F')
lista_F8G12 = normxgstsg12freqs8.flatten(order='F')
lista_F9G12 = normxgstsg12freqs9.flatten(order='F')
lista_F10G12 = normxgstsg12freqs10.flatten(order='F')
lista_F11G12 = normxgstsg12freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G12 = mean(lista_F1G12)
Promedio_lista_F2G12 = mean(lista_F2G12)
Promedio_lista_F3G12 = mean(lista_F3G12)
Promedio_lista_F4G12 = mean(lista_F4G12)
Promedio_lista_F5G12 = mean(lista_F5G12)
Promedio_lista_F6G12 = mean(lista_F6G12)
Promedio_lista_F7G12 = mean(lista_F7G12)
Promedio_lista_F8G12 = mean(lista_F8G12)
Promedio_lista_F9G12 = mean(lista_F9G12)
Promedio_lista_F10G12 = mean(lista_F10G12)
Promedio_lista_F11G12 = mean(lista_F11G12)
#Lista de Promedios
PromediosG12 = []
PromediosG12.append(Promedio_lista_F1G12)
PromediosG12.append(Promedio_lista_F2G12)
PromediosG12.append(Promedio_lista_F3G12)
PromediosG12.append(Promedio_lista_F4G12)
PromediosG12.append(Promedio_lista_F5G12)
PromediosG12.append(Promedio_lista_F6G12)
PromediosG12.append(Promedio_lista_F7G12)
PromediosG12.append(Promedio_lista_F8G12)
PromediosG12.append(Promedio_lista_F9G12)
PromediosG12.append(Promedio_lista_F10G12)
PromediosG12.append(Promedio_lista_F11G12)

Promedios = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6, 
             PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]
PromediosFrecuencias = np.array(Promedios); 
PromediosFrecuencias = np.transpose(PromediosFrecuencias)

DF = pd.DataFrame(PromediosFrecuencias) 
#DF.to_csv("Promedios Paciente 3 Canal 6.csv")

PromediosG1 = np.array(PromediosG1)
PromediosG1 = np.transpose(PromediosG1)

PromediosG2 = np.array(PromediosG2)
PromediosG2 = np.transpose(PromediosG2)

PromediosG3 = np.array(PromediosG3)
PromediosG3 = np.transpose(PromediosG3)

PromediosG4 = np.array(PromediosG4)
PromediosG4 = np.transpose(PromediosG4)

PromediosG5 = np.array(PromediosG5)
PromediosG5 = np.transpose(PromediosG5)

PromediosG6 = np.array(PromediosG6)
PromediosG6 = np.transpose(PromediosG6)

PromediosG7 = np.array(PromediosG7)
PromediosG7 = np.transpose(PromediosG7)

PromediosG8 = np.array(PromediosG8)
PromediosG8 = np.transpose(PromediosG8)

PromediosG9 = np.array(PromediosG9)
PromediosG9 = np.transpose(PromediosG9)

PromediosG10 = np.array(PromediosG10)
PromediosG10 = np.transpose(PromediosG10)

PromediosG11 = np.array(PromediosG11)
PromediosG11 = np.transpose(PromediosG11)

PromediosG12 = np.array(PromediosG12)
PromediosG12 = np.transpose(PromediosG12)

data = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6,
        PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]

#Secuencia Consecutiva de Matrices 
SPromediosG1 = list(itertools.chain(PromediosG1, PromediosG7))
SPromediosG1 = np.array(SPromediosG1)
SPromediosG2 = list(itertools.chain(PromediosG2, PromediosG8))
SPromediosG2 = np.array(SPromediosG2)
SPromediosG3 = list(itertools.chain(PromediosG3, PromediosG9))
SPromediosG3 = np.array(SPromediosG3)
SPromediosG4 = list(itertools.chain(PromediosG4, PromediosG10))
SPromediosG4 = np.array(SPromediosG4)
SPromediosG5 = list(itertools.chain(PromediosG5, PromediosG11))
SPromediosG5 = np.array(SPromediosG5)
SPromediosG6 = list(itertools.chain(PromediosG6, PromediosG12))
SPromediosG6 = np.array(SPromediosG6)
SPromedios = [SPromediosG1, SPromediosG2, SPromediosG3, SPromediosG4, SPromediosG5, SPromediosG6]

SPromediosU0 = list(itertools.chain(SPromedios[0], SPromedios[1]))
SPromediosU1 = list(itertools.chain(SPromediosU0, SPromedios[2]))
SPromediosU2 = list(itertools.chain(SPromediosU1, SPromedios[3]))
SPromediosU3 = list(itertools.chain(SPromediosU2, SPromedios[4]))
SPromediosU4 = list(itertools.chain(SPromediosU3, SPromedios[5]))
SPromediosU = np.array(SPromediosU4)

#Histograma
plt.figure(figsize =(16, 10))
plt.hist(SPromediosU)
plt.grid()
plt.title('Histograma de la Secuencia Consecutiva de Promedios')
plt.xlabel('Valor de la variable')
plt.ylabel('Conteo')
plt.savefig('Histograma Promedios Gesto 1 P3C6.png', dpi = 201)
plt.show()

#scipy stats Wilcoxon 
WCresultados = []
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[1], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[2], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)


stat, p_value = wilcoxon(SPromedios[3], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[4], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[5], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)

WCresultados = np.array(WCresultados)
WCresultados = np.reshape(WCresultados,[6,6])

DF = pd.DataFrame(WCresultados) 
DF.to_csv("Wilcoxon Resultados entre gestos distintos P3C6.csv")


emg_funcional = np.array(emgpacientes03[3])
#Lectura archivo 
samplerate = 1e3 #Frecuencia de muestreo
length = emg_funcional.shape[0] / samplerate #Duración total Electromiografía en segundos
#print("Duración Total = {length}s")

#Parametros CWT 
num = 4000 #Número de sampleo generado
wavelet = 'cmor1.0-1.5' # Bugs Fixed https://pywavelets.readthedocs.io/en/latest/release.1.1.0.html 
maxesc = np.log2(1500)
widths = np.logspace(0,maxesc, num=num, base=2,dtype='int')
#print("Escalas Usadas: ", widths)
dt = 1/samplerate # diferencia de intervalos de tiempo
frequencies = pywt.scale2frequency(wavelet, widths) / dt # Obtención de frecuencias correspondientes a escalas Modificar aquí precisi´´on
#print("Freqs asociadas a Escalas: ", frequencies)
# Calculo transformación continua de ondículas de los anchos de matriz numpy
wavelet_coeffs, freqs = pywt.cwt(emg_funcional, widths, wavelet=wavelet, sampling_period=samplerate, method='fft')
#print("Tamaño de wavelet transform: ", wavelet_coeffs.shape)
#print("Freqs asociadas a wavelet transform: ", freqs)

#Valores Absolutos de la CWT
abscwt = abs(wavelet_coeffs) # Devolución de ABS de los números complejos es la magnitud
gesto_1 = abscwt[:,763:2242]
gesto_2 = abscwt[:,4045:5628]
gesto_3 = abscwt[:,9070:11027]
gesto_4 = abscwt[:,14568:16474]
gesto_5 = abscwt[:,19795:22041]
gesto_6 = abscwt[:,25667:27748]
gesto_7 = abscwt[:,30558:32232]
gesto_8 = abscwt[:,34109:35928] 
gesto_9 = abscwt[:,39642:41659] 
gesto_10 = abscwt[:,45363:47547] 
gesto_11 = abscwt[:,50808:52751] 
gesto_12 = abscwt[:,56004:57708]

normxgsts = abscwt
#División de segmentos desde 0 a 57,69 Hz

#Segmentación 1
normxgstsg1freqs1 = normxgsts[3800:4000,762:2243]
normxgstsg1freqs2 = normxgsts[3600:3800,762:2243]
normxgstsg1freqs3 = normxgsts[3400:3600,762:2243]
normxgstsg1freqs4 = normxgsts[3200:3400,762:2243]
normxgstsg1freqs5 = normxgsts[3000:3200,762:2243]
normxgstsg1freqs6 = normxgsts[2800:3000,762:2243]
normxgstsg1freqs7 = normxgsts[2600:2800,762:2243]
normxgstsg1freqs8 = normxgsts[2400:2600,762:2243]
normxgstsg1freqs9 = normxgsts[2200:2400,762:2243]
normxgstsg1freqs10 = normxgsts[2000:2200,762:2243]
normxgstsg1freqs11 = normxgsts[1800:2000,762:2243]
#Transformación matrices de NumPy en vectores por columna
lista_F1G1 = normxgstsg1freqs1.flatten(order='F')
lista_F2G1 = normxgstsg1freqs2.flatten(order='F')
lista_F3G1 = normxgstsg1freqs3.flatten(order='F')
lista_F4G1 = normxgstsg1freqs4.flatten(order='F')
lista_F5G1 = normxgstsg1freqs5.flatten(order='F')
lista_F6G1 = normxgstsg1freqs6.flatten(order='F')
lista_F7G1 = normxgstsg1freqs7.flatten(order='F')
lista_F8G1 = normxgstsg1freqs8.flatten(order='F')
lista_F9G1 = normxgstsg1freqs9.flatten(order='F')
lista_F10G1 = normxgstsg1freqs10.flatten(order='F')
lista_F11G1 = normxgstsg1freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G1 = mean(lista_F1G1)
Promedio_lista_F2G1 = mean(lista_F2G1)
Promedio_lista_F3G1 = mean(lista_F3G1)
Promedio_lista_F4G1 = mean(lista_F4G1)
Promedio_lista_F5G1 = mean(lista_F5G1)
Promedio_lista_F6G1 = mean(lista_F6G1)
Promedio_lista_F7G1 = mean(lista_F7G1)
Promedio_lista_F8G1 = mean(lista_F8G1)
Promedio_lista_F9G1 = mean(lista_F9G1)
Promedio_lista_F10G1 = mean(lista_F10G1)
Promedio_lista_F11G1 = mean(lista_F11G1)
PromediosG1 = []
PromediosG1.append(Promedio_lista_F1G1)
PromediosG1.append(Promedio_lista_F2G1)
PromediosG1.append(Promedio_lista_F3G1)
PromediosG1.append(Promedio_lista_F4G1)
PromediosG1.append(Promedio_lista_F5G1)
PromediosG1.append(Promedio_lista_F6G1)
PromediosG1.append(Promedio_lista_F7G1)
PromediosG1.append(Promedio_lista_F8G1)
PromediosG1.append(Promedio_lista_F9G1)
PromediosG1.append(Promedio_lista_F10G1)
PromediosG1.append(Promedio_lista_F11G1)

#Segmentación G2
normxgstsg2freqs1 = normxgsts[3800:4000,4044:5629]
normxgstsg2freqs2 = normxgsts[3600:3800,4044:5629]
normxgstsg2freqs3 = normxgsts[3400:3600,4044:5629]
normxgstsg2freqs4 = normxgsts[3200:3400,4044:5629]
normxgstsg2freqs5 = normxgsts[3000:3200,4044:5629]
normxgstsg2freqs6 = normxgsts[2800:3000,4044:5629]
normxgstsg2freqs7 = normxgsts[2600:2800,4044:5629]
normxgstsg2freqs8 = normxgsts[2400:2600,4044:5629]
normxgstsg2freqs9 = normxgsts[2200:2400,4044:5629]
normxgstsg2freqs10 = normxgsts[2000:2200,4044:5629]
normxgstsg2freqs11 = normxgsts[1800:2000,4044:5629]
#Transformación matrices de NumPy en vectores por columna
lista_F1G2 = normxgstsg2freqs1.flatten(order='F')
lista_F2G2 = normxgstsg2freqs2.flatten(order='F')
lista_F3G2 = normxgstsg2freqs3.flatten(order='F')
lista_F4G2 = normxgstsg2freqs4.flatten(order='F')
lista_F5G2 = normxgstsg2freqs5.flatten(order='F')
lista_F6G2 = normxgstsg2freqs6.flatten(order='F')
lista_F7G2 = normxgstsg2freqs7.flatten(order='F')
lista_F8G2 = normxgstsg2freqs8.flatten(order='F')
lista_F9G2 = normxgstsg2freqs9.flatten(order='F')
lista_F10G2 = normxgstsg2freqs10.flatten(order='F')
lista_F11G2 = normxgstsg2freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G2 = mean(lista_F1G2)
Promedio_lista_F2G2 = mean(lista_F2G2)
Promedio_lista_F3G2 = mean(lista_F3G2)
Promedio_lista_F4G2 = mean(lista_F4G2)
Promedio_lista_F5G2 = mean(lista_F5G2)
Promedio_lista_F6G2 = mean(lista_F6G2)
Promedio_lista_F7G2 = mean(lista_F7G2)
Promedio_lista_F8G2 = mean(lista_F8G2)
Promedio_lista_F9G2 = mean(lista_F9G2)
Promedio_lista_F10G2 = mean(lista_F10G2)
Promedio_lista_F11G2 = mean(lista_F11G2)
#Lista de Promedios
PromediosG2 = []
PromediosG2.append(Promedio_lista_F1G2)
PromediosG2.append(Promedio_lista_F2G2)
PromediosG2.append(Promedio_lista_F3G2)
PromediosG2.append(Promedio_lista_F4G2)
PromediosG2.append(Promedio_lista_F5G2)
PromediosG2.append(Promedio_lista_F6G2)
PromediosG2.append(Promedio_lista_F7G2)
PromediosG2.append(Promedio_lista_F8G2)
PromediosG2.append(Promedio_lista_F9G2)
PromediosG2.append(Promedio_lista_F10G2)
PromediosG2.append(Promedio_lista_F11G2)
                               
#Segmentación G3
normxgstsg3freqs1 = normxgsts[3800:4000,9069:11028]
normxgstsg3freqs2 = normxgsts[3600:3800,9069:11028]
normxgstsg3freqs3 = normxgsts[3400:3600,9069:11028]
normxgstsg3freqs4 = normxgsts[3200:3400,9069:11028]
normxgstsg3freqs5 = normxgsts[3000:3200,9069:11028]
normxgstsg3freqs6 = normxgsts[2800:3000,9069:11028]
normxgstsg3freqs7 = normxgsts[2600:2800,9069:11028]
normxgstsg3freqs8 = normxgsts[2400:2600,9069:11028]
normxgstsg3freqs9 = normxgsts[2200:2400,9069:11028]
normxgstsg3freqs10 = normxgsts[2000:2200,9069:11028]
normxgstsg3freqs11 = normxgsts[1800:2000,9069:11028]
#Transformación matrices de NumPy en vectores por columna
lista_F1G3 = normxgstsg3freqs1.flatten(order='F')
lista_F2G3 = normxgstsg3freqs2.flatten(order='F')
lista_F3G3 = normxgstsg3freqs3.flatten(order='F')
lista_F4G3 = normxgstsg3freqs4.flatten(order='F')
lista_F5G3 = normxgstsg3freqs5.flatten(order='F')
lista_F6G3 = normxgstsg3freqs6.flatten(order='F')
lista_F7G3 = normxgstsg3freqs7.flatten(order='F')
lista_F8G3 = normxgstsg3freqs8.flatten(order='F')
lista_F9G3 = normxgstsg3freqs9.flatten(order='F')
lista_F10G3 = normxgstsg3freqs10.flatten(order='F')
lista_F11G3 = normxgstsg3freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G3 = mean(lista_F1G3)
Promedio_lista_F2G3 = mean(lista_F2G3)
Promedio_lista_F3G3 = mean(lista_F3G3)
Promedio_lista_F4G3 = mean(lista_F4G3)
Promedio_lista_F5G3 = mean(lista_F5G3)
Promedio_lista_F6G3 = mean(lista_F6G3)
Promedio_lista_F7G3 = mean(lista_F7G3)
Promedio_lista_F8G3 = mean(lista_F8G3)
Promedio_lista_F9G3 = mean(lista_F9G3)
Promedio_lista_F10G3 = mean(lista_F10G3)
Promedio_lista_F11G3 = mean(lista_F11G3)
#Lista de Promedios
PromediosG3 = []
PromediosG3.append(Promedio_lista_F1G3)
PromediosG3.append(Promedio_lista_F2G3)
PromediosG3.append(Promedio_lista_F3G3)
PromediosG3.append(Promedio_lista_F4G3)
PromediosG3.append(Promedio_lista_F5G3)
PromediosG3.append(Promedio_lista_F6G3)
PromediosG3.append(Promedio_lista_F7G3)
PromediosG3.append(Promedio_lista_F8G3)
PromediosG3.append(Promedio_lista_F9G3)
PromediosG3.append(Promedio_lista_F10G3)
PromediosG3.append(Promedio_lista_F11G3)   

#Segmentación G4
normxgstsg4freqs1 = normxgsts[3800:4000,14567:16475]
normxgstsg4freqs2 = normxgsts[3600:3800,14567:16475]
normxgstsg4freqs3 = normxgsts[3400:3600,14567:16475]
normxgstsg4freqs4 = normxgsts[3200:3400,14567:16475]
normxgstsg4freqs5 = normxgsts[3000:3200,14567:16475]
normxgstsg4freqs6 = normxgsts[2800:3000,14567:16475]
normxgstsg4freqs7 = normxgsts[2600:2800,14567:16475]
normxgstsg4freqs8 = normxgsts[2400:2600,14567:16475]
normxgstsg4freqs9 = normxgsts[2200:2400,14567:16475]
normxgstsg4freqs10 = normxgsts[2000:2200,14567:16475]
normxgstsg4freqs11 = normxgsts[1800:2000,14567:16475]
#Transformación matrices de NumPy en vectores por columna
lista_F1G4 = normxgstsg4freqs1.flatten(order='F')
lista_F2G4 = normxgstsg4freqs2.flatten(order='F')
lista_F3G4 = normxgstsg4freqs3.flatten(order='F')
lista_F4G4 = normxgstsg4freqs4.flatten(order='F')
lista_F5G4 = normxgstsg4freqs5.flatten(order='F')
lista_F6G4 = normxgstsg4freqs6.flatten(order='F')
lista_F7G4 = normxgstsg4freqs7.flatten(order='F')
lista_F8G4 = normxgstsg4freqs8.flatten(order='F')
lista_F9G4 = normxgstsg4freqs9.flatten(order='F')
lista_F10G4 = normxgstsg4freqs10.flatten(order='F')
lista_F11G4 = normxgstsg4freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G4 = mean(lista_F1G4)
Promedio_lista_F2G4 = mean(lista_F2G4)
Promedio_lista_F3G4 = mean(lista_F3G4)
Promedio_lista_F4G4 = mean(lista_F4G4)
Promedio_lista_F5G4 = mean(lista_F5G4)
Promedio_lista_F6G4 = mean(lista_F6G4)
Promedio_lista_F7G4 = mean(lista_F7G4)
Promedio_lista_F8G4 = mean(lista_F8G4)
Promedio_lista_F9G4 = mean(lista_F9G4)
Promedio_lista_F10G4 = mean(lista_F10G4)
Promedio_lista_F11G4 = mean(lista_F11G4)
#Lista de Promedios
PromediosG4 = []
PromediosG4.append(Promedio_lista_F1G4)
PromediosG4.append(Promedio_lista_F2G4)
PromediosG4.append(Promedio_lista_F3G4)
PromediosG4.append(Promedio_lista_F4G4)
PromediosG4.append(Promedio_lista_F5G4)
PromediosG4.append(Promedio_lista_F6G4)
PromediosG4.append(Promedio_lista_F7G4)
PromediosG4.append(Promedio_lista_F8G4)
PromediosG4.append(Promedio_lista_F9G4)
PromediosG4.append(Promedio_lista_F10G4)
PromediosG4.append(Promedio_lista_F11G4)    

#Segmentación G5
normxgstsg5freqs1 = normxgsts[3800:4000,19794:22042]
normxgstsg5freqs2 = normxgsts[3600:3800,19794:22042]
normxgstsg5freqs3 = normxgsts[3400:3600,19794:22042]
normxgstsg5freqs4 = normxgsts[3200:3400,19794:22042]
normxgstsg5freqs5 = normxgsts[3000:3200,19794:22042]
normxgstsg5freqs6 = normxgsts[2800:3000,19794:22042]
normxgstsg5freqs7 = normxgsts[2600:2800,19794:22042]
normxgstsg5freqs8 = normxgsts[2400:2600,19794:22042]
normxgstsg5freqs9 = normxgsts[2200:2400,19794:22042]
normxgstsg5freqs10 = normxgsts[2000:2200,19794:22042]
normxgstsg5freqs11 = normxgsts[1800:2000,19794:22042]
#Transformación matrices de NumPy en vectores por columna
lista_F1G5 = normxgstsg5freqs1.flatten(order='F')
lista_F2G5 = normxgstsg5freqs2.flatten(order='F')
lista_F3G5 = normxgstsg5freqs3.flatten(order='F')
lista_F4G5 = normxgstsg5freqs4.flatten(order='F')
lista_F5G5 = normxgstsg5freqs5.flatten(order='F')
lista_F6G5 = normxgstsg5freqs6.flatten(order='F')
lista_F7G5 = normxgstsg5freqs7.flatten(order='F')
lista_F8G5 = normxgstsg5freqs8.flatten(order='F')
lista_F9G5 = normxgstsg5freqs9.flatten(order='F')
lista_F10G5 = normxgstsg5freqs10.flatten(order='F')
lista_F11G5 = normxgstsg5freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G5 = mean(lista_F1G5)
Promedio_lista_F2G5 = mean(lista_F2G5)
Promedio_lista_F3G5 = mean(lista_F3G5)
Promedio_lista_F4G5 = mean(lista_F4G5)
Promedio_lista_F5G5 = mean(lista_F5G5)
Promedio_lista_F6G5 = mean(lista_F6G5)
Promedio_lista_F7G5 = mean(lista_F7G5)
Promedio_lista_F8G5 = mean(lista_F8G5)
Promedio_lista_F9G5 = mean(lista_F9G5)
Promedio_lista_F10G5 = mean(lista_F10G5)
Promedio_lista_F11G5 = mean(lista_F11G5)
#Lista de Promedios
PromediosG5 = []
PromediosG5.append(Promedio_lista_F1G5)
PromediosG5.append(Promedio_lista_F2G5)
PromediosG5.append(Promedio_lista_F3G5)
PromediosG5.append(Promedio_lista_F4G5)
PromediosG5.append(Promedio_lista_F5G5)
PromediosG5.append(Promedio_lista_F6G5)
PromediosG5.append(Promedio_lista_F7G5)
PromediosG5.append(Promedio_lista_F8G5)
PromediosG5.append(Promedio_lista_F9G5)
PromediosG5.append(Promedio_lista_F10G5)
PromediosG5.append(Promedio_lista_F11G5)

#Segmentación G6
normxgstsg6freqs1 = normxgsts[3800:4000,25666:27749]
normxgstsg6freqs2 = normxgsts[3600:3800,25666:27749]
normxgstsg6freqs3 = normxgsts[3400:3600,25666:27749]
normxgstsg6freqs4 = normxgsts[3200:3400,25666:27749]
normxgstsg6freqs5 = normxgsts[3000:3200,25666:27749]
normxgstsg6freqs6 = normxgsts[2800:3000,25666:27749]
normxgstsg6freqs7 = normxgsts[2600:2800,25666:27749]
normxgstsg6freqs8 = normxgsts[2400:2600,25666:27749]
normxgstsg6freqs9 = normxgsts[2200:2400,25666:27749]
normxgstsg6freqs10 = normxgsts[2000:2200,25666:27749]
normxgstsg6freqs11 = normxgsts[1800:2000,25666:27749]
#Transformación matrices de NumPy en vectores por columna
lista_F1G6 = normxgstsg6freqs1.flatten(order='F')
lista_F2G6 = normxgstsg6freqs2.flatten(order='F')
lista_F3G6 = normxgstsg6freqs3.flatten(order='F')
lista_F4G6 = normxgstsg6freqs4.flatten(order='F')
lista_F5G6 = normxgstsg6freqs5.flatten(order='F')
lista_F6G6 = normxgstsg6freqs6.flatten(order='F')
lista_F7G6 = normxgstsg6freqs7.flatten(order='F')
lista_F8G6 = normxgstsg6freqs8.flatten(order='F')
lista_F9G6 = normxgstsg6freqs9.flatten(order='F')
lista_F10G6 = normxgstsg6freqs10.flatten(order='F')
lista_F11G6 = normxgstsg6freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G6 = mean(lista_F1G6)
Promedio_lista_F2G6 = mean(lista_F2G6)
Promedio_lista_F3G6 = mean(lista_F3G6)
Promedio_lista_F4G6 = mean(lista_F4G6)
Promedio_lista_F5G6 = mean(lista_F5G6)
Promedio_lista_F6G6 = mean(lista_F6G6)
Promedio_lista_F7G6 = mean(lista_F7G6)
Promedio_lista_F8G6 = mean(lista_F8G6)
Promedio_lista_F9G6 = mean(lista_F9G6)
Promedio_lista_F10G6 = mean(lista_F10G6)
Promedio_lista_F11G6 = mean(lista_F11G6)
#Lista de Promedios
PromediosG6 = []
PromediosG6.append(Promedio_lista_F1G6)
PromediosG6.append(Promedio_lista_F2G6)
PromediosG6.append(Promedio_lista_F3G6)
PromediosG6.append(Promedio_lista_F4G6)
PromediosG6.append(Promedio_lista_F5G6)
PromediosG6.append(Promedio_lista_F6G6)
PromediosG6.append(Promedio_lista_F7G6)
PromediosG6.append(Promedio_lista_F8G6)
PromediosG6.append(Promedio_lista_F9G6)
PromediosG6.append(Promedio_lista_F10G6)
PromediosG6.append(Promedio_lista_F11G6)
 
#Segmentación G7
normxgstsg7freqs1 = normxgsts[3800:4000,30557:32233]
normxgstsg7freqs2 = normxgsts[3600:3800,30557:32233]
normxgstsg7freqs3 = normxgsts[3400:3600,30557:32233]
normxgstsg7freqs4 = normxgsts[3200:3400,30557:32233]
normxgstsg7freqs5 = normxgsts[3000:3200,30557:32233]
normxgstsg7freqs6 = normxgsts[2800:3000,30557:32233]
normxgstsg7freqs7 = normxgsts[2600:2800,30557:32233]
normxgstsg7freqs8 = normxgsts[2400:2600,30557:32233]
normxgstsg7freqs9 = normxgsts[2200:2400,30557:32233]
normxgstsg7freqs10 = normxgsts[2000:2200,30557:32233]
normxgstsg7freqs11 = normxgsts[1800:2000,30557:32233]
#Transformación matrices de NumPy en vectores por columna
lista_F1G7 = normxgstsg7freqs1.flatten(order='F')
lista_F2G7 = normxgstsg7freqs2.flatten(order='F')
lista_F3G7 = normxgstsg7freqs3.flatten(order='F')
lista_F4G7 = normxgstsg7freqs4.flatten(order='F')
lista_F5G7 = normxgstsg7freqs5.flatten(order='F')
lista_F6G7 = normxgstsg7freqs6.flatten(order='F')
lista_F7G7 = normxgstsg7freqs7.flatten(order='F')
lista_F8G7 = normxgstsg7freqs8.flatten(order='F')
lista_F9G7 = normxgstsg7freqs9.flatten(order='F')
lista_F10G7 = normxgstsg7freqs10.flatten(order='F')
lista_F11G7 = normxgstsg7freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G7 = mean(lista_F1G7)
Promedio_lista_F2G7 = mean(lista_F2G7)
Promedio_lista_F3G7 = mean(lista_F3G7)
Promedio_lista_F4G7 = mean(lista_F4G7)
Promedio_lista_F5G7 = mean(lista_F5G7)
Promedio_lista_F6G7 = mean(lista_F6G7)
Promedio_lista_F7G7 = mean(lista_F7G7)
Promedio_lista_F8G7 = mean(lista_F8G7)
Promedio_lista_F9G7 = mean(lista_F9G7)
Promedio_lista_F10G7 = mean(lista_F10G7)
Promedio_lista_F11G7 = mean(lista_F11G7)
#Lista de Promedios
PromediosG7 = []
PromediosG7.append(Promedio_lista_F1G7)
PromediosG7.append(Promedio_lista_F2G7)
PromediosG7.append(Promedio_lista_F3G7)
PromediosG7.append(Promedio_lista_F4G7)
PromediosG7.append(Promedio_lista_F5G7)
PromediosG7.append(Promedio_lista_F6G7)
PromediosG7.append(Promedio_lista_F7G7)
PromediosG7.append(Promedio_lista_F8G7)
PromediosG7.append(Promedio_lista_F9G7)
PromediosG7.append(Promedio_lista_F10G7)
PromediosG7.append(Promedio_lista_F11G7)
 
#Segmentación G8
normxgstsg8freqs1 = normxgsts[3800:4000,34108:35929]
normxgstsg8freqs2 = normxgsts[3600:3800,34108:35929]
normxgstsg8freqs3 = normxgsts[3400:3600,34108:35929]
normxgstsg8freqs4 = normxgsts[3200:3400,34108:35929]
normxgstsg8freqs5 = normxgsts[3000:3200,34108:35929]
normxgstsg8freqs6 = normxgsts[2800:3000,34108:35929]
normxgstsg8freqs7 = normxgsts[2600:2800,34108:35929]
normxgstsg8freqs8 = normxgsts[2400:2600,34108:35929]
normxgstsg8freqs9 = normxgsts[2200:2400,34108:35929]
normxgstsg8freqs10 = normxgsts[2000:2200,34108:35929]
normxgstsg8freqs11 = normxgsts[1800:2000,34108:35929]
#Transformación matrices de NumPy en vectores por columna
lista_F1G8 = normxgstsg8freqs1.flatten(order='F')
lista_F2G8 = normxgstsg8freqs2.flatten(order='F')
lista_F3G8 = normxgstsg8freqs3.flatten(order='F')
lista_F4G8 = normxgstsg8freqs4.flatten(order='F')
lista_F5G8 = normxgstsg8freqs5.flatten(order='F')
lista_F6G8 = normxgstsg8freqs6.flatten(order='F')
lista_F7G8 = normxgstsg8freqs7.flatten(order='F')
lista_F8G8 = normxgstsg8freqs8.flatten(order='F')
lista_F9G8 = normxgstsg8freqs9.flatten(order='F')
lista_F10G8 = normxgstsg8freqs10.flatten(order='F')
lista_F11G8 = normxgstsg8freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G8 = mean(lista_F1G8)
Promedio_lista_F2G8 = mean(lista_F2G8)
Promedio_lista_F3G8 = mean(lista_F3G8)
Promedio_lista_F4G8 = mean(lista_F4G8)
Promedio_lista_F5G8 = mean(lista_F5G8)
Promedio_lista_F6G8 = mean(lista_F6G8)
Promedio_lista_F7G8 = mean(lista_F7G8)
Promedio_lista_F8G8 = mean(lista_F8G8)
Promedio_lista_F9G8 = mean(lista_F9G8)
Promedio_lista_F10G8 = mean(lista_F10G8)
Promedio_lista_F11G8 = mean(lista_F11G8)
#Lista de Promedios
PromediosG8 = []
PromediosG8.append(Promedio_lista_F1G8)
PromediosG8.append(Promedio_lista_F2G8)
PromediosG8.append(Promedio_lista_F3G8)
PromediosG8.append(Promedio_lista_F4G8)
PromediosG8.append(Promedio_lista_F5G8)
PromediosG8.append(Promedio_lista_F6G8)
PromediosG8.append(Promedio_lista_F7G8)
PromediosG8.append(Promedio_lista_F8G8)
PromediosG8.append(Promedio_lista_F9G8)
PromediosG8.append(Promedio_lista_F10G8)
PromediosG8.append(Promedio_lista_F11G8)

#Segmentación G9
normxgstsg9freqs1 = normxgsts[3800:4000,39641:41660]
normxgstsg9freqs2 = normxgsts[3600:3800,39641:41660]
normxgstsg9freqs3 = normxgsts[3400:3600,39641:41660]
normxgstsg9freqs4 = normxgsts[3200:3400,39641:41660]
normxgstsg9freqs5 = normxgsts[3000:3200,39641:41660]
normxgstsg9freqs6 = normxgsts[2800:3000,39641:41660]
normxgstsg9freqs7 = normxgsts[2600:2800,39641:41660]
normxgstsg9freqs8 = normxgsts[2400:2600,39641:41660]
normxgstsg9freqs9 = normxgsts[2200:2400,39641:41660]
normxgstsg9freqs10 = normxgsts[2000:2200,39641:41660]
normxgstsg9freqs11 = normxgsts[1800:2000,39641:41660]
#Transformación matrices de NumPy en vectores por columna
lista_F1G9 = normxgstsg9freqs1.flatten(order='F')
lista_F2G9 = normxgstsg9freqs2.flatten(order='F')
lista_F3G9 = normxgstsg9freqs3.flatten(order='F')
lista_F4G9 = normxgstsg9freqs4.flatten(order='F')
lista_F5G9 = normxgstsg9freqs5.flatten(order='F')
lista_F6G9 = normxgstsg9freqs6.flatten(order='F')
lista_F7G9 = normxgstsg9freqs7.flatten(order='F')
lista_F8G9 = normxgstsg9freqs8.flatten(order='F')
lista_F9G9 = normxgstsg9freqs9.flatten(order='F')
lista_F10G9 = normxgstsg9freqs10.flatten(order='F')
lista_F11G9 = normxgstsg9freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G9 = mean(lista_F1G9)
Promedio_lista_F2G9 = mean(lista_F2G9)
Promedio_lista_F3G9 = mean(lista_F3G9)
Promedio_lista_F4G9 = mean(lista_F4G9)
Promedio_lista_F5G9 = mean(lista_F5G9)
Promedio_lista_F6G9 = mean(lista_F6G9)
Promedio_lista_F7G9 = mean(lista_F7G9)
Promedio_lista_F8G9 = mean(lista_F8G9)
Promedio_lista_F9G9 = mean(lista_F9G9)
Promedio_lista_F10G9 = mean(lista_F10G9)
Promedio_lista_F11G9 = mean(lista_F11G9)
#Lista de Promedios
PromediosG9 = []
PromediosG9.append(Promedio_lista_F1G9)
PromediosG9.append(Promedio_lista_F2G9)
PromediosG9.append(Promedio_lista_F3G9)
PromediosG9.append(Promedio_lista_F4G9)
PromediosG9.append(Promedio_lista_F5G9)
PromediosG9.append(Promedio_lista_F6G9)
PromediosG9.append(Promedio_lista_F7G9)
PromediosG9.append(Promedio_lista_F8G9)
PromediosG9.append(Promedio_lista_F9G9)
PromediosG9.append(Promedio_lista_F10G9)
PromediosG9.append(Promedio_lista_F11G9)

#Segmentación G10
normxgstsg10freqs1 = normxgsts[3800:4000,45362:47548]
normxgstsg10freqs2 = normxgsts[3600:3800,45362:47548]
normxgstsg10freqs3 = normxgsts[3400:3600,45362:47548]
normxgstsg10freqs4 = normxgsts[3200:3400,45362:47548]
normxgstsg10freqs5 = normxgsts[3000:3200,45362:47548]
normxgstsg10freqs6 = normxgsts[2800:3000,45362:47548]
normxgstsg10freqs7 = normxgsts[2600:2800,45362:47548]
normxgstsg10freqs8 = normxgsts[2400:2600,45362:47548]
normxgstsg10freqs9 = normxgsts[2200:2400,45362:47548]
normxgstsg10freqs10 = normxgsts[2000:2200,45362:47548]
normxgstsg10freqs11 = normxgsts[1800:2000,45362:47548]
#Transformación matrices de NumPy en vectores por columna
lista_F1G10 = normxgstsg10freqs1.flatten(order='F')
lista_F2G10 = normxgstsg10freqs2.flatten(order='F')
lista_F3G10 = normxgstsg10freqs3.flatten(order='F')
lista_F4G10 = normxgstsg10freqs4.flatten(order='F')
lista_F5G10 = normxgstsg10freqs5.flatten(order='F')
lista_F6G10 = normxgstsg10freqs6.flatten(order='F')
lista_F7G10 = normxgstsg10freqs7.flatten(order='F')
lista_F8G10 = normxgstsg10freqs8.flatten(order='F')
lista_F9G10 = normxgstsg10freqs9.flatten(order='F')
lista_F10G10 = normxgstsg10freqs10.flatten(order='F')
lista_F11G10 = normxgstsg10freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G10 = mean(lista_F1G10)
Promedio_lista_F2G10 = mean(lista_F2G10)
Promedio_lista_F3G10 = mean(lista_F3G10)
Promedio_lista_F4G10 = mean(lista_F4G10)
Promedio_lista_F5G10 = mean(lista_F5G10)
Promedio_lista_F6G10 = mean(lista_F6G10)
Promedio_lista_F7G10 = mean(lista_F7G10)
Promedio_lista_F8G10 = mean(lista_F8G10)
Promedio_lista_F9G10 = mean(lista_F9G10)
Promedio_lista_F10G10 = mean(lista_F10G10)
Promedio_lista_F11G10 = mean(lista_F11G10)
#Lista de Promedios
PromediosG10 = []
PromediosG10.append(Promedio_lista_F1G10)
PromediosG10.append(Promedio_lista_F2G10)
PromediosG10.append(Promedio_lista_F3G10)
PromediosG10.append(Promedio_lista_F4G10)
PromediosG10.append(Promedio_lista_F5G10)
PromediosG10.append(Promedio_lista_F6G10)
PromediosG10.append(Promedio_lista_F7G10)
PromediosG10.append(Promedio_lista_F8G10)
PromediosG10.append(Promedio_lista_F9G10)
PromediosG10.append(Promedio_lista_F10G10)
PromediosG10.append(Promedio_lista_F11G10)
 
#Segmentación G11
normxgstsg11freqs1 = normxgsts[3800:4000,50807:52752]
normxgstsg11freqs2 = normxgsts[3600:3800,50807:52752]
normxgstsg11freqs3 = normxgsts[3400:3600,50807:52752]
normxgstsg11freqs4 = normxgsts[3200:3400,50807:52752]
normxgstsg11freqs5 = normxgsts[3000:3200,50807:52752]
normxgstsg11freqs6 = normxgsts[2800:3000,50807:52752]
normxgstsg11freqs7 = normxgsts[2600:2800,50807:52752]
normxgstsg11freqs8 = normxgsts[2400:2600,50807:52752]
normxgstsg11freqs9 = normxgsts[2200:2400,50807:52752]
normxgstsg11freqs10 = normxgsts[2000:2200,50807:52752]
normxgstsg11freqs11 = normxgsts[1800:2000,50807:52752]
#Transformación matrices de NumPy en vectores por columna
lista_F1G11 = normxgstsg11freqs1.flatten(order='F')
lista_F2G11 = normxgstsg11freqs2.flatten(order='F')
lista_F3G11 = normxgstsg11freqs3.flatten(order='F')
lista_F4G11 = normxgstsg11freqs4.flatten(order='F')
lista_F5G11 = normxgstsg11freqs5.flatten(order='F')
lista_F6G11 = normxgstsg11freqs6.flatten(order='F')
lista_F7G11 = normxgstsg11freqs7.flatten(order='F')
lista_F8G11 = normxgstsg11freqs8.flatten(order='F')
lista_F9G11 = normxgstsg11freqs9.flatten(order='F')
lista_F10G11 = normxgstsg11freqs10.flatten(order='F')
lista_F11G11 = normxgstsg11freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G11 = mean(lista_F1G11)
Promedio_lista_F2G11 = mean(lista_F2G11)
Promedio_lista_F3G11 = mean(lista_F3G11)
Promedio_lista_F4G11 = mean(lista_F4G11)
Promedio_lista_F5G11 = mean(lista_F5G11)
Promedio_lista_F6G11 = mean(lista_F6G11)
Promedio_lista_F7G11 = mean(lista_F7G11)
Promedio_lista_F8G11 = mean(lista_F8G11)
Promedio_lista_F9G11 = mean(lista_F9G11)
Promedio_lista_F10G11 = mean(lista_F10G11)
Promedio_lista_F11G11 = mean(lista_F11G11)
#Lista de Promedios
PromediosG11 = []
PromediosG11.append(Promedio_lista_F1G11)
PromediosG11.append(Promedio_lista_F2G11)
PromediosG11.append(Promedio_lista_F3G11)
PromediosG11.append(Promedio_lista_F4G11)
PromediosG11.append(Promedio_lista_F5G11)
PromediosG11.append(Promedio_lista_F6G11)
PromediosG11.append(Promedio_lista_F7G11)
PromediosG11.append(Promedio_lista_F8G11)
PromediosG11.append(Promedio_lista_F9G11)
PromediosG11.append(Promedio_lista_F10G11)
PromediosG11.append(Promedio_lista_F11G11)

#Segmentación G12
normxgstsg12freqs1 = normxgsts[3800:4000,56003:57709]
normxgstsg12freqs2 = normxgsts[3600:3800,56003:57709]
normxgstsg12freqs3 = normxgsts[3400:3600,56003:57709]
normxgstsg12freqs4 = normxgsts[3200:3400,56003:57709]
normxgstsg12freqs5 = normxgsts[3000:3200,56003:57709]
normxgstsg12freqs6 = normxgsts[2800:3000,56003:57709]
normxgstsg12freqs7 = normxgsts[2600:2800,56003:57709]
normxgstsg12freqs8 = normxgsts[2400:2600,56003:57709]
normxgstsg12freqs9 = normxgsts[2200:2400,56003:57709]
normxgstsg12freqs10 = normxgsts[2000:2200,56003:57709]
normxgstsg12freqs11 = normxgsts[1800:2000,56003:57709]
#Transformación matrices de NumPy en vectores por columna
lista_F1G12 = normxgstsg12freqs1.flatten(order='F')
lista_F2G12 = normxgstsg12freqs2.flatten(order='F')
lista_F3G12 = normxgstsg12freqs3.flatten(order='F')
lista_F4G12 = normxgstsg12freqs4.flatten(order='F')
lista_F5G12 = normxgstsg12freqs5.flatten(order='F')
lista_F6G12 = normxgstsg12freqs6.flatten(order='F')
lista_F7G12 = normxgstsg12freqs7.flatten(order='F')
lista_F8G12 = normxgstsg12freqs8.flatten(order='F')
lista_F9G12 = normxgstsg12freqs9.flatten(order='F')
lista_F10G12 = normxgstsg12freqs10.flatten(order='F')
lista_F11G12 = normxgstsg12freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G12 = mean(lista_F1G12)
Promedio_lista_F2G12 = mean(lista_F2G12)
Promedio_lista_F3G12 = mean(lista_F3G12)
Promedio_lista_F4G12 = mean(lista_F4G12)
Promedio_lista_F5G12 = mean(lista_F5G12)
Promedio_lista_F6G12 = mean(lista_F6G12)
Promedio_lista_F7G12 = mean(lista_F7G12)
Promedio_lista_F8G12 = mean(lista_F8G12)
Promedio_lista_F9G12 = mean(lista_F9G12)
Promedio_lista_F10G12 = mean(lista_F10G12)
Promedio_lista_F11G12 = mean(lista_F11G12)
#Lista de Promedios
PromediosG12 = []
PromediosG12.append(Promedio_lista_F1G12)
PromediosG12.append(Promedio_lista_F2G12)
PromediosG12.append(Promedio_lista_F3G12)
PromediosG12.append(Promedio_lista_F4G12)
PromediosG12.append(Promedio_lista_F5G12)
PromediosG12.append(Promedio_lista_F6G12)
PromediosG12.append(Promedio_lista_F7G12)
PromediosG12.append(Promedio_lista_F8G12)
PromediosG12.append(Promedio_lista_F9G12)
PromediosG12.append(Promedio_lista_F10G12)
PromediosG12.append(Promedio_lista_F11G12)

Promedios = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6, 
             PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]
PromediosFrecuencias = np.array(Promedios); 
PromediosFrecuencias = np.transpose(PromediosFrecuencias)

DF = pd.DataFrame(PromediosFrecuencias)
#DF.to_csv("Promedios Paciente 3 Canal 8.csv")

PromediosG1 = np.array(PromediosG1)
PromediosG1 = np.transpose(PromediosG1)

PromediosG2 = np.array(PromediosG2)
PromediosG2 = np.transpose(PromediosG2)

PromediosG3 = np.array(PromediosG3)
PromediosG3 = np.transpose(PromediosG3)

PromediosG4 = np.array(PromediosG4)
PromediosG4 = np.transpose(PromediosG4)

PromediosG5 = np.array(PromediosG5)
PromediosG5 = np.transpose(PromediosG5)

PromediosG6 = np.array(PromediosG6)
PromediosG6 = np.transpose(PromediosG6)

PromediosG7 = np.array(PromediosG7)
PromediosG7 = np.transpose(PromediosG7)

PromediosG8 = np.array(PromediosG8)
PromediosG8 = np.transpose(PromediosG8)

PromediosG9 = np.array(PromediosG9)
PromediosG9 = np.transpose(PromediosG9)

PromediosG10 = np.array(PromediosG10)
PromediosG10 = np.transpose(PromediosG10)

PromediosG11 = np.array(PromediosG11)
PromediosG11 = np.transpose(PromediosG11)

PromediosG12 = np.array(PromediosG12)
PromediosG12 = np.transpose(PromediosG12)

data = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6,
        PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]

#Secuencia Consecutiva de Matrices 
SPromediosG1 = list(itertools.chain(PromediosG1, PromediosG7))
SPromediosG1 = np.array(SPromediosG1)
SPromediosG2 = list(itertools.chain(PromediosG2, PromediosG8))
SPromediosG2 = np.array(SPromediosG2)
SPromediosG3 = list(itertools.chain(PromediosG3, PromediosG9))
SPromediosG3 = np.array(SPromediosG3)
SPromediosG4 = list(itertools.chain(PromediosG4, PromediosG10))
SPromediosG4 = np.array(SPromediosG4)
SPromediosG5 = list(itertools.chain(PromediosG5, PromediosG11))
SPromediosG5 = np.array(SPromediosG5)
SPromediosG6 = list(itertools.chain(PromediosG6, PromediosG12))
SPromediosG6 = np.array(SPromediosG6)
SPromedios = [SPromediosG1, SPromediosG2, SPromediosG3, SPromediosG4, SPromediosG5, SPromediosG6]

SPromediosU0 = list(itertools.chain(SPromedios[0], SPromedios[1]))
SPromediosU1 = list(itertools.chain(SPromediosU0, SPromedios[2]))
SPromediosU2 = list(itertools.chain(SPromediosU1, SPromedios[3]))
SPromediosU3 = list(itertools.chain(SPromediosU2, SPromedios[4]))
SPromediosU4 = list(itertools.chain(SPromediosU3, SPromedios[5]))
SPromediosU = np.array(SPromediosU4)

#Histograma
plt.figure(figsize =(16, 10))
plt.hist(SPromediosU)
plt.grid()
plt.title('Histograma de la Secuencia Consecutiva de Promedios')
plt.xlabel('Valor de la variable')
plt.ylabel('Conteo')
plt.savefig('Histograma Promedios Gesto 1 P3C6.png', dpi = 201)
plt.show()

#scipy stats Wilcoxon 
WCresultados = []
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[1], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[2], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)


stat, p_value = wilcoxon(SPromedios[3], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[4], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[5], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)

WCresultados = np.array(WCresultados)
WCresultados = np.reshape(WCresultados,[6,6])

DF = pd.DataFrame(WCresultados) 
DF.to_csv("Wilcoxon Resultados entre gestos distintos P3C8.csv")


emgpacientes05 = [voltajesP05[3], voltajesP05[5]]

emg_funcional = np.array(emgpacientes05[0])
#Lectura archivo 
samplerate = 1e3 #Frecuencia de muestreo
length = emg_funcional.shape[0] / samplerate #Duración total Electromiografía en segundos
#print("Duración Total = {length}s")

#Parametros CWT 
num = 4000 #Número de sampleo generado
wavelet = 'cmor1.0-1.5' # Bugs Fixed https://pywavelets.readthedocs.io/en/latest/release.1.1.0.html 
maxesc = np.log2(1500)
widths = np.logspace(0,maxesc, num=num, base=2,dtype='int')
#print("Escalas Usadas: ", widths)
dt = 1/samplerate # diferencia de intervalos de tiempo
frequencies = pywt.scale2frequency(wavelet, widths) / dt # Obtención de frecuencias correspondientes a escalas Modificar aquí precisi´´on
#print("Freqs asociadas a Escalas: ", frequencies)
# Calculo transformación continua de ondículas de los anchos de matriz numpy
wavelet_coeffs, freqs = pywt.cwt(emg_funcional, widths, wavelet=wavelet, sampling_period=samplerate, method='fft')
#print("Tamaño de wavelet transform: ", wavelet_coeffs.shape)
#print("Freqs asociadas a wavelet transform: ", freqs)

#Valores Absolutos de la CWT
abscwt = abs(wavelet_coeffs) # Devolución de ABS de los números complejos es la magnitud
gesto_1 = abscwt[:,1869:3844]
gesto_2 = abscwt[:,5800:7641]
gesto_3 = abscwt[:,11454:13231]
gesto_4 = abscwt[:,16878:18594]
gesto_5 = abscwt[:,22382:24008]
gesto_6 = abscwt[:,27711:29312]
gesto_7 = abscwt[:,33286:35047]
gesto_8 = abscwt[:,37775:39818]
gesto_9 = abscwt[:,41665:43445]
gesto_10 = abscwt[:,47530:49202]
gesto_11 = abscwt[:,53231:55021]
gesto_12 = abscwt[:,58951:60721]
gesto_13 = abscwt[:,64906:66760]
gesto_14 = abscwt[:,70097:71992]    

normxgsts = abscwt
#División de segmentos desde 0 a 57,69 Hz

#Segmentación 1
normxgstsg1freqs1 = normxgsts[3800:4000,1868:3845]
normxgstsg1freqs2 = normxgsts[3600:3800,1868:3845]
normxgstsg1freqs3 = normxgsts[3400:3600,1868:3845]
normxgstsg1freqs4 = normxgsts[3200:3400,1868:3845]
normxgstsg1freqs5 = normxgsts[3000:3200,1868:3845]
normxgstsg1freqs6 = normxgsts[2800:3000,1868:3845]
normxgstsg1freqs7 = normxgsts[2600:2800,1868:3845]
normxgstsg1freqs8 = normxgsts[2400:2600,1868:3845]
normxgstsg1freqs9 = normxgsts[2200:2400,1868:3845]
normxgstsg1freqs10 = normxgsts[2000:2200,1868:3845]
normxgstsg1freqs11 = normxgsts[1800:2000,1868:3845]
#Transformación matrices de NumPy en vectores por columna
lista_F1G1 = normxgstsg1freqs1.flatten(order='F')
lista_F2G1 = normxgstsg1freqs2.flatten(order='F')
lista_F3G1 = normxgstsg1freqs3.flatten(order='F')
lista_F4G1 = normxgstsg1freqs4.flatten(order='F')
lista_F5G1 = normxgstsg1freqs5.flatten(order='F')
lista_F6G1 = normxgstsg1freqs6.flatten(order='F')
lista_F7G1 = normxgstsg1freqs7.flatten(order='F')
lista_F8G1 = normxgstsg1freqs8.flatten(order='F')
lista_F9G1 = normxgstsg1freqs9.flatten(order='F')
lista_F10G1 = normxgstsg1freqs10.flatten(order='F')
lista_F11G1 = normxgstsg1freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G1 = mean(lista_F1G1)
Promedio_lista_F2G1 = mean(lista_F2G1)
Promedio_lista_F3G1 = mean(lista_F3G1)
Promedio_lista_F4G1 = mean(lista_F4G1)
Promedio_lista_F5G1 = mean(lista_F5G1)
Promedio_lista_F6G1 = mean(lista_F6G1)
Promedio_lista_F7G1 = mean(lista_F7G1)
Promedio_lista_F8G1 = mean(lista_F8G1)
Promedio_lista_F9G1 = mean(lista_F9G1)
Promedio_lista_F10G1 = mean(lista_F10G1)
Promedio_lista_F11G1 = mean(lista_F11G1)
#Lista de Promedios
PromediosG1 = []
PromediosG1.append(Promedio_lista_F1G1)
PromediosG1.append(Promedio_lista_F2G1)
PromediosG1.append(Promedio_lista_F3G1)
PromediosG1.append(Promedio_lista_F4G1)
PromediosG1.append(Promedio_lista_F5G1)
PromediosG1.append(Promedio_lista_F6G1)
PromediosG1.append(Promedio_lista_F7G1)
PromediosG1.append(Promedio_lista_F8G1)
PromediosG1.append(Promedio_lista_F9G1)
PromediosG1.append(Promedio_lista_F10G1)
PromediosG1.append(Promedio_lista_F11G1)
 
#Segmentación G2
normxgstsg2freqs1 = normxgsts[3800:4000,5799:7642]
normxgstsg2freqs2 = normxgsts[3600:3800,5799:7642]
normxgstsg2freqs3 = normxgsts[3400:3600,5799:7642]
normxgstsg2freqs4 = normxgsts[3200:3400,5799:7642]
normxgstsg2freqs5 = normxgsts[3000:3200,5799:7642]
normxgstsg2freqs6 = normxgsts[2800:3000,5799:7642]
normxgstsg2freqs7 = normxgsts[2600:2800,5799:7642]
normxgstsg2freqs8 = normxgsts[2400:2600,5799:7642]
normxgstsg2freqs9 = normxgsts[2200:2400,5799:7642]
normxgstsg2freqs10 = normxgsts[2000:2200,5799:7642]
normxgstsg2freqs11 = normxgsts[1800:2000,5799:7642]
#Transformación matrices de NumPy en vectores por columna
lista_F1G2 = normxgstsg2freqs1.flatten(order='F')
lista_F2G2 = normxgstsg2freqs2.flatten(order='F')
lista_F3G2 = normxgstsg2freqs3.flatten(order='F')
lista_F4G2 = normxgstsg2freqs4.flatten(order='F')
lista_F5G2 = normxgstsg2freqs5.flatten(order='F')
lista_F6G2 = normxgstsg2freqs6.flatten(order='F')
lista_F7G2 = normxgstsg2freqs7.flatten(order='F')
lista_F8G2 = normxgstsg2freqs8.flatten(order='F')
lista_F9G2 = normxgstsg2freqs9.flatten(order='F')
lista_F10G2 = normxgstsg2freqs10.flatten(order='F')
lista_F11G2 = normxgstsg2freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G2 = mean(lista_F1G2)
Promedio_lista_F2G2 = mean(lista_F2G2)
Promedio_lista_F3G2 = mean(lista_F3G2)
Promedio_lista_F4G2 = mean(lista_F4G2)
Promedio_lista_F5G2 = mean(lista_F5G2)
Promedio_lista_F6G2 = mean(lista_F6G2)
Promedio_lista_F7G2 = mean(lista_F7G2)
Promedio_lista_F8G2 = mean(lista_F8G2)
Promedio_lista_F9G2 = mean(lista_F9G2)
Promedio_lista_F10G2 = mean(lista_F10G2)
Promedio_lista_F11G2 = mean(lista_F11G2)
#Lista de Promedios
PromediosG2 = []
PromediosG2.append(Promedio_lista_F1G2)
PromediosG2.append(Promedio_lista_F2G2)
PromediosG2.append(Promedio_lista_F3G2)
PromediosG2.append(Promedio_lista_F4G2)
PromediosG2.append(Promedio_lista_F5G2)
PromediosG2.append(Promedio_lista_F6G2)
PromediosG2.append(Promedio_lista_F7G2)
PromediosG2.append(Promedio_lista_F8G2)
PromediosG2.append(Promedio_lista_F9G2)
PromediosG2.append(Promedio_lista_F10G2)
PromediosG2.append(Promedio_lista_F11G2)
 
#Segmentación G3
normxgstsg3freqs1 = normxgsts[3800:4000,11453:13232]
normxgstsg3freqs2 = normxgsts[3600:3800,11453:13232]
normxgstsg3freqs3 = normxgsts[3400:3600,11453:13232]
normxgstsg3freqs4 = normxgsts[3200:3400,11453:13232]
normxgstsg3freqs5 = normxgsts[3000:3200,11453:13232]
normxgstsg3freqs6 = normxgsts[2800:3000,11453:13232]
normxgstsg3freqs7 = normxgsts[2600:2800,11453:13232]
normxgstsg3freqs8 = normxgsts[2400:2600,11453:13232]
normxgstsg3freqs9 = normxgsts[2200:2400,11453:13232]
normxgstsg3freqs10 = normxgsts[2000:2200,11453:13232]
normxgstsg3freqs11 = normxgsts[1800:2000,11453:13232]
#Transformación matrices de NumPy en vectores por columna
lista_F1G3 = normxgstsg3freqs1.flatten(order='F')
lista_F2G3 = normxgstsg3freqs2.flatten(order='F')
lista_F3G3 = normxgstsg3freqs3.flatten(order='F')
lista_F4G3 = normxgstsg3freqs4.flatten(order='F')
lista_F5G3 = normxgstsg3freqs5.flatten(order='F')
lista_F6G3 = normxgstsg3freqs6.flatten(order='F')
lista_F7G3 = normxgstsg3freqs7.flatten(order='F')
lista_F8G3 = normxgstsg3freqs8.flatten(order='F')
lista_F9G3 = normxgstsg3freqs9.flatten(order='F')
lista_F10G3 = normxgstsg3freqs10.flatten(order='F')
lista_F11G3 = normxgstsg3freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G3 = mean(lista_F1G3)
Promedio_lista_F2G3 = mean(lista_F2G3)
Promedio_lista_F3G3 = mean(lista_F3G3)
Promedio_lista_F4G3 = mean(lista_F4G3)
Promedio_lista_F5G3 = mean(lista_F5G3)
Promedio_lista_F6G3 = mean(lista_F6G3)
Promedio_lista_F7G3 = mean(lista_F7G3)
Promedio_lista_F8G3 = mean(lista_F8G3)
Promedio_lista_F9G3 = mean(lista_F9G3)
Promedio_lista_F10G3 = mean(lista_F10G3)
Promedio_lista_F11G3 = mean(lista_F11G3)
#Lista de Promedios
PromediosG3 = []
PromediosG3.append(Promedio_lista_F1G3)
PromediosG3.append(Promedio_lista_F2G3)
PromediosG3.append(Promedio_lista_F3G3)
PromediosG3.append(Promedio_lista_F4G3)
PromediosG3.append(Promedio_lista_F5G3)
PromediosG3.append(Promedio_lista_F6G3)
PromediosG3.append(Promedio_lista_F7G3)
PromediosG3.append(Promedio_lista_F8G3)
PromediosG3.append(Promedio_lista_F9G3)
PromediosG3.append(Promedio_lista_F10G3)
PromediosG3.append(Promedio_lista_F11G3)  

#Segmentación G4
normxgstsg4freqs1 = normxgsts[3800:4000,16877:18595]
normxgstsg4freqs2 = normxgsts[3600:3800,16877:18595]
normxgstsg4freqs3 = normxgsts[3400:3600,16877:18595]
normxgstsg4freqs4 = normxgsts[3200:3400,16877:18595]
normxgstsg4freqs5 = normxgsts[3000:3200,16877:18595]
normxgstsg4freqs6 = normxgsts[2800:3000,16877:18595]
normxgstsg4freqs7 = normxgsts[2600:2800,16877:18595]
normxgstsg4freqs8 = normxgsts[2400:2600,16877:18595]
normxgstsg4freqs9 = normxgsts[2200:2400,16877:18595]
normxgstsg4freqs10 = normxgsts[2000:2200,16877:18595]
normxgstsg4freqs11 = normxgsts[1800:2000,16877:18595]
#Transformación matrices de NumPy en vectores por columna
lista_F1G4 = normxgstsg4freqs1.flatten(order='F')
lista_F2G4 = normxgstsg4freqs2.flatten(order='F')
lista_F3G4 = normxgstsg4freqs3.flatten(order='F')
lista_F4G4 = normxgstsg4freqs4.flatten(order='F')
lista_F5G4 = normxgstsg4freqs5.flatten(order='F')
lista_F6G4 = normxgstsg4freqs6.flatten(order='F')
lista_F7G4 = normxgstsg4freqs7.flatten(order='F')
lista_F8G4 = normxgstsg4freqs8.flatten(order='F')
lista_F9G4 = normxgstsg4freqs9.flatten(order='F')
lista_F10G4 = normxgstsg4freqs10.flatten(order='F')
lista_F11G4 = normxgstsg4freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G4 = mean(lista_F1G4)
Promedio_lista_F2G4 = mean(lista_F2G4)
Promedio_lista_F3G4 = mean(lista_F3G4)
Promedio_lista_F4G4 = mean(lista_F4G4)
Promedio_lista_F5G4 = mean(lista_F5G4)
Promedio_lista_F6G4 = mean(lista_F6G4)
Promedio_lista_F7G4 = mean(lista_F7G4)
Promedio_lista_F8G4 = mean(lista_F8G4)
Promedio_lista_F9G4 = mean(lista_F9G4)
Promedio_lista_F10G4 = mean(lista_F10G4)
Promedio_lista_F11G4 = mean(lista_F11G4)
#Lista de Promedios
PromediosG4 = []
PromediosG4.append(Promedio_lista_F1G4)
PromediosG4.append(Promedio_lista_F2G4)
PromediosG4.append(Promedio_lista_F3G4)
PromediosG4.append(Promedio_lista_F4G4)
PromediosG4.append(Promedio_lista_F5G4)
PromediosG4.append(Promedio_lista_F6G4)
PromediosG4.append(Promedio_lista_F7G4)
PromediosG4.append(Promedio_lista_F8G4)
PromediosG4.append(Promedio_lista_F9G4)
PromediosG4.append(Promedio_lista_F10G4)
PromediosG4.append(Promedio_lista_F11G4)      

#Segmentación G5
normxgstsg5freqs1 = normxgsts[3800:4000,22381:24009]
normxgstsg5freqs2 = normxgsts[3600:3800,22381:24009]
normxgstsg5freqs3 = normxgsts[3400:3600,22381:24009]
normxgstsg5freqs4 = normxgsts[3200:3400,22381:24009]
normxgstsg5freqs5 = normxgsts[3000:3200,22381:24009]
normxgstsg5freqs6 = normxgsts[2800:3000,22381:24009]
normxgstsg5freqs7 = normxgsts[2600:2800,22381:24009]
normxgstsg5freqs8 = normxgsts[2400:2600,22381:24009]
normxgstsg5freqs9 = normxgsts[2200:2400,22381:24009]
normxgstsg5freqs10 = normxgsts[2000:2200,22381:24009]
normxgstsg5freqs11 = normxgsts[1800:2000,22381:24009]
#Transformación matrices de NumPy en vectores por columna
lista_F1G5 = normxgstsg5freqs1.flatten(order='F')
lista_F2G5 = normxgstsg5freqs2.flatten(order='F')
lista_F3G5 = normxgstsg5freqs3.flatten(order='F')
lista_F4G5 = normxgstsg5freqs4.flatten(order='F')
lista_F5G5 = normxgstsg5freqs5.flatten(order='F')
lista_F6G5 = normxgstsg5freqs6.flatten(order='F')
lista_F7G5 = normxgstsg5freqs7.flatten(order='F')
lista_F8G5 = normxgstsg5freqs8.flatten(order='F')
lista_F9G5 = normxgstsg5freqs9.flatten(order='F')
lista_F10G5 = normxgstsg5freqs10.flatten(order='F')
lista_F11G5 = normxgstsg5freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G5 = mean(lista_F1G5)
Promedio_lista_F2G5 = mean(lista_F2G5)
Promedio_lista_F3G5 = mean(lista_F3G5)
Promedio_lista_F4G5 = mean(lista_F4G5)
Promedio_lista_F5G5 = mean(lista_F5G5)
Promedio_lista_F6G5 = mean(lista_F6G5)
Promedio_lista_F7G5 = mean(lista_F7G5)
Promedio_lista_F8G5 = mean(lista_F8G5)
Promedio_lista_F9G5 = mean(lista_F9G5)
Promedio_lista_F10G5 = mean(lista_F10G5)
Promedio_lista_F11G5 = mean(lista_F11G5)
#Lista de Promedios
PromediosG5 = []
PromediosG5.append(Promedio_lista_F1G5)
PromediosG5.append(Promedio_lista_F2G5)
PromediosG5.append(Promedio_lista_F3G5)
PromediosG5.append(Promedio_lista_F4G5)
PromediosG5.append(Promedio_lista_F5G5)
PromediosG5.append(Promedio_lista_F6G5)
PromediosG5.append(Promedio_lista_F7G5)
PromediosG5.append(Promedio_lista_F8G5)
PromediosG5.append(Promedio_lista_F9G5)
PromediosG5.append(Promedio_lista_F10G5)
PromediosG5.append(Promedio_lista_F11G5)
 
#Segmentación G6
normxgstsg6freqs1 = normxgsts[3800:4000,27710:29313]
normxgstsg6freqs2 = normxgsts[3600:3800,27710:29313]
normxgstsg6freqs3 = normxgsts[3400:3600,27710:29313]
normxgstsg6freqs4 = normxgsts[3200:3400,27710:29313]
normxgstsg6freqs5 = normxgsts[3000:3200,27710:29313]
normxgstsg6freqs6 = normxgsts[2800:3000,27710:29313]
normxgstsg6freqs7 = normxgsts[2600:2800,27710:29313]
normxgstsg6freqs8 = normxgsts[2400:2600,27710:29313]
normxgstsg6freqs9 = normxgsts[2200:2400,27710:29313]
normxgstsg6freqs10 = normxgsts[2000:2200,27710:29313]
normxgstsg6freqs11 = normxgsts[1800:2000,27710:29313]
#Transformación matrices de NumPy en vectores por columna
lista_F1G6 = normxgstsg6freqs1.flatten(order='F')
lista_F2G6 = normxgstsg6freqs2.flatten(order='F')
lista_F3G6 = normxgstsg6freqs3.flatten(order='F')
lista_F4G6 = normxgstsg6freqs4.flatten(order='F')
lista_F5G6 = normxgstsg6freqs5.flatten(order='F')
lista_F6G6 = normxgstsg6freqs6.flatten(order='F')
lista_F7G6 = normxgstsg6freqs7.flatten(order='F')
lista_F8G6 = normxgstsg6freqs8.flatten(order='F')
lista_F9G6 = normxgstsg6freqs9.flatten(order='F')
lista_F10G6 = normxgstsg6freqs10.flatten(order='F')
lista_F11G6 = normxgstsg6freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G6 = mean(lista_F1G6)
Promedio_lista_F2G6 = mean(lista_F2G6)
Promedio_lista_F3G6 = mean(lista_F3G6)
Promedio_lista_F4G6 = mean(lista_F4G6)
Promedio_lista_F5G6 = mean(lista_F5G6)
Promedio_lista_F6G6 = mean(lista_F6G6)
Promedio_lista_F7G6 = mean(lista_F7G6)
Promedio_lista_F8G6 = mean(lista_F8G6)
Promedio_lista_F9G6 = mean(lista_F9G6)
Promedio_lista_F10G6 = mean(lista_F10G6)
Promedio_lista_F11G6 = mean(lista_F11G6)
#Lista de Promedios
PromediosG6 = []
PromediosG6.append(Promedio_lista_F1G6)
PromediosG6.append(Promedio_lista_F2G6)
PromediosG6.append(Promedio_lista_F3G6)
PromediosG6.append(Promedio_lista_F4G6)
PromediosG6.append(Promedio_lista_F5G6)
PromediosG6.append(Promedio_lista_F6G6)
PromediosG6.append(Promedio_lista_F7G6)
PromediosG6.append(Promedio_lista_F8G6)
PromediosG6.append(Promedio_lista_F9G6)
PromediosG6.append(Promedio_lista_F10G6)
PromediosG6.append(Promedio_lista_F11G6)
 
#Segmentación G7
normxgstsg7freqs1 = normxgsts[3800:4000,33285:35048]
normxgstsg7freqs2 = normxgsts[3600:3800,33285:35048]
normxgstsg7freqs3 = normxgsts[3400:3600,33285:35048]
normxgstsg7freqs4 = normxgsts[3200:3400,33285:35048]
normxgstsg7freqs5 = normxgsts[3000:3200,33285:35048]
normxgstsg7freqs6 = normxgsts[2800:3000,33285:35048]
normxgstsg7freqs7 = normxgsts[2600:2800,33285:35048]
normxgstsg7freqs8 = normxgsts[2400:2600,33285:35048]
normxgstsg7freqs9 = normxgsts[2200:2400,33285:35048]
normxgstsg7freqs10 = normxgsts[2000:2200,33285:35048]
normxgstsg7freqs11 = normxgsts[1800:2000,33285:35048]
#Transformación matrices de NumPy en vectores por columna
lista_F1G7 = normxgstsg7freqs1.flatten(order='F')
lista_F2G7 = normxgstsg7freqs2.flatten(order='F')
lista_F3G7 = normxgstsg7freqs3.flatten(order='F')
lista_F4G7 = normxgstsg7freqs4.flatten(order='F')
lista_F5G7 = normxgstsg7freqs5.flatten(order='F')
lista_F6G7 = normxgstsg7freqs6.flatten(order='F')
lista_F7G7 = normxgstsg7freqs7.flatten(order='F')
lista_F8G7 = normxgstsg7freqs8.flatten(order='F')
lista_F9G7 = normxgstsg7freqs9.flatten(order='F')
lista_F10G7 = normxgstsg7freqs10.flatten(order='F')
lista_F11G7 = normxgstsg7freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G7 = mean(lista_F1G7)
Promedio_lista_F2G7 = mean(lista_F2G7)
Promedio_lista_F3G7 = mean(lista_F3G7)
Promedio_lista_F4G7 = mean(lista_F4G7)
Promedio_lista_F5G7 = mean(lista_F5G7)
Promedio_lista_F6G7 = mean(lista_F6G7)
Promedio_lista_F7G7 = mean(lista_F7G7)
Promedio_lista_F8G7 = mean(lista_F8G7)
Promedio_lista_F9G7 = mean(lista_F9G7)
Promedio_lista_F10G7 = mean(lista_F10G7)
Promedio_lista_F11G7 = mean(lista_F11G7)
#Lista de Promedios
PromediosG7 = []
PromediosG7.append(Promedio_lista_F1G7)
PromediosG7.append(Promedio_lista_F2G7)
PromediosG7.append(Promedio_lista_F3G7)
PromediosG7.append(Promedio_lista_F4G7)
PromediosG7.append(Promedio_lista_F5G7)
PromediosG7.append(Promedio_lista_F6G7)
PromediosG7.append(Promedio_lista_F7G7)
PromediosG7.append(Promedio_lista_F8G7)
PromediosG7.append(Promedio_lista_F9G7)
PromediosG7.append(Promedio_lista_F10G7)
PromediosG7.append(Promedio_lista_F11G7)
 
#Segmentación G8
normxgstsg8freqs1 = normxgsts[3800:4000,37774:39819]
normxgstsg8freqs2 = normxgsts[3600:3800,37774:39819]
normxgstsg8freqs3 = normxgsts[3400:3600,37774:39819]
normxgstsg8freqs4 = normxgsts[3200:3400,37774:39819]
normxgstsg8freqs5 = normxgsts[3000:3200,37774:39819]
normxgstsg8freqs6 = normxgsts[2800:3000,37774:39819]
normxgstsg8freqs7 = normxgsts[2600:2800,37774:39819]
normxgstsg8freqs8 = normxgsts[2400:2600,37774:39819]
normxgstsg8freqs9 = normxgsts[2200:2400,37774:39819]
normxgstsg8freqs10 = normxgsts[2000:2200,37774:39819]
normxgstsg8freqs11 = normxgsts[1800:2000,37774:39819]
#Transformación matrices de NumPy en vectores por columna
lista_F1G8 = normxgstsg8freqs1.flatten(order='F')
lista_F2G8 = normxgstsg8freqs2.flatten(order='F')
lista_F3G8 = normxgstsg8freqs3.flatten(order='F')
lista_F4G8 = normxgstsg8freqs4.flatten(order='F')
lista_F5G8 = normxgstsg8freqs5.flatten(order='F')
lista_F6G8 = normxgstsg8freqs6.flatten(order='F')
lista_F7G8 = normxgstsg8freqs7.flatten(order='F')
lista_F8G8 = normxgstsg8freqs8.flatten(order='F')
lista_F9G8 = normxgstsg8freqs9.flatten(order='F')
lista_F10G8 = normxgstsg8freqs10.flatten(order='F')
lista_F11G8 = normxgstsg8freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G8 = mean(lista_F1G8)
Promedio_lista_F2G8 = mean(lista_F2G8)
Promedio_lista_F3G8 = mean(lista_F3G8)
Promedio_lista_F4G8 = mean(lista_F4G8)
Promedio_lista_F5G8 = mean(lista_F5G8)
Promedio_lista_F6G8 = mean(lista_F6G8)
Promedio_lista_F7G8 = mean(lista_F7G8)
Promedio_lista_F8G8 = mean(lista_F8G8)
Promedio_lista_F9G8 = mean(lista_F9G8)
Promedio_lista_F10G8 = mean(lista_F10G8)
Promedio_lista_F11G8 = mean(lista_F11G8)
#Lista de Promedios
PromediosG8 = []
PromediosG8.append(Promedio_lista_F1G8)
PromediosG8.append(Promedio_lista_F2G8)
PromediosG8.append(Promedio_lista_F3G8)
PromediosG8.append(Promedio_lista_F4G8)
PromediosG8.append(Promedio_lista_F5G8)
PromediosG8.append(Promedio_lista_F6G8)
PromediosG8.append(Promedio_lista_F7G8)
PromediosG8.append(Promedio_lista_F8G8)
PromediosG8.append(Promedio_lista_F9G8)
PromediosG8.append(Promedio_lista_F10G8)
PromediosG8.append(Promedio_lista_F11G8)
 
#Segmentación G9
normxgstsg9freqs1 = normxgsts[3800:4000,41664:43446]
normxgstsg9freqs2 = normxgsts[3600:3800,41664:43446]
normxgstsg9freqs3 = normxgsts[3400:3600,41664:43446]
normxgstsg9freqs4 = normxgsts[3200:3400,41664:43446]
normxgstsg9freqs5 = normxgsts[3000:3200,41664:43446]
normxgstsg9freqs6 = normxgsts[2800:3000,41664:43446]
normxgstsg9freqs7 = normxgsts[2600:2800,41664:43446]
normxgstsg9freqs8 = normxgsts[2400:2600,41664:43446]
normxgstsg9freqs9 = normxgsts[2200:2400,41664:43446]
normxgstsg9freqs10 = normxgsts[2000:2200,41664:43446]
normxgstsg9freqs11 = normxgsts[1800:2000,41664:43446]
#Transformación matrices de NumPy en vectores por columna
lista_F1G9 = normxgstsg9freqs1.flatten(order='F')
lista_F2G9 = normxgstsg9freqs2.flatten(order='F')
lista_F3G9 = normxgstsg9freqs3.flatten(order='F')
lista_F4G9 = normxgstsg9freqs4.flatten(order='F')
lista_F5G9 = normxgstsg9freqs5.flatten(order='F')
lista_F6G9 = normxgstsg9freqs6.flatten(order='F')
lista_F7G9 = normxgstsg9freqs7.flatten(order='F')
lista_F8G9 = normxgstsg9freqs8.flatten(order='F')
lista_F9G9 = normxgstsg9freqs9.flatten(order='F')
lista_F10G9 = normxgstsg9freqs10.flatten(order='F')
lista_F11G9 = normxgstsg9freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G9 = mean(lista_F1G9)
Promedio_lista_F2G9 = mean(lista_F2G9)
Promedio_lista_F3G9 = mean(lista_F3G9)
Promedio_lista_F4G9 = mean(lista_F4G9)
Promedio_lista_F5G9 = mean(lista_F5G9)
Promedio_lista_F6G9 = mean(lista_F6G9)
Promedio_lista_F7G9 = mean(lista_F7G9)
Promedio_lista_F8G9 = mean(lista_F8G9)
Promedio_lista_F9G9 = mean(lista_F9G9)
Promedio_lista_F10G9 = mean(lista_F10G9)
Promedio_lista_F11G9 = mean(lista_F11G9)
#Lista de Promedios
PromediosG9 = []
PromediosG9.append(Promedio_lista_F1G9)
PromediosG9.append(Promedio_lista_F2G9)
PromediosG9.append(Promedio_lista_F3G9)
PromediosG9.append(Promedio_lista_F4G9)
PromediosG9.append(Promedio_lista_F5G9)
PromediosG9.append(Promedio_lista_F6G9)
PromediosG9.append(Promedio_lista_F7G9)
PromediosG9.append(Promedio_lista_F8G9)
PromediosG9.append(Promedio_lista_F9G9)
PromediosG9.append(Promedio_lista_F10G9)
PromediosG9.append(Promedio_lista_F11G9)
 
#Segmentación G10
normxgstsg10freqs1 = normxgsts[3800:4000,47529:49203]
normxgstsg10freqs2 = normxgsts[3600:3800,47529:49203]
normxgstsg10freqs3 = normxgsts[3400:3600,47529:49203]
normxgstsg10freqs4 = normxgsts[3200:3400,47529:49203]
normxgstsg10freqs5 = normxgsts[3000:3200,47529:49203]
normxgstsg10freqs6 = normxgsts[2800:3000,47529:49203]
normxgstsg10freqs7 = normxgsts[2600:2800,47529:49203]
normxgstsg10freqs8 = normxgsts[2400:2600,47529:49203]
normxgstsg10freqs9 = normxgsts[2200:2400,47529:49203]
normxgstsg10freqs10 = normxgsts[2000:2200,47529:49203]
normxgstsg10freqs11 = normxgsts[1800:2000,47529:49203]
#Transformación matrices de NumPy en vectores por columna
lista_F1G10 = normxgstsg10freqs1.flatten(order='F')
lista_F2G10 = normxgstsg10freqs2.flatten(order='F')
lista_F3G10 = normxgstsg10freqs3.flatten(order='F')
lista_F4G10 = normxgstsg10freqs4.flatten(order='F')
lista_F5G10 = normxgstsg10freqs5.flatten(order='F')
lista_F6G10 = normxgstsg10freqs6.flatten(order='F')
lista_F7G10 = normxgstsg10freqs7.flatten(order='F')
lista_F8G10 = normxgstsg10freqs8.flatten(order='F')
lista_F9G10 = normxgstsg10freqs9.flatten(order='F')
lista_F10G10 = normxgstsg10freqs10.flatten(order='F')
lista_F11G10 = normxgstsg10freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G10 = mean(lista_F1G10)
Promedio_lista_F2G10 = mean(lista_F2G10)
Promedio_lista_F3G10 = mean(lista_F3G10)
Promedio_lista_F4G10 = mean(lista_F4G10)
Promedio_lista_F5G10 = mean(lista_F5G10)
Promedio_lista_F6G10 = mean(lista_F6G10)
Promedio_lista_F7G10 = mean(lista_F7G10)
Promedio_lista_F8G10 = mean(lista_F8G10)
Promedio_lista_F9G10 = mean(lista_F9G10)
Promedio_lista_F10G10 = mean(lista_F10G10)
Promedio_lista_F11G10 = mean(lista_F11G10)
#Lista de Promedios
PromediosG10 = []
PromediosG10.append(Promedio_lista_F1G10)
PromediosG10.append(Promedio_lista_F2G10)
PromediosG10.append(Promedio_lista_F3G10)
PromediosG10.append(Promedio_lista_F4G10)
PromediosG10.append(Promedio_lista_F5G10)
PromediosG10.append(Promedio_lista_F6G10)
PromediosG10.append(Promedio_lista_F7G10)
PromediosG10.append(Promedio_lista_F8G10)
PromediosG10.append(Promedio_lista_F9G10)
PromediosG10.append(Promedio_lista_F10G10)
PromediosG10.append(Promedio_lista_F11G10)
 
#Segmentación G11
normxgstsg11freqs1 = normxgsts[3800:4000,53230:55022]
normxgstsg11freqs2 = normxgsts[3600:3800,53230:55022]
normxgstsg11freqs3 = normxgsts[3400:3600,53230:55022]
normxgstsg11freqs4 = normxgsts[3200:3400,53230:55022]
normxgstsg11freqs5 = normxgsts[3000:3200,53230:55022]
normxgstsg11freqs6 = normxgsts[2800:3000,53230:55022]
normxgstsg11freqs7 = normxgsts[2600:2800,53230:55022]
normxgstsg11freqs8 = normxgsts[2400:2600,53230:55022]
normxgstsg11freqs9 = normxgsts[2200:2400,53230:55022]
normxgstsg11freqs10 = normxgsts[2000:2200,53230:55022]
normxgstsg11freqs11 = normxgsts[1800:2000,53230:55022]
#Transformación matrices de NumPy en vectores por columna
lista_F1G11 = normxgstsg11freqs1.flatten(order='F')
lista_F2G11 = normxgstsg11freqs2.flatten(order='F')
lista_F3G11 = normxgstsg11freqs3.flatten(order='F')
lista_F4G11 = normxgstsg11freqs4.flatten(order='F')
lista_F5G11 = normxgstsg11freqs5.flatten(order='F')
lista_F6G11 = normxgstsg11freqs6.flatten(order='F')
lista_F7G11 = normxgstsg11freqs7.flatten(order='F')
lista_F8G11 = normxgstsg11freqs8.flatten(order='F')
lista_F9G11 = normxgstsg11freqs9.flatten(order='F')
lista_F10G11 = normxgstsg11freqs10.flatten(order='F')
lista_F11G11 = normxgstsg11freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G11 = mean(lista_F1G11)
Promedio_lista_F2G11 = mean(lista_F2G11)
Promedio_lista_F3G11 = mean(lista_F3G11)
Promedio_lista_F4G11 = mean(lista_F4G11)
Promedio_lista_F5G11 = mean(lista_F5G11)
Promedio_lista_F6G11 = mean(lista_F6G11)
Promedio_lista_F7G11 = mean(lista_F7G11)
Promedio_lista_F8G11 = mean(lista_F8G11)
Promedio_lista_F9G11 = mean(lista_F9G11)
Promedio_lista_F10G11 = mean(lista_F10G11)
Promedio_lista_F11G11 = mean(lista_F11G11)
#Lista de Promedios
PromediosG11 = []
PromediosG11.append(Promedio_lista_F1G11)
PromediosG11.append(Promedio_lista_F2G11)
PromediosG11.append(Promedio_lista_F3G11)
PromediosG11.append(Promedio_lista_F4G11)
PromediosG11.append(Promedio_lista_F5G11)
PromediosG11.append(Promedio_lista_F6G11)
PromediosG11.append(Promedio_lista_F7G11)
PromediosG11.append(Promedio_lista_F8G11)
PromediosG11.append(Promedio_lista_F9G11)
PromediosG11.append(Promedio_lista_F10G11)
PromediosG11.append(Promedio_lista_F11G11)

#Segmentación G12
normxgstsg12freqs1 = normxgsts[3800:4000,58950:60722]
normxgstsg12freqs2 = normxgsts[3600:3800,58950:60722]
normxgstsg12freqs3 = normxgsts[3400:3600,58950:60722]
normxgstsg12freqs4 = normxgsts[3200:3400,58950:60722]
normxgstsg12freqs5 = normxgsts[3000:3200,58950:60722]
normxgstsg12freqs6 = normxgsts[2800:3000,58950:60722]
normxgstsg12freqs7 = normxgsts[2600:2800,58950:60722]
normxgstsg12freqs8 = normxgsts[2400:2600,58950:60722]
normxgstsg12freqs9 = normxgsts[2200:2400,58950:60722]
normxgstsg12freqs10 = normxgsts[2000:2200,58950:60722]
normxgstsg12freqs11 = normxgsts[1800:2000,58950:60722]
#Transformación matrices de NumPy en vectores por columna
lista_F1G12 = normxgstsg12freqs1.flatten(order='F')
lista_F2G12 = normxgstsg12freqs2.flatten(order='F')
lista_F3G12 = normxgstsg12freqs3.flatten(order='F')
lista_F4G12 = normxgstsg12freqs4.flatten(order='F')
lista_F5G12 = normxgstsg12freqs5.flatten(order='F')
lista_F6G12 = normxgstsg12freqs6.flatten(order='F')
lista_F7G12 = normxgstsg12freqs7.flatten(order='F')
lista_F8G12 = normxgstsg12freqs8.flatten(order='F')
lista_F9G12 = normxgstsg12freqs9.flatten(order='F')
lista_F10G12 = normxgstsg12freqs10.flatten(order='F')
lista_F11G12 = normxgstsg12freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G12 = mean(lista_F1G12)
Promedio_lista_F2G12 = mean(lista_F2G12)
Promedio_lista_F3G12 = mean(lista_F3G12)
Promedio_lista_F4G12 = mean(lista_F4G12)
Promedio_lista_F5G12 = mean(lista_F5G12)
Promedio_lista_F6G12 = mean(lista_F6G12)
Promedio_lista_F7G12 = mean(lista_F7G12)
Promedio_lista_F8G12 = mean(lista_F8G12)
Promedio_lista_F9G12 = mean(lista_F9G12)
Promedio_lista_F10G12 = mean(lista_F10G12)
Promedio_lista_F11G12 = mean(lista_F11G12)
#Lista de Promedios
PromediosG12 = []
PromediosG12.append(Promedio_lista_F1G12)
PromediosG12.append(Promedio_lista_F2G12)
PromediosG12.append(Promedio_lista_F3G12)
PromediosG12.append(Promedio_lista_F4G12)
PromediosG12.append(Promedio_lista_F5G12)
PromediosG12.append(Promedio_lista_F6G12)
PromediosG12.append(Promedio_lista_F7G12)
PromediosG12.append(Promedio_lista_F8G12)
PromediosG12.append(Promedio_lista_F9G12)
PromediosG12.append(Promedio_lista_F10G12)
PromediosG12.append(Promedio_lista_F11G12)

#Segmentación G13
normxgstsg13freqs1 = normxgsts[3800:4000,64905:66761] 
normxgstsg13freqs2 = normxgsts[3600:3800,64905:66761] 
normxgstsg13freqs3 = normxgsts[3400:3600,64905:66761] 
normxgstsg13freqs4 = normxgsts[3200:3400,64905:66761] 
normxgstsg13freqs5 = normxgsts[3000:3200,64905:66761] 
normxgstsg13freqs6 = normxgsts[2800:3000,64905:66761] 
normxgstsg13freqs7 = normxgsts[2600:2800,64905:66761] 
normxgstsg13freqs8 = normxgsts[2400:2600,64905:66761] 
normxgstsg13freqs9 = normxgsts[2200:2400,64905:66761] 
normxgstsg13freqs10 = normxgsts[2000:2200,64905:66761] 
normxgstsg13freqs11 = normxgsts[1800:2000,64905:66761] 
#Transformación matrices de NumPy en vectores por columna
lista_F1G13 = normxgstsg13freqs1.flatten(order='F')
lista_F2G13 = normxgstsg13freqs2.flatten(order='F')
lista_F3G13 = normxgstsg13freqs3.flatten(order='F')
lista_F4G13 = normxgstsg13freqs4.flatten(order='F')
lista_F5G13 = normxgstsg13freqs5.flatten(order='F')
lista_F6G13 = normxgstsg13freqs6.flatten(order='F')
lista_F7G13 = normxgstsg13freqs7.flatten(order='F')
lista_F8G13 = normxgstsg13freqs8.flatten(order='F')
lista_F9G13 = normxgstsg13freqs9.flatten(order='F')
lista_F10G13 = normxgstsg13freqs10.flatten(order='F')
lista_F11G13 = normxgstsg13freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G13 = mean(lista_F1G13)
Promedio_lista_F2G13 = mean(lista_F2G13)
Promedio_lista_F3G13 = mean(lista_F3G13)
Promedio_lista_F4G13 = mean(lista_F4G13)
Promedio_lista_F5G13 = mean(lista_F5G13)
Promedio_lista_F6G13 = mean(lista_F6G13)
Promedio_lista_F7G13 = mean(lista_F7G13)
Promedio_lista_F8G13 = mean(lista_F8G13)
Promedio_lista_F9G13 = mean(lista_F9G13)
Promedio_lista_F10G13 = mean(lista_F10G13)
Promedio_lista_F11G13 = mean(lista_F11G13)
#Lista de Promedios
PromediosG13 = []
PromediosG13.append(Promedio_lista_F1G13)
PromediosG13.append(Promedio_lista_F2G13)
PromediosG13.append(Promedio_lista_F3G13)
PromediosG13.append(Promedio_lista_F4G13)
PromediosG13.append(Promedio_lista_F5G13)
PromediosG13.append(Promedio_lista_F6G13)
PromediosG13.append(Promedio_lista_F7G13)
PromediosG13.append(Promedio_lista_F8G13)
PromediosG13.append(Promedio_lista_F9G13)
PromediosG13.append(Promedio_lista_F10G13)
PromediosG13.append(Promedio_lista_F11G13)

#Segmentación G14
normxgstsg14freqs1 = normxgsts[3800:4000,70096:71993]
normxgstsg14freqs2 = normxgsts[3600:3800,70096:71993]
normxgstsg14freqs3 = normxgsts[3400:3600,70096:71993]
normxgstsg14freqs4 = normxgsts[3200:3400,70096:71993]
normxgstsg14freqs5 = normxgsts[3000:3200,70096:71993]
normxgstsg14freqs6 = normxgsts[2800:3000,70096:71993]
normxgstsg14freqs7 = normxgsts[2600:2800,70096:71993]
normxgstsg14freqs8 = normxgsts[2400:2600,70096:71993]
normxgstsg14freqs9 = normxgsts[2200:2400,70096:71993]
normxgstsg14freqs10 = normxgsts[2000:2200,70096:71993]
normxgstsg14freqs11 = normxgsts[1800:2000,70096:71993]
#Transformación matrices de NumPy en vectores por columna
lista_F1G14 = normxgstsg14freqs1.flatten(order='F')
lista_F2G14 = normxgstsg14freqs2.flatten(order='F')
lista_F3G14 = normxgstsg14freqs3.flatten(order='F')
lista_F4G14 = normxgstsg14freqs4.flatten(order='F')
lista_F5G14 = normxgstsg14freqs5.flatten(order='F')
lista_F6G14 = normxgstsg14freqs6.flatten(order='F')
lista_F7G14 = normxgstsg14freqs7.flatten(order='F')
lista_F8G14 = normxgstsg14freqs8.flatten(order='F')
lista_F9G14 = normxgstsg14freqs9.flatten(order='F')
lista_F10G14 = normxgstsg14freqs10.flatten(order='F')
lista_F11G14 = normxgstsg14freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G14 = mean(lista_F1G14)
Promedio_lista_F2G14 = mean(lista_F2G14)
Promedio_lista_F3G14 = mean(lista_F3G14)
Promedio_lista_F4G14 = mean(lista_F4G14)
Promedio_lista_F5G14 = mean(lista_F5G14)
Promedio_lista_F6G14 = mean(lista_F6G14)
Promedio_lista_F7G14 = mean(lista_F7G14)
Promedio_lista_F8G14 = mean(lista_F8G14)
Promedio_lista_F9G14 = mean(lista_F9G14)
Promedio_lista_F10G14 = mean(lista_F10G14)
Promedio_lista_F11G14 = mean(lista_F11G14)
#Lista de Promedios
PromediosG14 = []
PromediosG14.append(Promedio_lista_F1G14)
PromediosG14.append(Promedio_lista_F2G14)
PromediosG14.append(Promedio_lista_F3G14)
PromediosG14.append(Promedio_lista_F4G14)
PromediosG14.append(Promedio_lista_F5G14)
PromediosG14.append(Promedio_lista_F6G14)
PromediosG14.append(Promedio_lista_F7G14)
PromediosG14.append(Promedio_lista_F8G14)
PromediosG14.append(Promedio_lista_F9G14)
PromediosG14.append(Promedio_lista_F10G14)
PromediosG14.append(Promedio_lista_F11G14)

Promedios = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6, 
             PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12,
             PromediosG13, PromediosG14]
PromediosFrecuencias = np.array(Promedios); 
PromediosFrecuencias = np.transpose(PromediosFrecuencias)

DF = pd.DataFrame(PromediosFrecuencias)
#DF.to_csv("Promedios Paciente 5 Canal 4.csv")

PromediosG1 = np.array(PromediosG1)
PromediosG1 = np.transpose(PromediosG1)

PromediosG2 = np.array(PromediosG2)
PromediosG2 = np.transpose(PromediosG2)

PromediosG3 = np.array(PromediosG3)
PromediosG3 = np.transpose(PromediosG3)

PromediosG4 = np.array(PromediosG4)
PromediosG4 = np.transpose(PromediosG4)

PromediosG5 = np.array(PromediosG5)
PromediosG5 = np.transpose(PromediosG5)

PromediosG6 = np.array(PromediosG6)
PromediosG6 = np.transpose(PromediosG6)

PromediosG7 = np.array(PromediosG7)
PromediosG7 = np.transpose(PromediosG7)

PromediosG8 = np.array(PromediosG8)
PromediosG8 = np.transpose(PromediosG8)

PromediosG9 = np.array(PromediosG9)
PromediosG9 = np.transpose(PromediosG9)

PromediosG10 = np.array(PromediosG10)
PromediosG10 = np.transpose(PromediosG10)

PromediosG11 = np.array(PromediosG11)
PromediosG11 = np.transpose(PromediosG11)

PromediosG12 = np.array(PromediosG12)
PromediosG12 = np.transpose(PromediosG12)

PromediosG13 = np.array(PromediosG13)
PromediosG13 = np.transpose(PromediosG13)

PromediosG14 = np.array(PromediosG14)
PromediosG14 = np.transpose(PromediosG14)

data = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6,
        PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12,
        PromediosG13, PromediosG14]

#Secuencia Consecutiva de Matrices 
SPromediosG1 = list(itertools.chain(PromediosG1, PromediosG8))
SPromediosG1 = np.array(SPromediosG1)
SPromediosG2 = list(itertools.chain(PromediosG2, PromediosG9))
SPromediosG2 = np.array(SPromediosG2)
SPromediosG3 = list(itertools.chain(PromediosG3, PromediosG10))
SPromediosG3 = np.array(SPromediosG3)
SPromediosG4 = list(itertools.chain(PromediosG4, PromediosG11))
SPromediosG4 = np.array(SPromediosG4)
SPromediosG5 = list(itertools.chain(PromediosG5, PromediosG12))
SPromediosG5 = np.array(SPromediosG5)
SPromediosG6 = list(itertools.chain(PromediosG6, PromediosG13))
SPromediosG6 = np.array(SPromediosG6)
SPromediosG7 = list(itertools.chain(PromediosG7, PromediosG14))
SPromediosG7 = np.array(SPromediosG7)
SPromedios = [SPromediosG1, SPromediosG2, SPromediosG3, SPromediosG4, SPromediosG5, SPromediosG6, SPromediosG7]

SPromediosU0 = list(itertools.chain(SPromedios[0], SPromedios[1]))
SPromediosU1 = list(itertools.chain(SPromediosU0, SPromedios[2]))
SPromediosU2 = list(itertools.chain(SPromediosU1, SPromedios[3]))
SPromediosU3 = list(itertools.chain(SPromediosU2, SPromedios[4]))
SPromediosU4 = list(itertools.chain(SPromediosU3, SPromedios[5]))
SPromediosU5 = list(itertools.chain(SPromediosU4, SPromedios[6]))
SPromediosU = np.array(SPromediosU5)

#Histograma
plt.figure(figsize =(16, 10))
plt.hist(SPromediosU)
plt.grid()
plt.title('Histograma de la Secuencia Consecutiva de Promedios')
plt.xlabel('Valor de la variable')
plt.ylabel('Conteo')
plt.savefig('Histograma Promedios Gesto 1 P3C6.png', dpi = 201)
plt.show()

#scipy stats Wilcoxon 
WCresultados = []
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[1], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[2], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[3], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[4], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[5], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[6], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[6], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[6], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[6], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[6], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[6], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)

WCresultados = np.array(WCresultados)
WCresultados = np.reshape(WCresultados,[7,7])

DF = pd.DataFrame(WCresultados) 
DF.to_csv("Wilcoxon Resultados entre gestos distintos P5C4.csv")

emgpacientes05 = [voltajesP05[3], voltajesP05[5]]

emg_funcional = np.array(emgpacientes05[1])
#Lectura archivo 
samplerate = 1e3 #Frecuencia de muestreo
length = emg_funcional.shape[0] / samplerate #Duración total Electromiografía en segundos
#print("Duración Total = {length}s")

#Parametros CWT 
num = 4000 #Número de sampleo generado
wavelet = 'cmor1.0-1.5' # Bugs Fixed https://pywavelets.readthedocs.io/en/latest/release.1.1.0.html 
maxesc = np.log2(1500)
widths = np.logspace(0,maxesc, num=num, base=2,dtype='int')
#print("Escalas Usadas: ", widths)
dt = 1/samplerate # diferencia de intervalos de tiempo
frequencies = pywt.scale2frequency(wavelet, widths) / dt # Obtención de frecuencias correspondientes a escalas Modificar aquí precisi´´on
#print("Freqs asociadas a Escalas: ", frequencies)
# Calculo transformación continua de ondículas de los anchos de matriz numpy
wavelet_coeffs, freqs = pywt.cwt(emg_funcional, widths, wavelet=wavelet, sampling_period=samplerate, method='fft')
#print("Tamaño de wavelet transform: ", wavelet_coeffs.shape)
#print("Freqs asociadas a wavelet transform: ", freqs)

#Valores Absolutos de la CWT
abscwt = abs(wavelet_coeffs) # Devolución de ABS de los números complejos es la magnitud
gesto_1 = abscwt[:,1869:3844]
gesto_2 = abscwt[:,5800:7641]
gesto_3 = abscwt[:,11454:13231]
gesto_4 = abscwt[:,16878:18594]
gesto_5 = abscwt[:,22382:24008]
gesto_6 = abscwt[:,27711:29312]
gesto_7 = abscwt[:,33286:35047]
gesto_8 = abscwt[:,37775:39818]
gesto_9 = abscwt[:,41665:43445]
gesto_10 = abscwt[:,47530:49202]
gesto_11 = abscwt[:,53231:55021]
gesto_12 = abscwt[:,58951:60721]
gesto_13 = abscwt[:,64906:66760]
gesto_14 = abscwt[:,70097:71992]

normxgsts = abscwt
#División de segmentos desde 0 a 57,69 Hz

#Segmentación G1
normxgstsg1freqs1 = normxgsts[3800:4000,1868:3845]
normxgstsg1freqs2 = normxgsts[3600:3800,1868:3845]
normxgstsg1freqs3 = normxgsts[3400:3600,1868:3845]
normxgstsg1freqs4 = normxgsts[3200:3400,1868:3845]
normxgstsg1freqs5 = normxgsts[3000:3200,1868:3845]
normxgstsg1freqs6 = normxgsts[2800:3000,1868:3845]
normxgstsg1freqs7 = normxgsts[2600:2800,1868:3845]
normxgstsg1freqs8 = normxgsts[2400:2600,1868:3845]
normxgstsg1freqs9 = normxgsts[2200:2400,1868:3845]
normxgstsg1freqs10 = normxgsts[2000:2200,1868:3845]
normxgstsg1freqs11 = normxgsts[1800:2000,1868:3845]
#Transformación matrices de NumPy en vectores por columna
lista_F1G1 = normxgstsg1freqs1.flatten(order='F')
lista_F2G1 = normxgstsg1freqs2.flatten(order='F')
lista_F3G1 = normxgstsg1freqs3.flatten(order='F')
lista_F4G1 = normxgstsg1freqs4.flatten(order='F')
lista_F5G1 = normxgstsg1freqs5.flatten(order='F')
lista_F6G1 = normxgstsg1freqs6.flatten(order='F')
lista_F7G1 = normxgstsg1freqs7.flatten(order='F')
lista_F8G1 = normxgstsg1freqs8.flatten(order='F')
lista_F9G1 = normxgstsg1freqs9.flatten(order='F')
lista_F10G1 = normxgstsg1freqs10.flatten(order='F')
lista_F11G1 = normxgstsg1freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G1 = mean(lista_F1G1)
Promedio_lista_F2G1 = mean(lista_F2G1)
Promedio_lista_F3G1 = mean(lista_F3G1)
Promedio_lista_F4G1 = mean(lista_F4G1)
Promedio_lista_F5G1 = mean(lista_F5G1)
Promedio_lista_F6G1 = mean(lista_F6G1)
Promedio_lista_F7G1 = mean(lista_F7G1)
Promedio_lista_F8G1 = mean(lista_F8G1)
Promedio_lista_F9G1 = mean(lista_F9G1)
Promedio_lista_F10G1 = mean(lista_F10G1)
Promedio_lista_F11G1 = mean(lista_F11G1)
#Lista de Promedios
PromediosG1 = []
PromediosG1.append(Promedio_lista_F1G1)
PromediosG1.append(Promedio_lista_F2G1)
PromediosG1.append(Promedio_lista_F3G1)
PromediosG1.append(Promedio_lista_F4G1)
PromediosG1.append(Promedio_lista_F5G1)
PromediosG1.append(Promedio_lista_F6G1)
PromediosG1.append(Promedio_lista_F7G1)
PromediosG1.append(Promedio_lista_F8G1)
PromediosG1.append(Promedio_lista_F9G1)
PromediosG1.append(Promedio_lista_F10G1)
PromediosG1.append(Promedio_lista_F11G1)
 
#Segmentación G2
normxgstsg2freqs1 = normxgsts[3800:4000,5799:7642]
normxgstsg2freqs2 = normxgsts[3600:3800,5799:7642]
normxgstsg2freqs3 = normxgsts[3400:3600,5799:7642]
normxgstsg2freqs4 = normxgsts[3200:3400,5799:7642]
normxgstsg2freqs5 = normxgsts[3000:3200,5799:7642]
normxgstsg2freqs6 = normxgsts[2800:3000,5799:7642]
normxgstsg2freqs7 = normxgsts[2600:2800,5799:7642]
normxgstsg2freqs8 = normxgsts[2400:2600,5799:7642]
normxgstsg2freqs9 = normxgsts[2200:2400,5799:7642]
normxgstsg2freqs10 = normxgsts[2000:2200,5799:7642]
normxgstsg2freqs11 = normxgsts[1800:2000,5799:7642]
#Transformación matrices de NumPy en vectores por columna
lista_F1G2 = normxgstsg2freqs1.flatten(order='F')
lista_F2G2 = normxgstsg2freqs2.flatten(order='F')
lista_F3G2 = normxgstsg2freqs3.flatten(order='F')
lista_F4G2 = normxgstsg2freqs4.flatten(order='F')
lista_F5G2 = normxgstsg2freqs5.flatten(order='F')
lista_F6G2 = normxgstsg2freqs6.flatten(order='F')
lista_F7G2 = normxgstsg2freqs7.flatten(order='F')
lista_F8G2 = normxgstsg2freqs8.flatten(order='F')
lista_F9G2 = normxgstsg2freqs9.flatten(order='F')
lista_F10G2 = normxgstsg2freqs10.flatten(order='F')
lista_F11G2 = normxgstsg2freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G2 = mean(lista_F1G2)
Promedio_lista_F2G2 = mean(lista_F2G2)
Promedio_lista_F3G2 = mean(lista_F3G2)
Promedio_lista_F4G2 = mean(lista_F4G2)
Promedio_lista_F5G2 = mean(lista_F5G2)
Promedio_lista_F6G2 = mean(lista_F6G2)
Promedio_lista_F7G2 = mean(lista_F7G2)
Promedio_lista_F8G2 = mean(lista_F8G2)
Promedio_lista_F9G2 = mean(lista_F9G2)
Promedio_lista_F10G2 = mean(lista_F10G2)
Promedio_lista_F11G2 = mean(lista_F11G2)
#Lista de Promedios
PromediosG2 = []
PromediosG2.append(Promedio_lista_F1G2)
PromediosG2.append(Promedio_lista_F2G2)
PromediosG2.append(Promedio_lista_F3G2)
PromediosG2.append(Promedio_lista_F4G2)
PromediosG2.append(Promedio_lista_F5G2)
PromediosG2.append(Promedio_lista_F6G2)
PromediosG2.append(Promedio_lista_F7G2)
PromediosG2.append(Promedio_lista_F8G2)
PromediosG2.append(Promedio_lista_F9G2)
PromediosG2.append(Promedio_lista_F10G2)
PromediosG2.append(Promedio_lista_F11G2)
 
#Segmentación G3
normxgstsg3freqs1 = normxgsts[3800:4000,11453:13232]
normxgstsg3freqs2 = normxgsts[3600:3800,11453:13232]
normxgstsg3freqs3 = normxgsts[3400:3600,11453:13232]
normxgstsg3freqs4 = normxgsts[3200:3400,11453:13232]
normxgstsg3freqs5 = normxgsts[3000:3200,11453:13232]
normxgstsg3freqs6 = normxgsts[2800:3000,11453:13232]
normxgstsg3freqs7 = normxgsts[2600:2800,11453:13232]
normxgstsg3freqs8 = normxgsts[2400:2600,11453:13232]
normxgstsg3freqs9 = normxgsts[2200:2400,11453:13232]
normxgstsg3freqs10 = normxgsts[2000:2200,11453:13232]
normxgstsg3freqs11 = normxgsts[1800:2000,11453:13232]
#Transformación matrices de NumPy en vectores por columna
lista_F1G3 = normxgstsg3freqs1.flatten(order='F')
lista_F2G3 = normxgstsg3freqs2.flatten(order='F')
lista_F3G3 = normxgstsg3freqs3.flatten(order='F')
lista_F4G3 = normxgstsg3freqs4.flatten(order='F')
lista_F5G3 = normxgstsg3freqs5.flatten(order='F')
lista_F6G3 = normxgstsg3freqs6.flatten(order='F')
lista_F7G3 = normxgstsg3freqs7.flatten(order='F')
lista_F8G3 = normxgstsg3freqs8.flatten(order='F')
lista_F9G3 = normxgstsg3freqs9.flatten(order='F')
lista_F10G3 = normxgstsg3freqs10.flatten(order='F')
lista_F11G3 = normxgstsg3freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G3 = mean(lista_F1G3)
Promedio_lista_F2G3 = mean(lista_F2G3)
Promedio_lista_F3G3 = mean(lista_F3G3)
Promedio_lista_F4G3 = mean(lista_F4G3)
Promedio_lista_F5G3 = mean(lista_F5G3)
Promedio_lista_F6G3 = mean(lista_F6G3)
Promedio_lista_F7G3 = mean(lista_F7G3)
Promedio_lista_F8G3 = mean(lista_F8G3)
Promedio_lista_F9G3 = mean(lista_F9G3)
Promedio_lista_F10G3 = mean(lista_F10G3)
Promedio_lista_F11G3 = mean(lista_F11G3)
#Lista de Promedios
PromediosG3 = []
PromediosG3.append(Promedio_lista_F1G3)
PromediosG3.append(Promedio_lista_F2G3)
PromediosG3.append(Promedio_lista_F3G3)
PromediosG3.append(Promedio_lista_F4G3)
PromediosG3.append(Promedio_lista_F5G3)
PromediosG3.append(Promedio_lista_F6G3)
PromediosG3.append(Promedio_lista_F7G3)
PromediosG3.append(Promedio_lista_F8G3)
PromediosG3.append(Promedio_lista_F9G3)
PromediosG3.append(Promedio_lista_F10G3)
PromediosG3.append(Promedio_lista_F11G3)  
 
#Segmentación G4
normxgstsg4freqs1 = normxgsts[3800:4000,16877:18595]
normxgstsg4freqs2 = normxgsts[3600:3800,16877:18595]
normxgstsg4freqs3 = normxgsts[3400:3600,16877:18595]
normxgstsg4freqs4 = normxgsts[3200:3400,16877:18595]
normxgstsg4freqs5 = normxgsts[3000:3200,16877:18595]
normxgstsg4freqs6 = normxgsts[2800:3000,16877:18595]
normxgstsg4freqs7 = normxgsts[2600:2800,16877:18595]
normxgstsg4freqs8 = normxgsts[2400:2600,16877:18595]
normxgstsg4freqs9 = normxgsts[2200:2400,16877:18595]
normxgstsg4freqs10 = normxgsts[2000:2200,16877:18595]
normxgstsg4freqs11 = normxgsts[1800:2000,16877:18595]
#Transformación matrices de NumPy en vectores por columna
lista_F1G4 = normxgstsg4freqs1.flatten(order='F')
lista_F2G4 = normxgstsg4freqs2.flatten(order='F')
lista_F3G4 = normxgstsg4freqs3.flatten(order='F')
lista_F4G4 = normxgstsg4freqs4.flatten(order='F')
lista_F5G4 = normxgstsg4freqs5.flatten(order='F')
lista_F6G4 = normxgstsg4freqs6.flatten(order='F')
lista_F7G4 = normxgstsg4freqs7.flatten(order='F')
lista_F8G4 = normxgstsg4freqs8.flatten(order='F')
lista_F9G4 = normxgstsg4freqs9.flatten(order='F')
lista_F10G4 = normxgstsg4freqs10.flatten(order='F')
lista_F11G4 = normxgstsg4freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G4 = mean(lista_F1G4)
Promedio_lista_F2G4 = mean(lista_F2G4)
Promedio_lista_F3G4 = mean(lista_F3G4)
Promedio_lista_F4G4 = mean(lista_F4G4)
Promedio_lista_F5G4 = mean(lista_F5G4)
Promedio_lista_F6G4 = mean(lista_F6G4)
Promedio_lista_F7G4 = mean(lista_F7G4)
Promedio_lista_F8G4 = mean(lista_F8G4)
Promedio_lista_F9G4 = mean(lista_F9G4)
Promedio_lista_F10G4 = mean(lista_F10G4)
Promedio_lista_F11G4 = mean(lista_F11G4)
#Lista de Promedios
PromediosG4 = []
PromediosG4.append(Promedio_lista_F1G4)
PromediosG4.append(Promedio_lista_F2G4)
PromediosG4.append(Promedio_lista_F3G4)
PromediosG4.append(Promedio_lista_F4G4)
PromediosG4.append(Promedio_lista_F5G4)
PromediosG4.append(Promedio_lista_F6G4)
PromediosG4.append(Promedio_lista_F7G4)
PromediosG4.append(Promedio_lista_F8G4)
PromediosG4.append(Promedio_lista_F9G4)
PromediosG4.append(Promedio_lista_F10G4)
PromediosG4.append(Promedio_lista_F11G4)      

 
#Segmentación G5
normxgstsg5freqs1 = normxgsts[3800:4000,22381:24009]
normxgstsg5freqs2 = normxgsts[3600:3800,22381:24009]
normxgstsg5freqs3 = normxgsts[3400:3600,22381:24009]
normxgstsg5freqs4 = normxgsts[3200:3400,22381:24009]
normxgstsg5freqs5 = normxgsts[3000:3200,22381:24009]
normxgstsg5freqs6 = normxgsts[2800:3000,22381:24009]
normxgstsg5freqs7 = normxgsts[2600:2800,22381:24009]
normxgstsg5freqs8 = normxgsts[2400:2600,22381:24009]
normxgstsg5freqs9 = normxgsts[2200:2400,22381:24009]
normxgstsg5freqs10 = normxgsts[2000:2200,22381:24009]
normxgstsg5freqs11 = normxgsts[1800:2000,22381:24009]
#Transformación matrices de NumPy en vectores por columna
lista_F1G5 = normxgstsg5freqs1.flatten(order='F')
lista_F2G5 = normxgstsg5freqs2.flatten(order='F')
lista_F3G5 = normxgstsg5freqs3.flatten(order='F')
lista_F4G5 = normxgstsg5freqs4.flatten(order='F')
lista_F5G5 = normxgstsg5freqs5.flatten(order='F')
lista_F6G5 = normxgstsg5freqs6.flatten(order='F')
lista_F7G5 = normxgstsg5freqs7.flatten(order='F')
lista_F8G5 = normxgstsg5freqs8.flatten(order='F')
lista_F9G5 = normxgstsg5freqs9.flatten(order='F')
lista_F10G5 = normxgstsg5freqs10.flatten(order='F')
lista_F11G5 = normxgstsg5freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G5 = mean(lista_F1G5)
Promedio_lista_F2G5 = mean(lista_F2G5)
Promedio_lista_F3G5 = mean(lista_F3G5)
Promedio_lista_F4G5 = mean(lista_F4G5)
Promedio_lista_F5G5 = mean(lista_F5G5)
Promedio_lista_F6G5 = mean(lista_F6G5)
Promedio_lista_F7G5 = mean(lista_F7G5)
Promedio_lista_F8G5 = mean(lista_F8G5)
Promedio_lista_F9G5 = mean(lista_F9G5)
Promedio_lista_F10G5 = mean(lista_F10G5)
Promedio_lista_F11G5 = mean(lista_F11G5)
#Lista de Promedios
PromediosG5 = []
PromediosG5.append(Promedio_lista_F1G5)
PromediosG5.append(Promedio_lista_F2G5)
PromediosG5.append(Promedio_lista_F3G5)
PromediosG5.append(Promedio_lista_F4G5)
PromediosG5.append(Promedio_lista_F5G5)
PromediosG5.append(Promedio_lista_F6G5)
PromediosG5.append(Promedio_lista_F7G5)
PromediosG5.append(Promedio_lista_F8G5)
PromediosG5.append(Promedio_lista_F9G5)
PromediosG5.append(Promedio_lista_F10G5)
PromediosG5.append(Promedio_lista_F11G5)

 
#Segmentación G6
normxgstsg6freqs1 = normxgsts[3800:4000,27710:29313]
normxgstsg6freqs2 = normxgsts[3600:3800,27710:29313]
normxgstsg6freqs3 = normxgsts[3400:3600,27710:29313]
normxgstsg6freqs4 = normxgsts[3200:3400,27710:29313]
normxgstsg6freqs5 = normxgsts[3000:3200,27710:29313]
normxgstsg6freqs6 = normxgsts[2800:3000,27710:29313]
normxgstsg6freqs7 = normxgsts[2600:2800,27710:29313]
normxgstsg6freqs8 = normxgsts[2400:2600,27710:29313]
normxgstsg6freqs9 = normxgsts[2200:2400,27710:29313]
normxgstsg6freqs10 = normxgsts[2000:2200,27710:29313]
normxgstsg6freqs11 = normxgsts[1800:2000,27710:29313]
#Transformación matrices de NumPy en vectores por columna
lista_F1G6 = normxgstsg6freqs1.flatten(order='F')
lista_F2G6 = normxgstsg6freqs2.flatten(order='F')
lista_F3G6 = normxgstsg6freqs3.flatten(order='F')
lista_F4G6 = normxgstsg6freqs4.flatten(order='F')
lista_F5G6 = normxgstsg6freqs5.flatten(order='F')
lista_F6G6 = normxgstsg6freqs6.flatten(order='F')
lista_F7G6 = normxgstsg6freqs7.flatten(order='F')
lista_F8G6 = normxgstsg6freqs8.flatten(order='F')
lista_F9G6 = normxgstsg6freqs9.flatten(order='F')
lista_F10G6 = normxgstsg6freqs10.flatten(order='F')
lista_F11G6 = normxgstsg6freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G6 = mean(lista_F1G6)
Promedio_lista_F2G6 = mean(lista_F2G6)
Promedio_lista_F3G6 = mean(lista_F3G6)
Promedio_lista_F4G6 = mean(lista_F4G6)
Promedio_lista_F5G6 = mean(lista_F5G6)
Promedio_lista_F6G6 = mean(lista_F6G6)
Promedio_lista_F7G6 = mean(lista_F7G6)
Promedio_lista_F8G6 = mean(lista_F8G6)
Promedio_lista_F9G6 = mean(lista_F9G6)
Promedio_lista_F10G6 = mean(lista_F10G6)
Promedio_lista_F11G6 = mean(lista_F11G6)
#Lista de Promedios
PromediosG6 = []
PromediosG6.append(Promedio_lista_F1G6)
PromediosG6.append(Promedio_lista_F2G6)
PromediosG6.append(Promedio_lista_F3G6)
PromediosG6.append(Promedio_lista_F4G6)
PromediosG6.append(Promedio_lista_F5G6)
PromediosG6.append(Promedio_lista_F6G6)
PromediosG6.append(Promedio_lista_F7G6)
PromediosG6.append(Promedio_lista_F8G6)
PromediosG6.append(Promedio_lista_F9G6)
PromediosG6.append(Promedio_lista_F10G6)
PromediosG6.append(Promedio_lista_F11G6)
 
#Segmentación G7
normxgstsg7freqs1 = normxgsts[3800:4000,33285:35048]
normxgstsg7freqs2 = normxgsts[3600:3800,33285:35048]
normxgstsg7freqs3 = normxgsts[3400:3600,33285:35048]
normxgstsg7freqs4 = normxgsts[3200:3400,33285:35048]
normxgstsg7freqs5 = normxgsts[3000:3200,33285:35048]
normxgstsg7freqs6 = normxgsts[2800:3000,33285:35048]
normxgstsg7freqs7 = normxgsts[2600:2800,33285:35048]
normxgstsg7freqs8 = normxgsts[2400:2600,33285:35048]
normxgstsg7freqs9 = normxgsts[2200:2400,33285:35048]
normxgstsg7freqs10 = normxgsts[2000:2200,33285:35048]
normxgstsg7freqs11 = normxgsts[1800:2000,33285:35048]
#Transformación matrices de NumPy en vectores por columna
lista_F1G7 = normxgstsg7freqs1.flatten(order='F')
lista_F2G7 = normxgstsg7freqs2.flatten(order='F')
lista_F3G7 = normxgstsg7freqs3.flatten(order='F')
lista_F4G7 = normxgstsg7freqs4.flatten(order='F')
lista_F5G7 = normxgstsg7freqs5.flatten(order='F')
lista_F6G7 = normxgstsg7freqs6.flatten(order='F')
lista_F7G7 = normxgstsg7freqs7.flatten(order='F')
lista_F8G7 = normxgstsg7freqs8.flatten(order='F')
lista_F9G7 = normxgstsg7freqs9.flatten(order='F')
lista_F10G7 = normxgstsg7freqs10.flatten(order='F')
lista_F11G7 = normxgstsg7freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G7 = mean(lista_F1G7)
Promedio_lista_F2G7 = mean(lista_F2G7)
Promedio_lista_F3G7 = mean(lista_F3G7)
Promedio_lista_F4G7 = mean(lista_F4G7)
Promedio_lista_F5G7 = mean(lista_F5G7)
Promedio_lista_F6G7 = mean(lista_F6G7)
Promedio_lista_F7G7 = mean(lista_F7G7)
Promedio_lista_F8G7 = mean(lista_F8G7)
Promedio_lista_F9G7 = mean(lista_F9G7)
Promedio_lista_F10G7 = mean(lista_F10G7)
Promedio_lista_F11G7 = mean(lista_F11G7)
#Lista de Promedios
PromediosG7 = []
PromediosG7.append(Promedio_lista_F1G7)
PromediosG7.append(Promedio_lista_F2G7)
PromediosG7.append(Promedio_lista_F3G7)
PromediosG7.append(Promedio_lista_F4G7)
PromediosG7.append(Promedio_lista_F5G7)
PromediosG7.append(Promedio_lista_F6G7)
PromediosG7.append(Promedio_lista_F7G7)
PromediosG7.append(Promedio_lista_F8G7)
PromediosG7.append(Promedio_lista_F9G7)
PromediosG7.append(Promedio_lista_F10G7)
PromediosG7.append(Promedio_lista_F11G7)
 
#Segmentación G8
normxgstsg8freqs1 = normxgsts[3800:4000,37774:39819]
normxgstsg8freqs2 = normxgsts[3600:3800,37774:39819]
normxgstsg8freqs3 = normxgsts[3400:3600,37774:39819]
normxgstsg8freqs4 = normxgsts[3200:3400,37774:39819]
normxgstsg8freqs5 = normxgsts[3000:3200,37774:39819]
normxgstsg8freqs6 = normxgsts[2800:3000,37774:39819]
normxgstsg8freqs7 = normxgsts[2600:2800,37774:39819]
normxgstsg8freqs8 = normxgsts[2400:2600,37774:39819]
normxgstsg8freqs9 = normxgsts[2200:2400,37774:39819]
normxgstsg8freqs10 = normxgsts[2000:2200,37774:39819]
normxgstsg8freqs11 = normxgsts[1800:2000,37774:39819]
#Transformación matrices de NumPy en vectores por columna
lista_F1G8 = normxgstsg8freqs1.flatten(order='F')
lista_F2G8 = normxgstsg8freqs2.flatten(order='F')
lista_F3G8 = normxgstsg8freqs3.flatten(order='F')
lista_F4G8 = normxgstsg8freqs4.flatten(order='F')
lista_F5G8 = normxgstsg8freqs5.flatten(order='F')
lista_F6G8 = normxgstsg8freqs6.flatten(order='F')
lista_F7G8 = normxgstsg8freqs7.flatten(order='F')
lista_F8G8 = normxgstsg8freqs8.flatten(order='F')
lista_F9G8 = normxgstsg8freqs9.flatten(order='F')
lista_F10G8 = normxgstsg8freqs10.flatten(order='F')
lista_F11G8 = normxgstsg8freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G8 = mean(lista_F1G8)
Promedio_lista_F2G8 = mean(lista_F2G8)
Promedio_lista_F3G8 = mean(lista_F3G8)
Promedio_lista_F4G8 = mean(lista_F4G8)
Promedio_lista_F5G8 = mean(lista_F5G8)
Promedio_lista_F6G8 = mean(lista_F6G8)
Promedio_lista_F7G8 = mean(lista_F7G8)
Promedio_lista_F8G8 = mean(lista_F8G8)
Promedio_lista_F9G8 = mean(lista_F9G8)
Promedio_lista_F10G8 = mean(lista_F10G8)
Promedio_lista_F11G8 = mean(lista_F11G8)
#Lista de Promedios
PromediosG8 = []
PromediosG8.append(Promedio_lista_F1G8)
PromediosG8.append(Promedio_lista_F2G8)
PromediosG8.append(Promedio_lista_F3G8)
PromediosG8.append(Promedio_lista_F4G8)
PromediosG8.append(Promedio_lista_F5G8)
PromediosG8.append(Promedio_lista_F6G8)
PromediosG8.append(Promedio_lista_F7G8)
PromediosG8.append(Promedio_lista_F8G8)
PromediosG8.append(Promedio_lista_F9G8)
PromediosG8.append(Promedio_lista_F10G8)
PromediosG8.append(Promedio_lista_F11G8)
 
#Segmentación G9
normxgstsg9freqs1 = normxgsts[3800:4000,41664:43446]
normxgstsg9freqs2 = normxgsts[3600:3800,41664:43446]
normxgstsg9freqs3 = normxgsts[3400:3600,41664:43446]
normxgstsg9freqs4 = normxgsts[3200:3400,41664:43446]
normxgstsg9freqs5 = normxgsts[3000:3200,41664:43446]
normxgstsg9freqs6 = normxgsts[2800:3000,41664:43446]
normxgstsg9freqs7 = normxgsts[2600:2800,41664:43446]
normxgstsg9freqs8 = normxgsts[2400:2600,41664:43446]
normxgstsg9freqs9 = normxgsts[2200:2400,41664:43446]
normxgstsg9freqs10 = normxgsts[2000:2200,41664:43446]
normxgstsg9freqs11 = normxgsts[1800:2000,41664:43446]
#Transformación matrices de NumPy en vectores por columna
lista_F1G9 = normxgstsg9freqs1.flatten(order='F')
lista_F2G9 = normxgstsg9freqs2.flatten(order='F')
lista_F3G9 = normxgstsg9freqs3.flatten(order='F')
lista_F4G9 = normxgstsg9freqs4.flatten(order='F')
lista_F5G9 = normxgstsg9freqs5.flatten(order='F')
lista_F6G9 = normxgstsg9freqs6.flatten(order='F')
lista_F7G9 = normxgstsg9freqs7.flatten(order='F')
lista_F8G9 = normxgstsg9freqs8.flatten(order='F')
lista_F9G9 = normxgstsg9freqs9.flatten(order='F')
lista_F10G9 = normxgstsg9freqs10.flatten(order='F')
lista_F11G9 = normxgstsg9freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G9 = mean(lista_F1G9)
Promedio_lista_F2G9 = mean(lista_F2G9)
Promedio_lista_F3G9 = mean(lista_F3G9)
Promedio_lista_F4G9 = mean(lista_F4G9)
Promedio_lista_F5G9 = mean(lista_F5G9)
Promedio_lista_F6G9 = mean(lista_F6G9)
Promedio_lista_F7G9 = mean(lista_F7G9)
Promedio_lista_F8G9 = mean(lista_F8G9)
Promedio_lista_F9G9 = mean(lista_F9G9)
Promedio_lista_F10G9 = mean(lista_F10G9)
Promedio_lista_F11G9 = mean(lista_F11G9)
#Lista de Promedios
PromediosG9 = []
PromediosG9.append(Promedio_lista_F1G9)
PromediosG9.append(Promedio_lista_F2G9)
PromediosG9.append(Promedio_lista_F3G9)
PromediosG9.append(Promedio_lista_F4G9)
PromediosG9.append(Promedio_lista_F5G9)
PromediosG9.append(Promedio_lista_F6G9)
PromediosG9.append(Promedio_lista_F7G9)
PromediosG9.append(Promedio_lista_F8G9)
PromediosG9.append(Promedio_lista_F9G9)
PromediosG9.append(Promedio_lista_F10G9)
PromediosG9.append(Promedio_lista_F11G9)

#Segmentación G10
normxgstsg10freqs1 = normxgsts[3800:4000,47529:49203]
normxgstsg10freqs2 = normxgsts[3600:3800,47529:49203]
normxgstsg10freqs3 = normxgsts[3400:3600,47529:49203]
normxgstsg10freqs4 = normxgsts[3200:3400,47529:49203]
normxgstsg10freqs5 = normxgsts[3000:3200,47529:49203]
normxgstsg10freqs6 = normxgsts[2800:3000,47529:49203]
normxgstsg10freqs7 = normxgsts[2600:2800,47529:49203]
normxgstsg10freqs8 = normxgsts[2400:2600,47529:49203]
normxgstsg10freqs9 = normxgsts[2200:2400,47529:49203]
normxgstsg10freqs10 = normxgsts[2000:2200,47529:49203]
normxgstsg10freqs11 = normxgsts[1800:2000,47529:49203]
#Transformación matrices de NumPy en vectores por columna
lista_F1G10 = normxgstsg10freqs1.flatten(order='F')
lista_F2G10 = normxgstsg10freqs2.flatten(order='F')
lista_F3G10 = normxgstsg10freqs3.flatten(order='F')
lista_F4G10 = normxgstsg10freqs4.flatten(order='F')
lista_F5G10 = normxgstsg10freqs5.flatten(order='F')
lista_F6G10 = normxgstsg10freqs6.flatten(order='F')
lista_F7G10 = normxgstsg10freqs7.flatten(order='F')
lista_F8G10 = normxgstsg10freqs8.flatten(order='F')
lista_F9G10 = normxgstsg10freqs9.flatten(order='F')
lista_F10G10 = normxgstsg10freqs10.flatten(order='F')
lista_F11G10 = normxgstsg10freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G10 = mean(lista_F1G10)
Promedio_lista_F2G10 = mean(lista_F2G10)
Promedio_lista_F3G10 = mean(lista_F3G10)
Promedio_lista_F4G10 = mean(lista_F4G10)
Promedio_lista_F5G10 = mean(lista_F5G10)
Promedio_lista_F6G10 = mean(lista_F6G10)
Promedio_lista_F7G10 = mean(lista_F7G10)
Promedio_lista_F8G10 = mean(lista_F8G10)
Promedio_lista_F9G10 = mean(lista_F9G10)
Promedio_lista_F10G10 = mean(lista_F10G10)
Promedio_lista_F11G10 = mean(lista_F11G10)
#Lista de Promedios
PromediosG10 = []
PromediosG10.append(Promedio_lista_F1G10)
PromediosG10.append(Promedio_lista_F2G10)
PromediosG10.append(Promedio_lista_F3G10)
PromediosG10.append(Promedio_lista_F4G10)
PromediosG10.append(Promedio_lista_F5G10)
PromediosG10.append(Promedio_lista_F6G10)
PromediosG10.append(Promedio_lista_F7G10)
PromediosG10.append(Promedio_lista_F8G10)
PromediosG10.append(Promedio_lista_F9G10)
PromediosG10.append(Promedio_lista_F10G10)
PromediosG10.append(Promedio_lista_F11G10)
 
#Segmentación G11
normxgstsg11freqs1 = normxgsts[3800:4000,53230:55022]
normxgstsg11freqs2 = normxgsts[3600:3800,53230:55022]
normxgstsg11freqs3 = normxgsts[3400:3600,53230:55022]
normxgstsg11freqs4 = normxgsts[3200:3400,53230:55022]
normxgstsg11freqs5 = normxgsts[3000:3200,53230:55022]
normxgstsg11freqs6 = normxgsts[2800:3000,53230:55022]
normxgstsg11freqs7 = normxgsts[2600:2800,53230:55022]
normxgstsg11freqs8 = normxgsts[2400:2600,53230:55022]
normxgstsg11freqs9 = normxgsts[2200:2400,53230:55022]
normxgstsg11freqs10 = normxgsts[2000:2200,53230:55022]
normxgstsg11freqs11 = normxgsts[1800:2000,53230:55022]
#Transformación matrices de NumPy en vectores por columna
lista_F1G11 = normxgstsg11freqs1.flatten(order='F')
lista_F2G11 = normxgstsg11freqs2.flatten(order='F')
lista_F3G11 = normxgstsg11freqs3.flatten(order='F')
lista_F4G11 = normxgstsg11freqs4.flatten(order='F')
lista_F5G11 = normxgstsg11freqs5.flatten(order='F')
lista_F6G11 = normxgstsg11freqs6.flatten(order='F')
lista_F7G11 = normxgstsg11freqs7.flatten(order='F')
lista_F8G11 = normxgstsg11freqs8.flatten(order='F')
lista_F9G11 = normxgstsg11freqs9.flatten(order='F')
lista_F10G11 = normxgstsg11freqs10.flatten(order='F')
lista_F11G11 = normxgstsg11freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G11 = mean(lista_F1G11)
Promedio_lista_F2G11 = mean(lista_F2G11)
Promedio_lista_F3G11 = mean(lista_F3G11)
Promedio_lista_F4G11 = mean(lista_F4G11)
Promedio_lista_F5G11 = mean(lista_F5G11)
Promedio_lista_F6G11 = mean(lista_F6G11)
Promedio_lista_F7G11 = mean(lista_F7G11)
Promedio_lista_F8G11 = mean(lista_F8G11)
Promedio_lista_F9G11 = mean(lista_F9G11)
Promedio_lista_F10G11 = mean(lista_F10G11)
Promedio_lista_F11G11 = mean(lista_F11G11)
#Lista de Promedios
PromediosG11 = []
PromediosG11.append(Promedio_lista_F1G11)
PromediosG11.append(Promedio_lista_F2G11)
PromediosG11.append(Promedio_lista_F3G11)
PromediosG11.append(Promedio_lista_F4G11)
PromediosG11.append(Promedio_lista_F5G11)
PromediosG11.append(Promedio_lista_F6G11)
PromediosG11.append(Promedio_lista_F7G11)
PromediosG11.append(Promedio_lista_F8G11)
PromediosG11.append(Promedio_lista_F9G11)
PromediosG11.append(Promedio_lista_F10G11)
PromediosG11.append(Promedio_lista_F11G11)
 
#Segmentación G12
normxgstsg12freqs1 = normxgsts[3800:4000,58950:60722]
normxgstsg12freqs2 = normxgsts[3600:3800,58950:60722]
normxgstsg12freqs3 = normxgsts[3400:3600,58950:60722]
normxgstsg12freqs4 = normxgsts[3200:3400,58950:60722]
normxgstsg12freqs5 = normxgsts[3000:3200,58950:60722]
normxgstsg12freqs6 = normxgsts[2800:3000,58950:60722]
normxgstsg12freqs7 = normxgsts[2600:2800,58950:60722]
normxgstsg12freqs8 = normxgsts[2400:2600,58950:60722]
normxgstsg12freqs9 = normxgsts[2200:2400,58950:60722]
normxgstsg12freqs10 = normxgsts[2000:2200,58950:60722]
normxgstsg12freqs11 = normxgsts[1800:2000,58950:60722]
#Transformación matrices de NumPy en vectores por columna
lista_F1G12 = normxgstsg12freqs1.flatten(order='F')
lista_F2G12 = normxgstsg12freqs2.flatten(order='F')
lista_F3G12 = normxgstsg12freqs3.flatten(order='F')
lista_F4G12 = normxgstsg12freqs4.flatten(order='F')
lista_F5G12 = normxgstsg12freqs5.flatten(order='F')
lista_F6G12 = normxgstsg12freqs6.flatten(order='F')
lista_F7G12 = normxgstsg12freqs7.flatten(order='F')
lista_F8G12 = normxgstsg12freqs8.flatten(order='F')
lista_F9G12 = normxgstsg12freqs9.flatten(order='F')
lista_F10G12 = normxgstsg12freqs10.flatten(order='F')
lista_F11G12 = normxgstsg12freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G12 = mean(lista_F1G12)
Promedio_lista_F2G12 = mean(lista_F2G12)
Promedio_lista_F3G12 = mean(lista_F3G12)
Promedio_lista_F4G12 = mean(lista_F4G12)
Promedio_lista_F5G12 = mean(lista_F5G12)
Promedio_lista_F6G12 = mean(lista_F6G12)
Promedio_lista_F7G12 = mean(lista_F7G12)
Promedio_lista_F8G12 = mean(lista_F8G12)
Promedio_lista_F9G12 = mean(lista_F9G12)
Promedio_lista_F10G12 = mean(lista_F10G12)
Promedio_lista_F11G12 = mean(lista_F11G12)
#Lista de Promedios
PromediosG12 = []
PromediosG12.append(Promedio_lista_F1G12)
PromediosG12.append(Promedio_lista_F2G12)
PromediosG12.append(Promedio_lista_F3G12)
PromediosG12.append(Promedio_lista_F4G12)
PromediosG12.append(Promedio_lista_F5G12)
PromediosG12.append(Promedio_lista_F6G12)
PromediosG12.append(Promedio_lista_F7G12)
PromediosG12.append(Promedio_lista_F8G12)
PromediosG12.append(Promedio_lista_F9G12)
PromediosG12.append(Promedio_lista_F10G12)
PromediosG12.append(Promedio_lista_F11G12)
 
#Segmentación G13
normxgstsg13freqs1 = normxgsts[3800:4000,64905:66761] 
normxgstsg13freqs2 = normxgsts[3600:3800,64905:66761] 
normxgstsg13freqs3 = normxgsts[3400:3600,64905:66761] 
normxgstsg13freqs4 = normxgsts[3200:3400,64905:66761] 
normxgstsg13freqs5 = normxgsts[3000:3200,64905:66761] 
normxgstsg13freqs6 = normxgsts[2800:3000,64905:66761] 
normxgstsg13freqs7 = normxgsts[2600:2800,64905:66761] 
normxgstsg13freqs8 = normxgsts[2400:2600,64905:66761] 
normxgstsg13freqs9 = normxgsts[2200:2400,64905:66761] 
normxgstsg13freqs10 = normxgsts[2000:2200,64905:66761] 
normxgstsg13freqs11 = normxgsts[1800:2000,64905:66761] 
#Transformación matrices de NumPy en vectores por columna
lista_F1G13 = normxgstsg13freqs1.flatten(order='F')
lista_F2G13 = normxgstsg13freqs2.flatten(order='F')
lista_F3G13 = normxgstsg13freqs3.flatten(order='F')
lista_F4G13 = normxgstsg13freqs4.flatten(order='F')
lista_F5G13 = normxgstsg13freqs5.flatten(order='F')
lista_F6G13 = normxgstsg13freqs6.flatten(order='F')
lista_F7G13 = normxgstsg13freqs7.flatten(order='F')
lista_F8G13 = normxgstsg13freqs8.flatten(order='F')
lista_F9G13 = normxgstsg13freqs9.flatten(order='F')
lista_F10G13 = normxgstsg13freqs10.flatten(order='F')
lista_F11G13 = normxgstsg13freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G13 = mean(lista_F1G13)
Promedio_lista_F2G13 = mean(lista_F2G13)
Promedio_lista_F3G13 = mean(lista_F3G13)
Promedio_lista_F4G13 = mean(lista_F4G13)
Promedio_lista_F5G13 = mean(lista_F5G13)
Promedio_lista_F6G13 = mean(lista_F6G13)
Promedio_lista_F7G13 = mean(lista_F7G13)
Promedio_lista_F8G13 = mean(lista_F8G13)
Promedio_lista_F9G13 = mean(lista_F9G13)
Promedio_lista_F10G13 = mean(lista_F10G13)
Promedio_lista_F11G13 = mean(lista_F11G13)
#Lista de Promedios
PromediosG13 = []
PromediosG13.append(Promedio_lista_F1G13)
PromediosG13.append(Promedio_lista_F2G13)
PromediosG13.append(Promedio_lista_F3G13)
PromediosG13.append(Promedio_lista_F4G13)
PromediosG13.append(Promedio_lista_F5G13)
PromediosG13.append(Promedio_lista_F6G13)
PromediosG13.append(Promedio_lista_F7G13)
PromediosG13.append(Promedio_lista_F8G13)
PromediosG13.append(Promedio_lista_F9G13)
PromediosG13.append(Promedio_lista_F10G13)
PromediosG13.append(Promedio_lista_F11G13)
 
#Segmentación G14
normxgstsg14freqs1 = normxgsts[3800:4000,70096:71993]
normxgstsg14freqs2 = normxgsts[3600:3800,70096:71993]
normxgstsg14freqs3 = normxgsts[3400:3600,70096:71993]
normxgstsg14freqs4 = normxgsts[3200:3400,70096:71993]
normxgstsg14freqs5 = normxgsts[3000:3200,70096:71993]
normxgstsg14freqs6 = normxgsts[2800:3000,70096:71993]
normxgstsg14freqs7 = normxgsts[2600:2800,70096:71993]
normxgstsg14freqs8 = normxgsts[2400:2600,70096:71993]
normxgstsg14freqs9 = normxgsts[2200:2400,70096:71993]
normxgstsg14freqs10 = normxgsts[2000:2200,70096:71993]
normxgstsg14freqs11 = normxgsts[1800:2000,70096:71993]
#Transformación matrices de NumPy en vectores por columna
lista_F1G14 = normxgstsg14freqs1.flatten(order='F')
lista_F2G14 = normxgstsg14freqs2.flatten(order='F')
lista_F3G14 = normxgstsg14freqs3.flatten(order='F')
lista_F4G14 = normxgstsg14freqs4.flatten(order='F')
lista_F5G14 = normxgstsg14freqs5.flatten(order='F')
lista_F6G14 = normxgstsg14freqs6.flatten(order='F')
lista_F7G14 = normxgstsg14freqs7.flatten(order='F')
lista_F8G14 = normxgstsg14freqs8.flatten(order='F')
lista_F9G14 = normxgstsg14freqs9.flatten(order='F')
lista_F10G14 = normxgstsg14freqs10.flatten(order='F')
lista_F11G14 = normxgstsg14freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G14 = mean(lista_F1G14)
Promedio_lista_F2G14 = mean(lista_F2G14)
Promedio_lista_F3G14 = mean(lista_F3G14)
Promedio_lista_F4G14 = mean(lista_F4G14)
Promedio_lista_F5G14 = mean(lista_F5G14)
Promedio_lista_F6G14 = mean(lista_F6G14)
Promedio_lista_F7G14 = mean(lista_F7G14)
Promedio_lista_F8G14 = mean(lista_F8G14)
Promedio_lista_F9G14 = mean(lista_F9G14)
Promedio_lista_F10G14 = mean(lista_F10G14)
Promedio_lista_F11G14 = mean(lista_F11G14)
#Lista de Promedios
PromediosG14 = []
PromediosG14.append(Promedio_lista_F1G14)
PromediosG14.append(Promedio_lista_F2G14)
PromediosG14.append(Promedio_lista_F3G14)
PromediosG14.append(Promedio_lista_F4G14)
PromediosG14.append(Promedio_lista_F5G14)
PromediosG14.append(Promedio_lista_F6G14)
PromediosG14.append(Promedio_lista_F7G14)
PromediosG14.append(Promedio_lista_F8G14)
PromediosG14.append(Promedio_lista_F9G14)
PromediosG14.append(Promedio_lista_F10G14)
PromediosG14.append(Promedio_lista_F11G14)

Promedios = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6, 
             PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12,
             PromediosG13, PromediosG14]
PromediosFrecuencias = np.array(Promedios); 
PromediosFrecuencias = np.transpose(PromediosFrecuencias)

DF = pd.DataFrame(PromediosFrecuencias)
#DF.to_csv("Promedios Paciente 5 Canal 6.csv")

PromediosG1 = np.array(PromediosG1)
PromediosG1 = np.transpose(PromediosG1)

PromediosG2 = np.array(PromediosG2)
PromediosG2 = np.transpose(PromediosG2)

PromediosG3 = np.array(PromediosG3)
PromediosG3 = np.transpose(PromediosG3)

PromediosG4 = np.array(PromediosG4)
PromediosG4 = np.transpose(PromediosG4)

PromediosG5 = np.array(PromediosG5)
PromediosG5 = np.transpose(PromediosG5)

PromediosG6 = np.array(PromediosG6)
PromediosG6 = np.transpose(PromediosG6)

PromediosG7 = np.array(PromediosG7)
PromediosG7 = np.transpose(PromediosG7)

PromediosG8 = np.array(PromediosG8)
PromediosG8 = np.transpose(PromediosG8)

PromediosG9 = np.array(PromediosG9)
PromediosG9 = np.transpose(PromediosG9)

PromediosG10 = np.array(PromediosG10)
PromediosG10 = np.transpose(PromediosG10)

PromediosG11 = np.array(PromediosG11)
PromediosG11 = np.transpose(PromediosG11)

PromediosG12 = np.array(PromediosG12)
PromediosG12 = np.transpose(PromediosG12)

PromediosG13 = np.array(PromediosG13)
PromediosG13 = np.transpose(PromediosG13)

PromediosG14 = np.array(PromediosG14)
PromediosG14 = np.transpose(PromediosG14)

data = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6,
        PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12,
        PromediosG13, PromediosG14]

#Secuencia Consecutiva de Matrices 
SPromediosG1 = list(itertools.chain(PromediosG1, PromediosG8))
SPromediosG1 = np.array(SPromediosG1)
SPromediosG2 = list(itertools.chain(PromediosG2, PromediosG9))
SPromediosG2 = np.array(SPromediosG2)
SPromediosG3 = list(itertools.chain(PromediosG3, PromediosG10))
SPromediosG3 = np.array(SPromediosG3)
SPromediosG4 = list(itertools.chain(PromediosG4, PromediosG11))
SPromediosG4 = np.array(SPromediosG4)
SPromediosG5 = list(itertools.chain(PromediosG5, PromediosG12))
SPromediosG5 = np.array(SPromediosG5)
SPromediosG6 = list(itertools.chain(PromediosG6, PromediosG13))
SPromediosG6 = np.array(SPromediosG6)
SPromediosG7 = list(itertools.chain(PromediosG7, PromediosG14))
SPromediosG7 = np.array(SPromediosG7)
SPromedios = [SPromediosG1, SPromediosG2, SPromediosG3, SPromediosG4, SPromediosG5, SPromediosG6, SPromediosG7]

SPromediosU0 = list(itertools.chain(SPromedios[0], SPromedios[1]))
SPromediosU1 = list(itertools.chain(SPromediosU0, SPromedios[2]))
SPromediosU2 = list(itertools.chain(SPromediosU1, SPromedios[3]))
SPromediosU3 = list(itertools.chain(SPromediosU2, SPromedios[4]))
SPromediosU4 = list(itertools.chain(SPromediosU3, SPromedios[5]))
SPromediosU5 = list(itertools.chain(SPromediosU4, SPromedios[6]))
SPromediosU = np.array(SPromediosU5)

#Histograma
plt.figure(figsize =(16, 10))
plt.hist(SPromediosU)
plt.grid()
plt.title('Histograma de la Secuencia Consecutiva de Promedios')
plt.xlabel('Valor de la variable')
plt.ylabel('Conteo')
plt.savefig('Histograma Promedios Gesto 1 P3C6.png', dpi = 201)
plt.show()

#scipy stats Wilcoxon 
WCresultados = []
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[1], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[2], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[3], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[4], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[5], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[6], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[6], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[6], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[6], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[6], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[6], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[6], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)

WCresultados = np.array(WCresultados)
WCresultados = np.reshape(WCresultados,[7,7])

DF = pd.DataFrame(WCresultados) 
DF.to_csv("Wilcoxon Resultados entre gestos distintos P5C6.csv")

emg_funcional = np.array(voltajesP06[7])
#Lectura archivo 
samplerate = 1e3 #Frecuencia de muestreo
length = emg_funcional.shape[0] / samplerate #Duración total Electromiografía en segundos
#print("Duración Total = {length}s")

#Parametros CWT 
num = 4000 #Número de sampleo generado
wavelet = 'cmor1.0-1.5' # Bugs Fixed https://pywavelets.readthedocs.io/en/latest/release.1.1.0.html 
maxesc = np.log2(1500)
widths = np.logspace(0,maxesc, num=num, base=2,dtype='int')
#print("Escalas Usadas: ", widths)
dt = 1/samplerate # diferencia de intervalos de tiempo
frequencies = pywt.scale2frequency(wavelet, widths) / dt # Obtención de frecuencias correspondientes a escalas Modificar aquí precisi´´on
#print("Freqs asociadas a Escalas: ", frequencies)
# Calculo transformación continua de ondículas de los anchos de matriz numpy
wavelet_coeffs, freqs = pywt.cwt(emg_funcional, widths, wavelet=wavelet, sampling_period=samplerate, method='fft')
#print("Tamaño de wavelet transform: ", wavelet_coeffs.shape)
#print("Freqs asociadas a wavelet transform: ", freqs)

#Valores Absolutos de la CWT
abscwt = abs(wavelet_coeffs) # Devolución de ABS de los números complejos es la magnitud
gesto_1 = abscwt[:,1886:3664]
gesto_2 = abscwt[:,5866:7599]
gesto_3 = abscwt[:,11636:13371]
gesto_4 = abscwt[:,17022:18830]
gesto_5 = abscwt[:,24638:26767]
gesto_6 = abscwt[:,30392:32344]
gesto_7 = abscwt[:,34754:36511]
gesto_8 = abscwt[:,38419:40171]
gesto_9 = abscwt[:,43904:45710]
gesto_10 = abscwt[:,49402:51242]
gesto_11 = abscwt[:,55608:57365]
gesto_12 = abscwt[:,62000:63678]

normxgsts = abscwt
#División de segmentos desde 0 a 57,69 Hz

#Segmentación G1
normxgstsg1freqs1 = normxgsts[3800:4000,1885:3665]
normxgstsg1freqs2 = normxgsts[3600:3800,1885:3665]
normxgstsg1freqs3 = normxgsts[3400:3600,1885:3665]
normxgstsg1freqs4 = normxgsts[3200:3400,1885:3665]
normxgstsg1freqs5 = normxgsts[3000:3200,1885:3665]
normxgstsg1freqs6 = normxgsts[2800:3000,1885:3665]
normxgstsg1freqs7 = normxgsts[2600:2800,1885:3665]
normxgstsg1freqs8 = normxgsts[2400:2600,1885:3665]
normxgstsg1freqs9 = normxgsts[2200:2400,1885:3665]
normxgstsg1freqs10 = normxgsts[2000:2200,1885:3665]
normxgstsg1freqs11 = normxgsts[1800:2000,1885:3665]
#Transformación matrices de NumPy en vectores por columna
lista_F1G1 = normxgstsg1freqs1.flatten(order='F')
lista_F2G1 = normxgstsg1freqs2.flatten(order='F')
lista_F3G1 = normxgstsg1freqs3.flatten(order='F')
lista_F4G1 = normxgstsg1freqs4.flatten(order='F')
lista_F5G1 = normxgstsg1freqs5.flatten(order='F')
lista_F6G1 = normxgstsg1freqs6.flatten(order='F')
lista_F7G1 = normxgstsg1freqs7.flatten(order='F')
lista_F8G1 = normxgstsg1freqs8.flatten(order='F')
lista_F9G1 = normxgstsg1freqs9.flatten(order='F')
lista_F10G1 = normxgstsg1freqs10.flatten(order='F')
lista_F11G1 = normxgstsg1freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G1 = mean(lista_F1G1)
Promedio_lista_F2G1 = mean(lista_F2G1)
Promedio_lista_F3G1 = mean(lista_F3G1)
Promedio_lista_F4G1 = mean(lista_F4G1)
Promedio_lista_F5G1 = mean(lista_F5G1)
Promedio_lista_F6G1 = mean(lista_F6G1)
Promedio_lista_F7G1 = mean(lista_F7G1)
Promedio_lista_F8G1 = mean(lista_F8G1)
Promedio_lista_F9G1 = mean(lista_F9G1)
Promedio_lista_F10G1 = mean(lista_F10G1)
Promedio_lista_F11G1 = mean(lista_F11G1)
#Lista de Promedios
PromediosG1 = []
PromediosG1.append(Promedio_lista_F1G1)
PromediosG1.append(Promedio_lista_F2G1)
PromediosG1.append(Promedio_lista_F3G1)
PromediosG1.append(Promedio_lista_F4G1)
PromediosG1.append(Promedio_lista_F5G1)
PromediosG1.append(Promedio_lista_F6G1)
PromediosG1.append(Promedio_lista_F7G1)
PromediosG1.append(Promedio_lista_F8G1)
PromediosG1.append(Promedio_lista_F9G1)
PromediosG1.append(Promedio_lista_F10G1)
PromediosG1.append(Promedio_lista_F11G1)
 
#Segmentación G2
normxgstsg2freqs1 = normxgsts[3800:4000,5865:7600]
normxgstsg2freqs2 = normxgsts[3600:3800,5865:7600]
normxgstsg2freqs3 = normxgsts[3400:3600,5865:7600]
normxgstsg2freqs4 = normxgsts[3200:3400,5865:7600]
normxgstsg2freqs5 = normxgsts[3000:3200,5865:7600]
normxgstsg2freqs6 = normxgsts[2800:3000,5865:7600]
normxgstsg2freqs7 = normxgsts[2600:2800,5865:7600]
normxgstsg2freqs8 = normxgsts[2400:2600,5865:7600]
normxgstsg2freqs9 = normxgsts[2200:2400,5865:7600]
normxgstsg2freqs10 = normxgsts[2000:2200,5865:7600]
normxgstsg2freqs11 = normxgsts[1800:2000,5865:7600]
#Transformación matrices de NumPy en vectores por columna
lista_F1G2 = normxgstsg2freqs1.flatten(order='F')
lista_F2G2 = normxgstsg2freqs2.flatten(order='F')
lista_F3G2 = normxgstsg2freqs3.flatten(order='F')
lista_F4G2 = normxgstsg2freqs4.flatten(order='F')
lista_F5G2 = normxgstsg2freqs5.flatten(order='F')
lista_F6G2 = normxgstsg2freqs6.flatten(order='F')
lista_F7G2 = normxgstsg2freqs7.flatten(order='F')
lista_F8G2 = normxgstsg2freqs8.flatten(order='F')
lista_F9G2 = normxgstsg2freqs9.flatten(order='F')
lista_F10G2 = normxgstsg2freqs10.flatten(order='F')
lista_F11G2 = normxgstsg2freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G2 = mean(lista_F1G2)
Promedio_lista_F2G2 = mean(lista_F2G2)
Promedio_lista_F3G2 = mean(lista_F3G2)
Promedio_lista_F4G2 = mean(lista_F4G2)
Promedio_lista_F5G2 = mean(lista_F5G2)
Promedio_lista_F6G2 = mean(lista_F6G2)
Promedio_lista_F7G2 = mean(lista_F7G2)
Promedio_lista_F8G2 = mean(lista_F8G2)
Promedio_lista_F9G2 = mean(lista_F9G2)
Promedio_lista_F10G2 = mean(lista_F10G2)
Promedio_lista_F11G2 = mean(lista_F11G2)
#Lista de Promedios
PromediosG2 = []
PromediosG2.append(Promedio_lista_F1G2)
PromediosG2.append(Promedio_lista_F2G2)
PromediosG2.append(Promedio_lista_F3G2)
PromediosG2.append(Promedio_lista_F4G2)
PromediosG2.append(Promedio_lista_F5G2)
PromediosG2.append(Promedio_lista_F6G2)
PromediosG2.append(Promedio_lista_F7G2)
PromediosG2.append(Promedio_lista_F8G2)
PromediosG2.append(Promedio_lista_F9G2)
PromediosG2.append(Promedio_lista_F10G2)
PromediosG2.append(Promedio_lista_F11G2)
 
#Segmentación G3
normxgstsg3freqs1 = normxgsts[3800:4000,11635:13372]
normxgstsg3freqs2 = normxgsts[3600:3800,11635:13372]
normxgstsg3freqs3 = normxgsts[3400:3600,11635:13372]
normxgstsg3freqs4 = normxgsts[3200:3400,11635:13372]
normxgstsg3freqs5 = normxgsts[3000:3200,11635:13372]
normxgstsg3freqs6 = normxgsts[2800:3000,11635:13372]
normxgstsg3freqs7 = normxgsts[2600:2800,11635:13372]
normxgstsg3freqs8 = normxgsts[2400:2600,11635:13372]
normxgstsg3freqs9 = normxgsts[2200:2400,11635:13372]
normxgstsg3freqs10 = normxgsts[2000:2200,11635:13372]
normxgstsg3freqs11 = normxgsts[1800:2000,11635:13372]
#Transformación matrices de NumPy en vectores por columna
lista_F1G3 = normxgstsg3freqs1.flatten(order='F')
lista_F2G3 = normxgstsg3freqs2.flatten(order='F')
lista_F3G3 = normxgstsg3freqs3.flatten(order='F')
lista_F4G3 = normxgstsg3freqs4.flatten(order='F')
lista_F5G3 = normxgstsg3freqs5.flatten(order='F')
lista_F6G3 = normxgstsg3freqs6.flatten(order='F')
lista_F7G3 = normxgstsg3freqs7.flatten(order='F')
lista_F8G3 = normxgstsg3freqs8.flatten(order='F')
lista_F9G3 = normxgstsg3freqs9.flatten(order='F')
lista_F10G3 = normxgstsg3freqs10.flatten(order='F')
lista_F11G3 = normxgstsg3freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G3 = mean(lista_F1G3)
Promedio_lista_F2G3 = mean(lista_F2G3)
Promedio_lista_F3G3 = mean(lista_F3G3)
Promedio_lista_F4G3 = mean(lista_F4G3)
Promedio_lista_F5G3 = mean(lista_F5G3)
Promedio_lista_F6G3 = mean(lista_F6G3)
Promedio_lista_F7G3 = mean(lista_F7G3)
Promedio_lista_F8G3 = mean(lista_F8G3)
Promedio_lista_F9G3 = mean(lista_F9G3)
Promedio_lista_F10G3 = mean(lista_F10G3)
Promedio_lista_F11G3 = mean(lista_F11G3)
#Lista de Promedios
PromediosG3 = []
PromediosG3.append(Promedio_lista_F1G3)
PromediosG3.append(Promedio_lista_F2G3)
PromediosG3.append(Promedio_lista_F3G3)
PromediosG3.append(Promedio_lista_F4G3)
PromediosG3.append(Promedio_lista_F5G3)
PromediosG3.append(Promedio_lista_F6G3)
PromediosG3.append(Promedio_lista_F7G3)
PromediosG3.append(Promedio_lista_F8G3)
PromediosG3.append(Promedio_lista_F9G3)
PromediosG3.append(Promedio_lista_F10G3)
PromediosG3.append(Promedio_lista_F11G3)  
 
#Segmentación G4
normxgstsg4freqs1 = normxgsts[3800:4000,17021:18831]
normxgstsg4freqs2 = normxgsts[3600:3800,17021:18831]
normxgstsg4freqs3 = normxgsts[3400:3600,17021:18831]
normxgstsg4freqs4 = normxgsts[3200:3400,17021:18831]
normxgstsg4freqs5 = normxgsts[3000:3200,17021:18831]
normxgstsg4freqs6 = normxgsts[2800:3000,17021:18831]
normxgstsg4freqs7 = normxgsts[2600:2800,17021:18831]
normxgstsg4freqs8 = normxgsts[2400:2600,17021:18831]
normxgstsg4freqs9 = normxgsts[2200:2400,17021:18831]
normxgstsg4freqs10 = normxgsts[2000:2199,17021:18831]
normxgstsg4freqs11 = normxgsts[1800:2000,17021:18831]
#Transformación matrices de NumPy en vectores por columna
lista_F1G4 = normxgstsg4freqs1.flatten(order='F')
lista_F2G4 = normxgstsg4freqs2.flatten(order='F')
lista_F3G4 = normxgstsg4freqs3.flatten(order='F')
lista_F4G4 = normxgstsg4freqs4.flatten(order='F')
lista_F5G4 = normxgstsg4freqs5.flatten(order='F')
lista_F6G4 = normxgstsg4freqs6.flatten(order='F')
lista_F7G4 = normxgstsg4freqs7.flatten(order='F')
lista_F8G4 = normxgstsg4freqs8.flatten(order='F')
lista_F9G4 = normxgstsg4freqs9.flatten(order='F')
lista_F10G4 = normxgstsg4freqs10.flatten(order='F')
lista_F11G4 = normxgstsg4freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G4 = mean(lista_F1G4)
Promedio_lista_F2G4 = mean(lista_F2G4)
Promedio_lista_F3G4 = mean(lista_F3G4)
Promedio_lista_F4G4 = mean(lista_F4G4)
Promedio_lista_F5G4 = mean(lista_F5G4)
Promedio_lista_F6G4 = mean(lista_F6G4)
Promedio_lista_F7G4 = mean(lista_F7G4)
Promedio_lista_F8G4 = mean(lista_F8G4)
Promedio_lista_F9G4 = mean(lista_F9G4)
Promedio_lista_F10G4 = mean(lista_F10G4)
Promedio_lista_F11G4 = mean(lista_F11G4)
#Lista de Promedios
PromediosG4 = []
PromediosG4.append(Promedio_lista_F1G4)
PromediosG4.append(Promedio_lista_F2G4)
PromediosG4.append(Promedio_lista_F3G4)
PromediosG4.append(Promedio_lista_F4G4)
PromediosG4.append(Promedio_lista_F5G4)
PromediosG4.append(Promedio_lista_F6G4)
PromediosG4.append(Promedio_lista_F7G4)
PromediosG4.append(Promedio_lista_F8G4)
PromediosG4.append(Promedio_lista_F9G4)
PromediosG4.append(Promedio_lista_F10G4)
PromediosG4.append(Promedio_lista_F11G4)      
 
#Segmentación G5
normxgstsg5freqs1 = normxgsts[3800:4000,24637:26768]
normxgstsg5freqs2 = normxgsts[3600:3800,24637:26768]
normxgstsg5freqs3 = normxgsts[3400:3600,24637:26768]
normxgstsg5freqs4 = normxgsts[3200:3400,24637:26768]
normxgstsg5freqs5 = normxgsts[3000:3200,24637:26768]
normxgstsg5freqs6 = normxgsts[2800:3000,24637:26768]
normxgstsg5freqs7 = normxgsts[2600:2800,24637:26768]
normxgstsg5freqs8 = normxgsts[2400:2600,24637:26768]
normxgstsg5freqs9 = normxgsts[2200:2400,24637:26768]
normxgstsg5freqs10 = normxgsts[2000:2200,24637:26768]
normxgstsg5freqs11 = normxgsts[1800:2000,24637:26768]
#Transformación matrices de NumPy en vectores por columna
lista_F1G5 = normxgstsg5freqs1.flatten(order='F')
lista_F2G5 = normxgstsg5freqs2.flatten(order='F')
lista_F3G5 = normxgstsg5freqs3.flatten(order='F')
lista_F4G5 = normxgstsg5freqs4.flatten(order='F')
lista_F5G5 = normxgstsg5freqs5.flatten(order='F')
lista_F6G5 = normxgstsg5freqs6.flatten(order='F')
lista_F7G5 = normxgstsg5freqs7.flatten(order='F')
lista_F8G5 = normxgstsg5freqs8.flatten(order='F')
lista_F9G5 = normxgstsg5freqs9.flatten(order='F')
lista_F10G5 = normxgstsg5freqs10.flatten(order='F')
lista_F11G5 = normxgstsg5freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G5 = mean(lista_F1G5)
Promedio_lista_F2G5 = mean(lista_F2G5)
Promedio_lista_F3G5 = mean(lista_F3G5)
Promedio_lista_F4G5 = mean(lista_F4G5)
Promedio_lista_F5G5 = mean(lista_F5G5)
Promedio_lista_F6G5 = mean(lista_F6G5)
Promedio_lista_F7G5 = mean(lista_F7G5)
Promedio_lista_F8G5 = mean(lista_F8G5)
Promedio_lista_F9G5 = mean(lista_F9G5)
Promedio_lista_F10G5 = mean(lista_F10G5)
Promedio_lista_F11G5 = mean(lista_F11G5)
#Lista de Promedios
PromediosG5 = []
PromediosG5.append(Promedio_lista_F1G5)
PromediosG5.append(Promedio_lista_F2G5)
PromediosG5.append(Promedio_lista_F3G5)
PromediosG5.append(Promedio_lista_F4G5)
PromediosG5.append(Promedio_lista_F5G5)
PromediosG5.append(Promedio_lista_F6G5)
PromediosG5.append(Promedio_lista_F7G5)
PromediosG5.append(Promedio_lista_F8G5)
PromediosG5.append(Promedio_lista_F9G5)
PromediosG5.append(Promedio_lista_F10G5)
PromediosG5.append(Promedio_lista_F11G5)
 
#Segmentación G6
normxgstsg6freqs1 = normxgsts[3800:4000,30391:32345]
normxgstsg6freqs2 = normxgsts[3600:3800,30391:32345]
normxgstsg6freqs3 = normxgsts[3400:3600,30391:32345]
normxgstsg6freqs4 = normxgsts[3200:3400,30391:32345]
normxgstsg6freqs5 = normxgsts[3000:3200,30391:32345]
normxgstsg6freqs6 = normxgsts[2800:3000,30391:32345]
normxgstsg6freqs7 = normxgsts[2600:2800,30391:32345]
normxgstsg6freqs8 = normxgsts[2400:2600,30391:32345]
normxgstsg6freqs9 = normxgsts[2200:2400,30391:32345]
normxgstsg6freqs10 = normxgsts[2000:2200,30391:32345]
normxgstsg6freqs11 = normxgsts[1800:2000,30391:32345]
#Transformación matrices de NumPy en vectores por columna
lista_F1G6 = normxgstsg6freqs1.flatten(order='F')
lista_F2G6 = normxgstsg6freqs2.flatten(order='F')
lista_F3G6 = normxgstsg6freqs3.flatten(order='F')
lista_F4G6 = normxgstsg6freqs4.flatten(order='F')
lista_F5G6 = normxgstsg6freqs5.flatten(order='F')
lista_F6G6 = normxgstsg6freqs6.flatten(order='F')
lista_F7G6 = normxgstsg6freqs7.flatten(order='F')
lista_F8G6 = normxgstsg6freqs8.flatten(order='F')
lista_F9G6 = normxgstsg6freqs9.flatten(order='F')
lista_F10G6 = normxgstsg6freqs10.flatten(order='F')
lista_F11G6 = normxgstsg6freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G6 = mean(lista_F1G6)
Promedio_lista_F2G6 = mean(lista_F2G6)
Promedio_lista_F3G6 = mean(lista_F3G6)
Promedio_lista_F4G6 = mean(lista_F4G6)
Promedio_lista_F5G6 = mean(lista_F5G6)
Promedio_lista_F6G6 = mean(lista_F6G6)
Promedio_lista_F7G6 = mean(lista_F7G6)
Promedio_lista_F8G6 = mean(lista_F8G6)
Promedio_lista_F9G6 = mean(lista_F9G6)
Promedio_lista_F10G6 = mean(lista_F10G6)
Promedio_lista_F11G6 = mean(lista_F11G6)
#Lista de Promedios
PromediosG6 = []
PromediosG6.append(Promedio_lista_F1G6)
PromediosG6.append(Promedio_lista_F2G6)
PromediosG6.append(Promedio_lista_F3G6)
PromediosG6.append(Promedio_lista_F4G6)
PromediosG6.append(Promedio_lista_F5G6)
PromediosG6.append(Promedio_lista_F6G6)
PromediosG6.append(Promedio_lista_F7G6)
PromediosG6.append(Promedio_lista_F8G6)
PromediosG6.append(Promedio_lista_F9G6)
PromediosG6.append(Promedio_lista_F10G6)
PromediosG6.append(Promedio_lista_F11G6)
 
#Segmentación G7
normxgstsg7freqs1 = normxgsts[3800:4000,34753:36512]
normxgstsg7freqs2 = normxgsts[3600:3800,34753:36512]
normxgstsg7freqs3 = normxgsts[3400:3600,34753:36512]
normxgstsg7freqs4 = normxgsts[3200:3400,34753:36512]
normxgstsg7freqs5 = normxgsts[3000:3200,34753:36512]
normxgstsg7freqs6 = normxgsts[2800:3000,34753:36512]
normxgstsg7freqs7 = normxgsts[2600:2800,34753:36512]
normxgstsg7freqs8 = normxgsts[2400:2600,34753:36512]
normxgstsg7freqs9 = normxgsts[2200:2400,34753:36512]
normxgstsg7freqs10 = normxgsts[2000:2200,34753:36512]
normxgstsg7freqs11 = normxgsts[1800:2000,34753:36512]
#Transformación matrices de NumPy en vectores por columna
lista_F1G7 = normxgstsg7freqs1.flatten(order='F')
lista_F2G7 = normxgstsg7freqs2.flatten(order='F')
lista_F3G7 = normxgstsg7freqs3.flatten(order='F')
lista_F4G7 = normxgstsg7freqs4.flatten(order='F')
lista_F5G7 = normxgstsg7freqs5.flatten(order='F')
lista_F6G7 = normxgstsg7freqs6.flatten(order='F')
lista_F7G7 = normxgstsg7freqs7.flatten(order='F')
lista_F8G7 = normxgstsg7freqs8.flatten(order='F')
lista_F9G7 = normxgstsg7freqs9.flatten(order='F')
lista_F10G7 = normxgstsg7freqs10.flatten(order='F')
lista_F11G7 = normxgstsg7freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G7 = mean(lista_F1G7)
Promedio_lista_F2G7 = mean(lista_F2G7)
Promedio_lista_F3G7 = mean(lista_F3G7)
Promedio_lista_F4G7 = mean(lista_F4G7)
Promedio_lista_F5G7 = mean(lista_F5G7)
Promedio_lista_F6G7 = mean(lista_F6G7)
Promedio_lista_F7G7 = mean(lista_F7G7)
Promedio_lista_F8G7 = mean(lista_F8G7)
Promedio_lista_F9G7 = mean(lista_F9G7)
Promedio_lista_F10G7 = mean(lista_F10G7)
Promedio_lista_F11G7 = mean(lista_F11G7)
#Lista de Promedios
PromediosG7 = []
PromediosG7.append(Promedio_lista_F1G7)
PromediosG7.append(Promedio_lista_F2G7)
PromediosG7.append(Promedio_lista_F3G7)
PromediosG7.append(Promedio_lista_F4G7)
PromediosG7.append(Promedio_lista_F5G7)
PromediosG7.append(Promedio_lista_F6G7)
PromediosG7.append(Promedio_lista_F7G7)
PromediosG7.append(Promedio_lista_F8G7)
PromediosG7.append(Promedio_lista_F9G7)
PromediosG7.append(Promedio_lista_F10G7)
PromediosG7.append(Promedio_lista_F11G7)
 
#Segmentación G8
normxgstsg8freqs1 = normxgsts[3800:4000,38418:40172]
normxgstsg8freqs2 = normxgsts[3600:3800,38418:40172]
normxgstsg8freqs3 = normxgsts[3400:3600,38418:40172]
normxgstsg8freqs4 = normxgsts[3200:3400,38418:40172]
normxgstsg8freqs5 = normxgsts[3000:3200,38418:40172]
normxgstsg8freqs6 = normxgsts[2800:3000,38418:40172]
normxgstsg8freqs7 = normxgsts[2600:2800,38418:40172]
normxgstsg8freqs8 = normxgsts[2400:2600,38418:40172]
normxgstsg8freqs9 = normxgsts[2200:2400,38418:40172]
normxgstsg8freqs10 = normxgsts[2000:2200,38418:40172]
normxgstsg8freqs11 = normxgsts[1800:2000,38418:40172]
#Transformación matrices de NumPy en vectores por columna
lista_F1G8 = normxgstsg8freqs1.flatten(order='F')
lista_F2G8 = normxgstsg8freqs2.flatten(order='F')
lista_F3G8 = normxgstsg8freqs3.flatten(order='F')
lista_F4G8 = normxgstsg8freqs4.flatten(order='F')
lista_F5G8 = normxgstsg8freqs5.flatten(order='F')
lista_F6G8 = normxgstsg8freqs6.flatten(order='F')
lista_F7G8 = normxgstsg8freqs7.flatten(order='F')
lista_F8G8 = normxgstsg8freqs8.flatten(order='F')
lista_F9G8 = normxgstsg8freqs9.flatten(order='F')
lista_F10G8 = normxgstsg8freqs10.flatten(order='F')
lista_F11G8 = normxgstsg8freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G8 = mean(lista_F1G8)
Promedio_lista_F2G8 = mean(lista_F2G8)
Promedio_lista_F3G8 = mean(lista_F3G8)
Promedio_lista_F4G8 = mean(lista_F4G8)
Promedio_lista_F5G8 = mean(lista_F5G8)
Promedio_lista_F6G8 = mean(lista_F6G8)
Promedio_lista_F7G8 = mean(lista_F7G8)
Promedio_lista_F8G8 = mean(lista_F8G8)
Promedio_lista_F9G8 = mean(lista_F9G8)
Promedio_lista_F10G8 = mean(lista_F10G8)
Promedio_lista_F11G8 = mean(lista_F11G8)
#Lista de Promedios
PromediosG8 = []
PromediosG8.append(Promedio_lista_F1G8)
PromediosG8.append(Promedio_lista_F2G8)
PromediosG8.append(Promedio_lista_F3G8)
PromediosG8.append(Promedio_lista_F4G8)
PromediosG8.append(Promedio_lista_F5G8)
PromediosG8.append(Promedio_lista_F6G8)
PromediosG8.append(Promedio_lista_F7G8)
PromediosG8.append(Promedio_lista_F8G8)
PromediosG8.append(Promedio_lista_F9G8)
PromediosG8.append(Promedio_lista_F10G8)
PromediosG8.append(Promedio_lista_F11G8)
 
#Segmentación G9
normxgstsg9freqs1 = normxgsts[3800:4000,43903:45711]
normxgstsg9freqs2 = normxgsts[3600:3800,43903:45711]
normxgstsg9freqs3 = normxgsts[3400:3600,43903:45711]
normxgstsg9freqs4 = normxgsts[3200:3400,43903:45711]
normxgstsg9freqs5 = normxgsts[3000:3200,43903:45711]
normxgstsg9freqs6 = normxgsts[2800:3000,43903:45711]
normxgstsg9freqs7 = normxgsts[2600:2800,43903:45711]
normxgstsg9freqs8 = normxgsts[2400:2600,43903:45711]
normxgstsg9freqs9 = normxgsts[2200:2400,43903:45711]
normxgstsg9freqs10 = normxgsts[2000:2200,43903:45711]
normxgstsg9freqs11 = normxgsts[1800:2000,43903:45711]
#Transformación matrices de NumPy en vectores por columna
lista_F1G9 = normxgstsg9freqs1.flatten(order='F')
lista_F2G9 = normxgstsg9freqs2.flatten(order='F')
lista_F3G9 = normxgstsg9freqs3.flatten(order='F')
lista_F4G9 = normxgstsg9freqs4.flatten(order='F')
lista_F5G9 = normxgstsg9freqs5.flatten(order='F')
lista_F6G9 = normxgstsg9freqs6.flatten(order='F')
lista_F7G9 = normxgstsg9freqs7.flatten(order='F')
lista_F8G9 = normxgstsg9freqs8.flatten(order='F')
lista_F9G9 = normxgstsg9freqs9.flatten(order='F')
lista_F10G9 = normxgstsg9freqs10.flatten(order='F')
lista_F11G9 = normxgstsg9freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G9 = mean(lista_F1G9)
Promedio_lista_F2G9 = mean(lista_F2G9)
Promedio_lista_F3G9 = mean(lista_F3G9)
Promedio_lista_F4G9 = mean(lista_F4G9)
Promedio_lista_F5G9 = mean(lista_F5G9)
Promedio_lista_F6G9 = mean(lista_F6G9)
Promedio_lista_F7G9 = mean(lista_F7G9)
Promedio_lista_F8G9 = mean(lista_F8G9)
Promedio_lista_F9G9 = mean(lista_F9G9)
Promedio_lista_F10G9 = mean(lista_F10G9)
Promedio_lista_F11G9 = mean(lista_F11G9)
#Lista de Promedios
PromediosG9 = []
PromediosG9.append(Promedio_lista_F1G9)
PromediosG9.append(Promedio_lista_F2G9)
PromediosG9.append(Promedio_lista_F3G9)
PromediosG9.append(Promedio_lista_F4G9)
PromediosG9.append(Promedio_lista_F5G9)
PromediosG9.append(Promedio_lista_F6G9)
PromediosG9.append(Promedio_lista_F7G9)
PromediosG9.append(Promedio_lista_F8G9)
PromediosG9.append(Promedio_lista_F9G9)
PromediosG9.append(Promedio_lista_F10G9)
PromediosG9.append(Promedio_lista_F11G9)
 
#Segmentación G10
normxgstsg10freqs1 = normxgsts[3800:4000,49401:51243] 
normxgstsg10freqs2 = normxgsts[3600:3800,49401:51243] 
normxgstsg10freqs3 = normxgsts[3400:3600,49401:51243] 
normxgstsg10freqs4 = normxgsts[3200:3400,49401:51243] 
normxgstsg10freqs5 = normxgsts[3000:3200,49401:51243] 
normxgstsg10freqs6 = normxgsts[2800:3000,49401:51243] 
normxgstsg10freqs7 = normxgsts[2600:2800,49401:51243] 
normxgstsg10freqs8 = normxgsts[2400:2600,49401:51243] 
normxgstsg10freqs9 = normxgsts[2200:2400,49401:51243] 
normxgstsg10freqs10 = normxgsts[2000:2200,49401:51243] 
normxgstsg10freqs11 = normxgsts[1800:2000,49401:51243] 
#Transformación matrices de NumPy en vectores por columna
lista_F1G10 = normxgstsg10freqs1.flatten(order='F')
lista_F2G10 = normxgstsg10freqs2.flatten(order='F')
lista_F3G10 = normxgstsg10freqs3.flatten(order='F')
lista_F4G10 = normxgstsg10freqs4.flatten(order='F')
lista_F5G10 = normxgstsg10freqs5.flatten(order='F')
lista_F6G10 = normxgstsg10freqs6.flatten(order='F')
lista_F7G10 = normxgstsg10freqs7.flatten(order='F')
lista_F8G10 = normxgstsg10freqs8.flatten(order='F')
lista_F9G10 = normxgstsg10freqs9.flatten(order='F')
lista_F10G10 = normxgstsg10freqs10.flatten(order='F')
lista_F11G10 = normxgstsg10freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G10 = mean(lista_F1G10)
Promedio_lista_F2G10 = mean(lista_F2G10)
Promedio_lista_F3G10 = mean(lista_F3G10)
Promedio_lista_F4G10 = mean(lista_F4G10)
Promedio_lista_F5G10 = mean(lista_F5G10)
Promedio_lista_F6G10 = mean(lista_F6G10)
Promedio_lista_F7G10 = mean(lista_F7G10)
Promedio_lista_F8G10 = mean(lista_F8G10)
Promedio_lista_F9G10 = mean(lista_F9G10)
Promedio_lista_F10G10 = mean(lista_F10G10)
Promedio_lista_F11G10 = mean(lista_F11G10)
#Lista de Promedios
PromediosG10 = []
PromediosG10.append(Promedio_lista_F1G10)
PromediosG10.append(Promedio_lista_F2G10)
PromediosG10.append(Promedio_lista_F3G10)
PromediosG10.append(Promedio_lista_F4G10)
PromediosG10.append(Promedio_lista_F5G10)
PromediosG10.append(Promedio_lista_F6G10)
PromediosG10.append(Promedio_lista_F7G10)
PromediosG10.append(Promedio_lista_F8G10)
PromediosG10.append(Promedio_lista_F9G10)
PromediosG10.append(Promedio_lista_F10G10)
PromediosG10.append(Promedio_lista_F11G10)
 
#Segmentación G11
normxgstsg11freqs1 = normxgsts[3800:4000,55607:57366]
normxgstsg11freqs2 = normxgsts[3600:3800,55607:57366]
normxgstsg11freqs3 = normxgsts[3400:3600,55607:57366]
normxgstsg11freqs4 = normxgsts[3200:3400,55607:57366]
normxgstsg11freqs5 = normxgsts[3000:3200,55607:57366]
normxgstsg11freqs6 = normxgsts[2800:3000,55607:57366]
normxgstsg11freqs7 = normxgsts[2600:2800,55607:57366]
normxgstsg11freqs8 = normxgsts[2400:2600,55607:57366]
normxgstsg11freqs9 = normxgsts[2200:2400,55607:57366]
normxgstsg11freqs10 = normxgsts[2000:2200,55607:57366]
normxgstsg11freqs11 = normxgsts[1800:2000,55607:57366]
#Transformación matrices de NumPy en vectores por columna
lista_F1G11 = normxgstsg11freqs1.flatten(order='F')
lista_F2G11 = normxgstsg11freqs2.flatten(order='F')
lista_F3G11 = normxgstsg11freqs3.flatten(order='F')
lista_F4G11 = normxgstsg11freqs4.flatten(order='F')
lista_F5G11 = normxgstsg11freqs5.flatten(order='F')
lista_F6G11 = normxgstsg11freqs6.flatten(order='F')
lista_F7G11 = normxgstsg11freqs7.flatten(order='F')
lista_F8G11 = normxgstsg11freqs8.flatten(order='F')
lista_F9G11 = normxgstsg11freqs9.flatten(order='F')
lista_F10G11 = normxgstsg11freqs10.flatten(order='F')
lista_F11G11 = normxgstsg11freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G11 = mean(lista_F1G11)
Promedio_lista_F2G11 = mean(lista_F2G11)
Promedio_lista_F3G11 = mean(lista_F3G11)
Promedio_lista_F4G11 = mean(lista_F4G11)
Promedio_lista_F5G11 = mean(lista_F5G11)
Promedio_lista_F6G11 = mean(lista_F6G11)
Promedio_lista_F7G11 = mean(lista_F7G11)
Promedio_lista_F8G11 = mean(lista_F8G11)
Promedio_lista_F9G11 = mean(lista_F9G11)
Promedio_lista_F10G11 = mean(lista_F10G11)
Promedio_lista_F11G11 = mean(lista_F11G11)
#Lista de Promedios
PromediosG11 = []
PromediosG11.append(Promedio_lista_F1G11)
PromediosG11.append(Promedio_lista_F2G11)
PromediosG11.append(Promedio_lista_F3G11)
PromediosG11.append(Promedio_lista_F4G11)
PromediosG11.append(Promedio_lista_F5G11)
PromediosG11.append(Promedio_lista_F6G11)
PromediosG11.append(Promedio_lista_F7G11)
PromediosG11.append(Promedio_lista_F8G11)
PromediosG11.append(Promedio_lista_F9G11)
PromediosG11.append(Promedio_lista_F10G11)
PromediosG11.append(Promedio_lista_F11G11)
 
#Segmentación G12
normxgstsg12freqs1 = normxgsts[3800:4000,61999:63679]
normxgstsg12freqs2 = normxgsts[3600:3800,61999:63679]
normxgstsg12freqs3 = normxgsts[3400:3600,61999:63679]
normxgstsg12freqs4 = normxgsts[3200:3400,61999:63679]
normxgstsg12freqs5 = normxgsts[3000:3200,61999:63679]
normxgstsg12freqs6 = normxgsts[2800:3000,61999:63679]
normxgstsg12freqs7 = normxgsts[2600:2800,61999:63679]
normxgstsg12freqs8 = normxgsts[2400:2600,61999:63679]
normxgstsg12freqs9 = normxgsts[2200:2400,61999:63679]
normxgstsg12freqs10 = normxgsts[2000:2200,61999:63679]
normxgstsg12freqs11 = normxgsts[1800:2000,61999:63679]
#Transformación matrices de NumPy en vectores por columna
lista_F1G12 = normxgstsg12freqs1.flatten(order='F')
lista_F2G12 = normxgstsg12freqs2.flatten(order='F')
lista_F3G12 = normxgstsg12freqs3.flatten(order='F')
lista_F4G12 = normxgstsg12freqs4.flatten(order='F')
lista_F5G12 = normxgstsg12freqs5.flatten(order='F')
lista_F6G12 = normxgstsg12freqs6.flatten(order='F')
lista_F7G12 = normxgstsg12freqs7.flatten(order='F')
lista_F8G12 = normxgstsg12freqs8.flatten(order='F')
lista_F9G12 = normxgstsg12freqs9.flatten(order='F')
lista_F10G12 = normxgstsg12freqs10.flatten(order='F')
lista_F11G12 = normxgstsg12freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G12 = mean(lista_F1G12)
Promedio_lista_F2G12 = mean(lista_F2G12)
Promedio_lista_F3G12 = mean(lista_F3G12)
Promedio_lista_F4G12 = mean(lista_F4G12)
Promedio_lista_F5G12 = mean(lista_F5G12)
Promedio_lista_F6G12 = mean(lista_F6G12)
Promedio_lista_F7G12 = mean(lista_F7G12)
Promedio_lista_F8G12 = mean(lista_F8G12)
Promedio_lista_F9G12 = mean(lista_F9G12)
Promedio_lista_F10G12 = mean(lista_F10G12)
Promedio_lista_F11G12 = mean(lista_F11G12)
#Lista de Promedios
PromediosG12 = []
PromediosG12.append(Promedio_lista_F1G12)
PromediosG12.append(Promedio_lista_F2G12)
PromediosG12.append(Promedio_lista_F3G12)
PromediosG12.append(Promedio_lista_F4G12)
PromediosG12.append(Promedio_lista_F5G12)
PromediosG12.append(Promedio_lista_F6G12)
PromediosG12.append(Promedio_lista_F7G12)
PromediosG12.append(Promedio_lista_F8G12)
PromediosG12.append(Promedio_lista_F9G12)
PromediosG12.append(Promedio_lista_F10G12)
PromediosG12.append(Promedio_lista_F11G12)

Promedios = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6, 
             PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]
PromediosFrecuencias = np.array(Promedios); 
PromediosFrecuencias = np.transpose(PromediosFrecuencias)

DF = pd.DataFrame(PromediosFrecuencias)
#DF.to_csv("Promedios Paciente 6 Canal 8.csv")

PromediosG1 = np.array(PromediosG1)
PromediosG1 = np.transpose(PromediosG1)

PromediosG2 = np.array(PromediosG2)
PromediosG2 = np.transpose(PromediosG2)

PromediosG3 = np.array(PromediosG3)
PromediosG3 = np.transpose(PromediosG3)

PromediosG4 = np.array(PromediosG4)
PromediosG4 = np.transpose(PromediosG4)

PromediosG5 = np.array(PromediosG5)
PromediosG5 = np.transpose(PromediosG5)

PromediosG6 = np.array(PromediosG6)
PromediosG6 = np.transpose(PromediosG6)

PromediosG7 = np.array(PromediosG7)
PromediosG7 = np.transpose(PromediosG7)

PromediosG8 = np.array(PromediosG8)
PromediosG8 = np.transpose(PromediosG8)

PromediosG9 = np.array(PromediosG9)
PromediosG9 = np.transpose(PromediosG9)

PromediosG10 = np.array(PromediosG10)
PromediosG10 = np.transpose(PromediosG10)

PromediosG11 = np.array(PromediosG11)
PromediosG11 = np.transpose(PromediosG11)

PromediosG12 = np.array(PromediosG12)
PromediosG12 = np.transpose(PromediosG12)

data = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6,
        PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]

#Secuencia Consecutiva de Matrices 
SPromediosG1 = list(itertools.chain(PromediosG1, PromediosG7))
SPromediosG1 = np.array(SPromediosG1)
SPromediosG2 = list(itertools.chain(PromediosG2, PromediosG8))
SPromediosG2 = np.array(SPromediosG2)
SPromediosG3 = list(itertools.chain(PromediosG3, PromediosG9))
SPromediosG3 = np.array(SPromediosG3)
SPromediosG4 = list(itertools.chain(PromediosG4, PromediosG10))
SPromediosG4 = np.array(SPromediosG4)
SPromediosG5 = list(itertools.chain(PromediosG5, PromediosG11))
SPromediosG5 = np.array(SPromediosG5)
SPromediosG6 = list(itertools.chain(PromediosG6, PromediosG12))
SPromediosG6 = np.array(SPromediosG6)
SPromedios = [SPromediosG1, SPromediosG2, SPromediosG3, SPromediosG4, SPromediosG5, SPromediosG6]

SPromediosU0 = list(itertools.chain(SPromedios[0], SPromedios[1]))
SPromediosU1 = list(itertools.chain(SPromediosU0, SPromedios[2]))
SPromediosU2 = list(itertools.chain(SPromediosU1, SPromedios[3]))
SPromediosU3 = list(itertools.chain(SPromediosU2, SPromedios[4]))
SPromediosU4 = list(itertools.chain(SPromediosU3, SPromedios[5]))
SPromediosU = np.array(SPromediosU4)

#Histograma
plt.figure(figsize =(16, 10))
plt.hist(SPromediosU)
plt.grid()
plt.title('Histograma de la Secuencia Consecutiva de Promedios')
plt.xlabel('Valor de la variable')
plt.ylabel('Conteo')
plt.savefig('Histograma Promedios Gesto 1 P3C6.png', dpi = 201)
plt.show()

#scipy stats Wilcoxon 
WCresultados = []
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[1], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[2], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)


stat, p_value = wilcoxon(SPromedios[3], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[4], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[5], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)

WCresultados = np.array(WCresultados)
WCresultados = np.reshape(WCresultados,[6,6])

DF = pd.DataFrame(WCresultados) 
DF.to_csv("Wilcoxon Resultados entre gestos distintos P6C8.csv")

emg_funcional = np.array(voltajesP07[4])
#Lectura archivo 
samplerate = 1e3 #Frecuencia de muestreo
length = emg_funcional.shape[0] / samplerate #Duración total Electromiografía en segundos
#print("Duración Total = {length}s")

#Parametros CWT 
num = 4000 #Número de sampleo generado
wavelet = 'cmor1.0-1.5' # Bugs Fixed https://pywavelets.readthedocs.io/en/latest/release.1.1.0.html 
maxesc = np.log2(1500)
widths = np.logspace(0,maxesc, num=num, base=2,dtype='int')
#print("Escalas Usadas: ", widths)
dt = 1/samplerate # diferencia de intervalos de tiempo
frequencies = pywt.scale2frequency(wavelet, widths) / dt # Obtención de frecuencias correspondientes a escalas Modificar aquí precisi´´on
#print("Freqs asociadas a Escalas: ", frequencies)
# Calculo transformación continua de ondículas de los anchos de matriz numpy
wavelet_coeffs, freqs = pywt.cwt(emg_funcional, widths, wavelet=wavelet, sampling_period=samplerate, method='fft')
#print("Tamaño de wavelet transform: ", wavelet_coeffs.shape)
#print("Freqs asociadas a wavelet transform: ", freqs)

#Valores Absolutos de la CWT
abscwt = abs(wavelet_coeffs) # Devolución de ABS de los números complejos es la magnitud
gesto_1 = abscwt[:,0:3882]
gesto_2 = abscwt[:,6167:8161]
gesto_3 = abscwt[:,12656:14942]
gesto_4 = abscwt[:,20753:23405]
gesto_5 = abscwt[:,27657:30644]
gesto_6 = abscwt[:,34211:36657]
gesto_7 = abscwt[:,39280:41151]
gesto_8 = abscwt[:,43230:45592]
gesto_9 = abscwt[:,49301:52048]
gesto_10 = abscwt[:,57023:59580]
gesto_11 = abscwt[:,63843:66522]
gesto_12 = abscwt[:,70997:73703]

normxgsts = abscwt
#División de segmentos desde 0 a 57,69 Hz

#Segmentación G1
normxgstsg1freqs1 = normxgsts[3800:4000,0:3883]
normxgstsg1freqs2 = normxgsts[3600:3800,0:3883]
normxgstsg1freqs3 = normxgsts[3400:3600,0:3883]
normxgstsg1freqs4 = normxgsts[3200:3400,0:3883]
normxgstsg1freqs5 = normxgsts[3000:3200,0:3883]
normxgstsg1freqs6 = normxgsts[2800:3000,0:3883]
normxgstsg1freqs7 = normxgsts[2600:2800,0:3883]
normxgstsg1freqs8 = normxgsts[2400:2600,0:3883]
normxgstsg1freqs9 = normxgsts[2200:2400,0:3883]
normxgstsg1freqs10 = normxgsts[2000:2200,0:3883]
normxgstsg1freqs11 = normxgsts[1800:2000,0:3883]
#Transformación matrices de NumPy en vectores por columna
lista_F1G1 = normxgstsg1freqs1.flatten(order='F')
lista_F2G1 = normxgstsg1freqs2.flatten(order='F')
lista_F3G1 = normxgstsg1freqs3.flatten(order='F')
lista_F4G1 = normxgstsg1freqs4.flatten(order='F')
lista_F5G1 = normxgstsg1freqs5.flatten(order='F')
lista_F6G1 = normxgstsg1freqs6.flatten(order='F')
lista_F7G1 = normxgstsg1freqs7.flatten(order='F')
lista_F8G1 = normxgstsg1freqs8.flatten(order='F')
lista_F9G1 = normxgstsg1freqs9.flatten(order='F')
lista_F10G1 = normxgstsg1freqs10.flatten(order='F')
lista_F11G1 = normxgstsg1freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G1 = mean(lista_F1G1)
Promedio_lista_F2G1 = mean(lista_F2G1)
Promedio_lista_F3G1 = mean(lista_F3G1)
Promedio_lista_F4G1 = mean(lista_F4G1)
Promedio_lista_F5G1 = mean(lista_F5G1)
Promedio_lista_F6G1 = mean(lista_F6G1)
Promedio_lista_F7G1 = mean(lista_F7G1)
Promedio_lista_F8G1 = mean(lista_F8G1)
Promedio_lista_F9G1 = mean(lista_F9G1)
Promedio_lista_F10G1 = mean(lista_F10G1)
Promedio_lista_F11G1 = mean(lista_F11G1)
#Lista de Promedios
PromediosG1 = []
PromediosG1.append(Promedio_lista_F1G1)
PromediosG1.append(Promedio_lista_F2G1)
PromediosG1.append(Promedio_lista_F3G1)
PromediosG1.append(Promedio_lista_F4G1)
PromediosG1.append(Promedio_lista_F5G1)
PromediosG1.append(Promedio_lista_F6G1)
PromediosG1.append(Promedio_lista_F7G1)
PromediosG1.append(Promedio_lista_F8G1)
PromediosG1.append(Promedio_lista_F9G1)
PromediosG1.append(Promedio_lista_F10G1)
PromediosG1.append(Promedio_lista_F11G1)
 
#Segmentación G2
normxgstsg2freqs1 = normxgsts[3800:4000,6166:8162]
normxgstsg2freqs2 = normxgsts[3600:3800,6166:8162]
normxgstsg2freqs3 = normxgsts[3400:3600,6166:8162]
normxgstsg2freqs4 = normxgsts[3200:3400,6166:8162]
normxgstsg2freqs5 = normxgsts[3000:3200,6166:8162]
normxgstsg2freqs6 = normxgsts[2800:3000,6166:8162]
normxgstsg2freqs7 = normxgsts[2600:2800,6166:8162]
normxgstsg2freqs8 = normxgsts[2400:2600,6166:8162]
normxgstsg2freqs9 = normxgsts[2200:2400,6166:8162]
normxgstsg2freqs10 = normxgsts[2000:2200,6166:8162]
normxgstsg2freqs11 = normxgsts[1800:2000,6166:8162]
#Transformación matrices de NumPy en vectores por columna
lista_F1G2 = normxgstsg2freqs1.flatten(order='F')
lista_F2G2 = normxgstsg2freqs2.flatten(order='F')
lista_F3G2 = normxgstsg2freqs3.flatten(order='F')
lista_F4G2 = normxgstsg2freqs4.flatten(order='F')
lista_F5G2 = normxgstsg2freqs5.flatten(order='F')
lista_F6G2 = normxgstsg2freqs6.flatten(order='F')
lista_F7G2 = normxgstsg2freqs7.flatten(order='F')
lista_F8G2 = normxgstsg2freqs8.flatten(order='F')
lista_F9G2 = normxgstsg2freqs9.flatten(order='F')
lista_F10G2 = normxgstsg2freqs10.flatten(order='F')
lista_F11G2 = normxgstsg2freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G2 = mean(lista_F1G2)
Promedio_lista_F2G2 = mean(lista_F2G2)
Promedio_lista_F3G2 = mean(lista_F3G2)
Promedio_lista_F4G2 = mean(lista_F4G2)
Promedio_lista_F5G2 = mean(lista_F5G2)
Promedio_lista_F6G2 = mean(lista_F6G2)
Promedio_lista_F7G2 = mean(lista_F7G2)
Promedio_lista_F8G2 = mean(lista_F8G2)
Promedio_lista_F9G2 = mean(lista_F9G2)
Promedio_lista_F10G2 = mean(lista_F10G2)
Promedio_lista_F11G2 = mean(lista_F11G2)
#Lista de Promedios
PromediosG2 = []
PromediosG2.append(Promedio_lista_F1G2)
PromediosG2.append(Promedio_lista_F2G2)
PromediosG2.append(Promedio_lista_F3G2)
PromediosG2.append(Promedio_lista_F4G2)
PromediosG2.append(Promedio_lista_F5G2)
PromediosG2.append(Promedio_lista_F6G2)
PromediosG2.append(Promedio_lista_F7G2)
PromediosG2.append(Promedio_lista_F8G2)
PromediosG2.append(Promedio_lista_F9G2)
PromediosG2.append(Promedio_lista_F10G2)
PromediosG2.append(Promedio_lista_F11G2)
 
#Segmentación G3
normxgstsg3freqs1 = normxgsts[3800:4000,12655:14943]
normxgstsg3freqs2 = normxgsts[3600:3800,12655:14943]
normxgstsg3freqs3 = normxgsts[3400:3600,12655:14943]
normxgstsg3freqs4 = normxgsts[3200:3400,12655:14943]
normxgstsg3freqs5 = normxgsts[3000:3200,12655:14943]
normxgstsg3freqs6 = normxgsts[2800:3000,12655:14943]
normxgstsg3freqs7 = normxgsts[2600:2800,12655:14943]
normxgstsg3freqs8 = normxgsts[2400:2600,12655:14943]
normxgstsg3freqs9 = normxgsts[2200:2400,12655:14943]
normxgstsg3freqs10 = normxgsts[2000:2200,12655:14943]
normxgstsg3freqs11 = normxgsts[1800:2000,12655:14943]
#Transformación matrices de NumPy en vectores por columna
lista_F1G3 = normxgstsg3freqs1.flatten(order='F')
lista_F2G3 = normxgstsg3freqs2.flatten(order='F')
lista_F3G3 = normxgstsg3freqs3.flatten(order='F')
lista_F4G3 = normxgstsg3freqs4.flatten(order='F')
lista_F5G3 = normxgstsg3freqs5.flatten(order='F')
lista_F6G3 = normxgstsg3freqs6.flatten(order='F')
lista_F7G3 = normxgstsg3freqs7.flatten(order='F')
lista_F8G3 = normxgstsg3freqs8.flatten(order='F')
lista_F9G3 = normxgstsg3freqs9.flatten(order='F')
lista_F10G3 = normxgstsg3freqs10.flatten(order='F')
lista_F11G3 = normxgstsg3freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G3 = mean(lista_F1G3)
Promedio_lista_F2G3 = mean(lista_F2G3)
Promedio_lista_F3G3 = mean(lista_F3G3)
Promedio_lista_F4G3 = mean(lista_F4G3)
Promedio_lista_F5G3 = mean(lista_F5G3)
Promedio_lista_F6G3 = mean(lista_F6G3)
Promedio_lista_F7G3 = mean(lista_F7G3)
Promedio_lista_F8G3 = mean(lista_F8G3)
Promedio_lista_F9G3 = mean(lista_F9G3)
Promedio_lista_F10G3 = mean(lista_F10G3)
Promedio_lista_F11G3 = mean(lista_F11G3)
#Lista de Promedios
PromediosG3 = []
PromediosG3.append(Promedio_lista_F1G3)
PromediosG3.append(Promedio_lista_F2G3)
PromediosG3.append(Promedio_lista_F3G3)
PromediosG3.append(Promedio_lista_F4G3)
PromediosG3.append(Promedio_lista_F5G3)
PromediosG3.append(Promedio_lista_F6G3)
PromediosG3.append(Promedio_lista_F7G3)
PromediosG3.append(Promedio_lista_F8G3)
PromediosG3.append(Promedio_lista_F9G3)
PromediosG3.append(Promedio_lista_F10G3)
PromediosG3.append(Promedio_lista_F11G3)  
 
#Segmentación G4
normxgstsg4freqs1 = normxgsts[3800:4000,20752:23406]
normxgstsg4freqs2 = normxgsts[3600:3800,20752:23406]
normxgstsg4freqs3 = normxgsts[3400:3600,20752:23406]
normxgstsg4freqs4 = normxgsts[3200:3400,20752:23406]
normxgstsg4freqs5 = normxgsts[3000:3200,20752:23406]
normxgstsg4freqs6 = normxgsts[2800:3000,20752:23406]
normxgstsg4freqs7 = normxgsts[2600:2800,20752:23406]
normxgstsg4freqs8 = normxgsts[2400:2600,20752:23406]
normxgstsg4freqs9 = normxgsts[2200:2400,20752:23406]
normxgstsg4freqs10 = normxgsts[2000:2200,20752:23406]
normxgstsg4freqs11 = normxgsts[1800:2000,20752:23406]
#Transformación matrices de NumPy en vectores por columna
lista_F1G4 = normxgstsg4freqs1.flatten(order='F')
lista_F2G4 = normxgstsg4freqs2.flatten(order='F')
lista_F3G4 = normxgstsg4freqs3.flatten(order='F')
lista_F4G4 = normxgstsg4freqs4.flatten(order='F')
lista_F5G4 = normxgstsg4freqs5.flatten(order='F')
lista_F6G4 = normxgstsg4freqs6.flatten(order='F')
lista_F7G4 = normxgstsg4freqs7.flatten(order='F')
lista_F8G4 = normxgstsg4freqs8.flatten(order='F')
lista_F9G4 = normxgstsg4freqs9.flatten(order='F')
lista_F10G4 = normxgstsg4freqs10.flatten(order='F')
lista_F11G4 = normxgstsg4freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G4 = mean(lista_F1G4)
Promedio_lista_F2G4 = mean(lista_F2G4)
Promedio_lista_F3G4 = mean(lista_F3G4)
Promedio_lista_F4G4 = mean(lista_F4G4)
Promedio_lista_F5G4 = mean(lista_F5G4)
Promedio_lista_F6G4 = mean(lista_F6G4)
Promedio_lista_F7G4 = mean(lista_F7G4)
Promedio_lista_F8G4 = mean(lista_F8G4)
Promedio_lista_F9G4 = mean(lista_F9G4)
Promedio_lista_F10G4 = mean(lista_F10G4)
Promedio_lista_F11G4 = mean(lista_F11G4)
#Lista de Promedios
PromediosG4 = []
PromediosG4.append(Promedio_lista_F1G4)
PromediosG4.append(Promedio_lista_F2G4)
PromediosG4.append(Promedio_lista_F3G4)
PromediosG4.append(Promedio_lista_F4G4)
PromediosG4.append(Promedio_lista_F5G4)
PromediosG4.append(Promedio_lista_F6G4)
PromediosG4.append(Promedio_lista_F7G4)
PromediosG4.append(Promedio_lista_F8G4)
PromediosG4.append(Promedio_lista_F9G4)
PromediosG4.append(Promedio_lista_F10G4)
PromediosG4.append(Promedio_lista_F11G4)      
 
#Segmentación G5
normxgstsg5freqs1 = normxgsts[3800:4000,27656:30645]
normxgstsg5freqs2 = normxgsts[3600:3800,27656:30645]
normxgstsg5freqs3 = normxgsts[3400:3600,27656:30645]
normxgstsg5freqs4 = normxgsts[3200:3400,27656:30645]
normxgstsg5freqs5 = normxgsts[3000:3200,27656:30645]
normxgstsg5freqs6 = normxgsts[2800:3000,27656:30645]
normxgstsg5freqs7 = normxgsts[2600:2800,27656:30645]
normxgstsg5freqs8 = normxgsts[2400:2600,27656:30645]
normxgstsg5freqs9 = normxgsts[2200:2400,27656:30645]
normxgstsg5freqs10 = normxgsts[2000:2200,27656:30645]
normxgstsg5freqs11 = normxgsts[1800:2000,27656:30645]
#Transformación matrices de NumPy en vectores por columna
lista_F1G5 = normxgstsg5freqs1.flatten(order='F')
lista_F2G5 = normxgstsg5freqs2.flatten(order='F')
lista_F3G5 = normxgstsg5freqs3.flatten(order='F')
lista_F4G5 = normxgstsg5freqs4.flatten(order='F')
lista_F5G5 = normxgstsg5freqs5.flatten(order='F')
lista_F6G5 = normxgstsg5freqs6.flatten(order='F')
lista_F7G5 = normxgstsg5freqs7.flatten(order='F')
lista_F8G5 = normxgstsg5freqs8.flatten(order='F')
lista_F9G5 = normxgstsg5freqs9.flatten(order='F')
lista_F10G5 = normxgstsg5freqs10.flatten(order='F')
lista_F11G5 = normxgstsg5freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G5 = mean(lista_F1G5)
Promedio_lista_F2G5 = mean(lista_F2G5)
Promedio_lista_F3G5 = mean(lista_F3G5)
Promedio_lista_F4G5 = mean(lista_F4G5)
Promedio_lista_F5G5 = mean(lista_F5G5)
Promedio_lista_F6G5 = mean(lista_F6G5)
Promedio_lista_F7G5 = mean(lista_F7G5)
Promedio_lista_F8G5 = mean(lista_F8G5)
Promedio_lista_F9G5 = mean(lista_F9G5)
Promedio_lista_F10G5 = mean(lista_F10G5)
Promedio_lista_F11G5 = mean(lista_F11G5)
#Lista de Promedios
PromediosG5 = []
PromediosG5.append(Promedio_lista_F1G5)
PromediosG5.append(Promedio_lista_F2G5)
PromediosG5.append(Promedio_lista_F3G5)
PromediosG5.append(Promedio_lista_F4G5)
PromediosG5.append(Promedio_lista_F5G5)
PromediosG5.append(Promedio_lista_F6G5)
PromediosG5.append(Promedio_lista_F7G5)
PromediosG5.append(Promedio_lista_F8G5)
PromediosG5.append(Promedio_lista_F9G5)
PromediosG5.append(Promedio_lista_F10G5)
PromediosG5.append(Promedio_lista_F11G5)
 
#Segmentación G6
normxgstsg6freqs1 = normxgsts[3800:4000,34210:36658]
normxgstsg6freqs2 = normxgsts[3600:3800,34210:36658]
normxgstsg6freqs3 = normxgsts[3400:3600,34210:36658]
normxgstsg6freqs4 = normxgsts[3200:3400,34210:36658]
normxgstsg6freqs5 = normxgsts[3000:3200,34210:36658]
normxgstsg6freqs6 = normxgsts[2800:3000,34210:36658]
normxgstsg6freqs7 = normxgsts[2600:2800,34210:36658]
normxgstsg6freqs8 = normxgsts[2400:2600,34210:36658]
normxgstsg6freqs9 = normxgsts[2200:2400,34210:36658]
normxgstsg6freqs10 = normxgsts[2000:2200,34210:36658]
normxgstsg6freqs11 = normxgsts[1800:2000,34210:36658]
#Transformación matrices de NumPy en vectores por columna
lista_F1G6 = normxgstsg6freqs1.flatten(order='F')
lista_F2G6 = normxgstsg6freqs2.flatten(order='F')
lista_F3G6 = normxgstsg6freqs3.flatten(order='F')
lista_F4G6 = normxgstsg6freqs4.flatten(order='F')
lista_F5G6 = normxgstsg6freqs5.flatten(order='F')
lista_F6G6 = normxgstsg6freqs6.flatten(order='F')
lista_F7G6 = normxgstsg6freqs7.flatten(order='F')
lista_F8G6 = normxgstsg6freqs8.flatten(order='F')
lista_F9G6 = normxgstsg6freqs9.flatten(order='F')
lista_F10G6 = normxgstsg6freqs10.flatten(order='F')
lista_F11G6 = normxgstsg6freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G6 = mean(lista_F1G6)
Promedio_lista_F2G6 = mean(lista_F2G6)
Promedio_lista_F3G6 = mean(lista_F3G6)
Promedio_lista_F4G6 = mean(lista_F4G6)
Promedio_lista_F5G6 = mean(lista_F5G6)
Promedio_lista_F6G6 = mean(lista_F6G6)
Promedio_lista_F7G6 = mean(lista_F7G6)
Promedio_lista_F8G6 = mean(lista_F8G6)
Promedio_lista_F9G6 = mean(lista_F9G6)
Promedio_lista_F10G6 = mean(lista_F10G6)
Promedio_lista_F11G6 = mean(lista_F11G6)
#Lista de Promedios
PromediosG6 = []
PromediosG6.append(Promedio_lista_F1G6)
PromediosG6.append(Promedio_lista_F2G6)
PromediosG6.append(Promedio_lista_F3G6)
PromediosG6.append(Promedio_lista_F4G6)
PromediosG6.append(Promedio_lista_F5G6)
PromediosG6.append(Promedio_lista_F6G6)
PromediosG6.append(Promedio_lista_F7G6)
PromediosG6.append(Promedio_lista_F8G6)
PromediosG6.append(Promedio_lista_F9G6)
PromediosG6.append(Promedio_lista_F10G6)
PromediosG6.append(Promedio_lista_F11G6)
 
#Segmentación G7
normxgstsg7freqs1 = normxgsts[3800:4000,39279:41152]
normxgstsg7freqs2 = normxgsts[3600:3800,39279:41152]
normxgstsg7freqs3 = normxgsts[3400:3600,39279:41152]
normxgstsg7freqs4 = normxgsts[3200:3400,39279:41152]
normxgstsg7freqs5 = normxgsts[3000:3200,39279:41152]
normxgstsg7freqs6 = normxgsts[2800:3000,39279:41152]
normxgstsg7freqs7 = normxgsts[2600:2800,39279:41152]
normxgstsg7freqs8 = normxgsts[2400:2600,39279:41152]
normxgstsg7freqs9 = normxgsts[2200:2400,39279:41152]
normxgstsg7freqs10 = normxgsts[2000:2200,39279:41152]
normxgstsg7freqs11 = normxgsts[1800:2000,39279:41152]
#Transformación matrices de NumPy en vectores por columna
lista_F1G7 = normxgstsg7freqs1.flatten(order='F')
lista_F2G7 = normxgstsg7freqs2.flatten(order='F')
lista_F3G7 = normxgstsg7freqs3.flatten(order='F')
lista_F4G7 = normxgstsg7freqs4.flatten(order='F')
lista_F5G7 = normxgstsg7freqs5.flatten(order='F')
lista_F6G7 = normxgstsg7freqs6.flatten(order='F')
lista_F7G7 = normxgstsg7freqs7.flatten(order='F')
lista_F8G7 = normxgstsg7freqs8.flatten(order='F')
lista_F9G7 = normxgstsg7freqs9.flatten(order='F')
lista_F10G7 = normxgstsg7freqs10.flatten(order='F')
lista_F11G7 = normxgstsg7freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G7 = mean(lista_F1G7)
Promedio_lista_F2G7 = mean(lista_F2G7)
Promedio_lista_F3G7 = mean(lista_F3G7)
Promedio_lista_F4G7 = mean(lista_F4G7)
Promedio_lista_F5G7 = mean(lista_F5G7)
Promedio_lista_F6G7 = mean(lista_F6G7)
Promedio_lista_F7G7 = mean(lista_F7G7)
Promedio_lista_F8G7 = mean(lista_F8G7)
Promedio_lista_F9G7 = mean(lista_F9G7)
Promedio_lista_F10G7 = mean(lista_F10G7)
Promedio_lista_F11G7 = mean(lista_F11G7)
#Lista de Promedios
PromediosG7 = []
PromediosG7.append(Promedio_lista_F1G7)
PromediosG7.append(Promedio_lista_F2G7)
PromediosG7.append(Promedio_lista_F3G7)
PromediosG7.append(Promedio_lista_F4G7)
PromediosG7.append(Promedio_lista_F5G7)
PromediosG7.append(Promedio_lista_F6G7)
PromediosG7.append(Promedio_lista_F7G7)
PromediosG7.append(Promedio_lista_F8G7)
PromediosG7.append(Promedio_lista_F9G7)
PromediosG7.append(Promedio_lista_F10G7)
PromediosG7.append(Promedio_lista_F11G7)
 
#Segmentación G8
normxgstsg8freqs1 = normxgsts[3800:4000,43229:45593]
normxgstsg8freqs2 = normxgsts[3600:3800,43229:45593]
normxgstsg8freqs3 = normxgsts[3400:3600,43229:45593]
normxgstsg8freqs4 = normxgsts[3200:3400,43229:45593]
normxgstsg8freqs5 = normxgsts[3000:3200,43229:45593]
normxgstsg8freqs6 = normxgsts[2800:3000,43229:45593]
normxgstsg8freqs7 = normxgsts[2600:2800,43229:45593]
normxgstsg8freqs8 = normxgsts[2400:2600,43229:45593]
normxgstsg8freqs9 = normxgsts[2200:2400,43229:45593]
normxgstsg8freqs10 = normxgsts[2000:2200,43229:45593]
normxgstsg8freqs11 = normxgsts[1800:2000,43229:45593]
#Transformación matrices de NumPy en vectores por columna
lista_F1G8 = normxgstsg8freqs1.flatten(order='F')
lista_F2G8 = normxgstsg8freqs2.flatten(order='F')
lista_F3G8 = normxgstsg8freqs3.flatten(order='F')
lista_F4G8 = normxgstsg8freqs4.flatten(order='F')
lista_F5G8 = normxgstsg8freqs5.flatten(order='F')
lista_F6G8 = normxgstsg8freqs6.flatten(order='F')
lista_F7G8 = normxgstsg8freqs7.flatten(order='F')
lista_F8G8 = normxgstsg8freqs8.flatten(order='F')
lista_F9G8 = normxgstsg8freqs9.flatten(order='F')
lista_F10G8 = normxgstsg8freqs10.flatten(order='F')
lista_F11G8 = normxgstsg8freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G8 = mean(lista_F1G8)
Promedio_lista_F2G8 = mean(lista_F2G8)
Promedio_lista_F3G8 = mean(lista_F3G8)
Promedio_lista_F4G8 = mean(lista_F4G8)
Promedio_lista_F5G8 = mean(lista_F5G8)
Promedio_lista_F6G8 = mean(lista_F6G8)
Promedio_lista_F7G8 = mean(lista_F7G8)
Promedio_lista_F8G8 = mean(lista_F8G8)
Promedio_lista_F9G8 = mean(lista_F9G8)
Promedio_lista_F10G8 = mean(lista_F10G8)
Promedio_lista_F11G8 = mean(lista_F11G8)
#Lista de Promedios
PromediosG8 = []
PromediosG8.append(Promedio_lista_F1G8)
PromediosG8.append(Promedio_lista_F2G8)
PromediosG8.append(Promedio_lista_F3G8)
PromediosG8.append(Promedio_lista_F4G8)
PromediosG8.append(Promedio_lista_F5G8)
PromediosG8.append(Promedio_lista_F6G8)
PromediosG8.append(Promedio_lista_F7G8)
PromediosG8.append(Promedio_lista_F8G8)
PromediosG8.append(Promedio_lista_F9G8)
PromediosG8.append(Promedio_lista_F10G8)
PromediosG8.append(Promedio_lista_F11G8)
 
#Segmentación G9
normxgstsg9freqs1 = normxgsts[3800:4000,49300:52049]
normxgstsg9freqs2 = normxgsts[3600:3800,49300:52049]
normxgstsg9freqs3 = normxgsts[3400:3600,49300:52049]
normxgstsg9freqs4 = normxgsts[3200:3400,49300:52049]
normxgstsg9freqs5 = normxgsts[3000:3200,49300:52049]
normxgstsg9freqs6 = normxgsts[2800:3000,49300:52049]
normxgstsg9freqs7 = normxgsts[2600:2800,49300:52049]
normxgstsg9freqs8 = normxgsts[2400:2600,49300:52049]
normxgstsg9freqs9 = normxgsts[2200:2400,49300:52049]
normxgstsg9freqs10 = normxgsts[2000:2200,49300:52049]
normxgstsg9freqs11 = normxgsts[1800:2000,49300:52049]
#Transformación matrices de NumPy en vectores por columna
lista_F1G9 = normxgstsg9freqs1.flatten(order='F')
lista_F2G9 = normxgstsg9freqs2.flatten(order='F')
lista_F3G9 = normxgstsg9freqs3.flatten(order='F')
lista_F4G9 = normxgstsg9freqs4.flatten(order='F')
lista_F5G9 = normxgstsg9freqs5.flatten(order='F')
lista_F6G9 = normxgstsg9freqs6.flatten(order='F')
lista_F7G9 = normxgstsg9freqs7.flatten(order='F')
lista_F8G9 = normxgstsg9freqs8.flatten(order='F')
lista_F9G9 = normxgstsg9freqs9.flatten(order='F')
lista_F10G9 = normxgstsg9freqs10.flatten(order='F')
lista_F11G9 = normxgstsg9freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G9 = mean(lista_F1G9)
Promedio_lista_F2G9 = mean(lista_F2G9)
Promedio_lista_F3G9 = mean(lista_F3G9)
Promedio_lista_F4G9 = mean(lista_F4G9)
Promedio_lista_F5G9 = mean(lista_F5G9)
Promedio_lista_F6G9 = mean(lista_F6G9)
Promedio_lista_F7G9 = mean(lista_F7G9)
Promedio_lista_F8G9 = mean(lista_F8G9)
Promedio_lista_F9G9 = mean(lista_F9G9)
Promedio_lista_F10G9 = mean(lista_F10G9)
Promedio_lista_F11G9 = mean(lista_F11G9)
#Lista de Promedios
PromediosG9 = []
PromediosG9.append(Promedio_lista_F1G9)
PromediosG9.append(Promedio_lista_F2G9)
PromediosG9.append(Promedio_lista_F3G9)
PromediosG9.append(Promedio_lista_F4G9)
PromediosG9.append(Promedio_lista_F5G9)
PromediosG9.append(Promedio_lista_F6G9)
PromediosG9.append(Promedio_lista_F7G9)
PromediosG9.append(Promedio_lista_F8G9)
PromediosG9.append(Promedio_lista_F9G9)
PromediosG9.append(Promedio_lista_F10G9)
PromediosG9.append(Promedio_lista_F11G9)
 
#Segmentación G10
normxgstsg10freqs1 = normxgsts[3800:4000,57022:59581]
normxgstsg10freqs2 = normxgsts[3600:3800,57022:59581]
normxgstsg10freqs3 = normxgsts[3400:3600,57022:59581]
normxgstsg10freqs4 = normxgsts[3200:3400,57022:59581]
normxgstsg10freqs5 = normxgsts[3000:3200,57022:59581]
normxgstsg10freqs6 = normxgsts[2800:3000,57022:59581]
normxgstsg10freqs7 = normxgsts[2600:2800,57022:59581]
normxgstsg10freqs8 = normxgsts[2400:2600,57022:59581]
normxgstsg10freqs9 = normxgsts[2200:2400,57022:59581]
normxgstsg10freqs10 = normxgsts[2000:2200,57022:59581]
normxgstsg10freqs11 = normxgsts[1800:2000,57022:59581]
#Transformación matrices de NumPy en vectores por columna
lista_F1G10 = normxgstsg10freqs1.flatten(order='F')
lista_F2G10 = normxgstsg10freqs2.flatten(order='F')
lista_F3G10 = normxgstsg10freqs3.flatten(order='F')
lista_F4G10 = normxgstsg10freqs4.flatten(order='F')
lista_F5G10 = normxgstsg10freqs5.flatten(order='F')
lista_F6G10 = normxgstsg10freqs6.flatten(order='F')
lista_F7G10 = normxgstsg10freqs7.flatten(order='F')
lista_F8G10 = normxgstsg10freqs8.flatten(order='F')
lista_F9G10 = normxgstsg10freqs9.flatten(order='F')
lista_F10G10 = normxgstsg10freqs10.flatten(order='F')
lista_F11G10 = normxgstsg10freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G10 = mean(lista_F1G10)
Promedio_lista_F2G10 = mean(lista_F2G10)
Promedio_lista_F3G10 = mean(lista_F3G10)
Promedio_lista_F4G10 = mean(lista_F4G10)
Promedio_lista_F5G10 = mean(lista_F5G10)
Promedio_lista_F6G10 = mean(lista_F6G10)
Promedio_lista_F7G10 = mean(lista_F7G10)
Promedio_lista_F8G10 = mean(lista_F8G10)
Promedio_lista_F9G10 = mean(lista_F9G10)
Promedio_lista_F10G10 = mean(lista_F10G10)
Promedio_lista_F11G10 = mean(lista_F11G10)
#Lista de Promedios
PromediosG10 = []
PromediosG10.append(Promedio_lista_F1G10)
PromediosG10.append(Promedio_lista_F2G10)
PromediosG10.append(Promedio_lista_F3G10)
PromediosG10.append(Promedio_lista_F4G10)
PromediosG10.append(Promedio_lista_F5G10)
PromediosG10.append(Promedio_lista_F6G10)
PromediosG10.append(Promedio_lista_F7G10)
PromediosG10.append(Promedio_lista_F8G10)
PromediosG10.append(Promedio_lista_F9G10)
PromediosG10.append(Promedio_lista_F10G10)
PromediosG10.append(Promedio_lista_F11G10)
 
#Segmentación G11
normxgstsg11freqs1 = normxgsts[3800:4000,63842:66523]
normxgstsg11freqs2 = normxgsts[3600:3800,63842:66523]
normxgstsg11freqs3 = normxgsts[3400:3600,63842:66523]
normxgstsg11freqs4 = normxgsts[3200:3400,63842:66523]
normxgstsg11freqs5 = normxgsts[3000:3200,63842:66523]
normxgstsg11freqs6 = normxgsts[2800:3000,63842:66523]
normxgstsg11freqs7 = normxgsts[2600:2800,63842:66523]
normxgstsg11freqs8 = normxgsts[2400:2600,63842:66523]
normxgstsg11freqs9 = normxgsts[2200:2400,63842:66523]
normxgstsg11freqs10 = normxgsts[2000:2200,63842:66523]
normxgstsg11freqs11 = normxgsts[1800:2000,63842:66523]
#Transformación matrices de NumPy en vectores por columna
lista_F1G11 = normxgstsg11freqs1.flatten(order='F')
lista_F2G11 = normxgstsg11freqs2.flatten(order='F')
lista_F3G11 = normxgstsg11freqs3.flatten(order='F')
lista_F4G11 = normxgstsg11freqs4.flatten(order='F')
lista_F5G11 = normxgstsg11freqs5.flatten(order='F')
lista_F6G11 = normxgstsg11freqs6.flatten(order='F')
lista_F7G11 = normxgstsg11freqs7.flatten(order='F')
lista_F8G11 = normxgstsg11freqs8.flatten(order='F')
lista_F9G11 = normxgstsg11freqs9.flatten(order='F')
lista_F10G11 = normxgstsg11freqs10.flatten(order='F')
lista_F11G11 = normxgstsg11freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G11 = mean(lista_F1G11)
Promedio_lista_F2G11 = mean(lista_F2G11)
Promedio_lista_F3G11 = mean(lista_F3G11)
Promedio_lista_F4G11 = mean(lista_F4G11)
Promedio_lista_F5G11 = mean(lista_F5G11)
Promedio_lista_F6G11 = mean(lista_F6G11)
Promedio_lista_F7G11 = mean(lista_F7G11)
Promedio_lista_F8G11 = mean(lista_F8G11)
Promedio_lista_F9G11 = mean(lista_F9G11)
Promedio_lista_F10G11 = mean(lista_F10G11)
Promedio_lista_F11G11 = mean(lista_F11G11)
#Lista de Promedios
PromediosG11 = []
PromediosG11.append(Promedio_lista_F1G11)
PromediosG11.append(Promedio_lista_F2G11)
PromediosG11.append(Promedio_lista_F3G11)
PromediosG11.append(Promedio_lista_F4G11)
PromediosG11.append(Promedio_lista_F5G11)
PromediosG11.append(Promedio_lista_F6G11)
PromediosG11.append(Promedio_lista_F7G11)
PromediosG11.append(Promedio_lista_F8G11)
PromediosG11.append(Promedio_lista_F9G11)
PromediosG11.append(Promedio_lista_F10G11)
PromediosG11.append(Promedio_lista_F11G11)
 
#Segmentación G12
normxgstsg12freqs1 = normxgsts[3800:4000,70996:73704]
normxgstsg12freqs2 = normxgsts[3600:3800,70996:73704]
normxgstsg12freqs3 = normxgsts[3400:3600,70996:73704]
normxgstsg12freqs4 = normxgsts[3200:3400,70996:73704]
normxgstsg12freqs5 = normxgsts[3000:3200,70996:73704]
normxgstsg12freqs6 = normxgsts[2800:3000,70996:73704]
normxgstsg12freqs7 = normxgsts[2600:2800,70996:73704]
normxgstsg12freqs8 = normxgsts[2400:2600,70996:73704]
normxgstsg12freqs9 = normxgsts[2200:2400,70996:73704]
normxgstsg12freqs10 = normxgsts[2000:2200,70996:73704]
normxgstsg12freqs11 = normxgsts[1800:2000,70996:73704]
#Transformación matrices de NumPy en vectores por columna
lista_F1G12 = normxgstsg12freqs1.flatten(order='F')
lista_F2G12 = normxgstsg12freqs2.flatten(order='F')
lista_F3G12 = normxgstsg12freqs3.flatten(order='F')
lista_F4G12 = normxgstsg12freqs4.flatten(order='F')
lista_F5G12 = normxgstsg12freqs5.flatten(order='F')
lista_F6G12 = normxgstsg12freqs6.flatten(order='F')
lista_F7G12 = normxgstsg12freqs7.flatten(order='F')
lista_F8G12 = normxgstsg12freqs8.flatten(order='F')
lista_F9G12 = normxgstsg12freqs9.flatten(order='F')
lista_F10G12 = normxgstsg12freqs10.flatten(order='F')
lista_F11G12 = normxgstsg12freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G12 = mean(lista_F1G12)
Promedio_lista_F2G12 = mean(lista_F2G12)
Promedio_lista_F3G12 = mean(lista_F3G12)
Promedio_lista_F4G12 = mean(lista_F4G12)
Promedio_lista_F5G12 = mean(lista_F5G12)
Promedio_lista_F6G12 = mean(lista_F6G12)
Promedio_lista_F7G12 = mean(lista_F7G12)
Promedio_lista_F8G12 = mean(lista_F8G12)
Promedio_lista_F9G12 = mean(lista_F9G12)
Promedio_lista_F10G12 = mean(lista_F10G12)
Promedio_lista_F11G12 = mean(lista_F11G12)
#Lista de Promedios
PromediosG12 = []
PromediosG12.append(Promedio_lista_F1G12)
PromediosG12.append(Promedio_lista_F2G12)
PromediosG12.append(Promedio_lista_F3G12)
PromediosG12.append(Promedio_lista_F4G12)
PromediosG12.append(Promedio_lista_F5G12)
PromediosG12.append(Promedio_lista_F6G12)
PromediosG12.append(Promedio_lista_F7G12)
PromediosG12.append(Promedio_lista_F8G12)
PromediosG12.append(Promedio_lista_F9G12)
PromediosG12.append(Promedio_lista_F10G12)
PromediosG12.append(Promedio_lista_F11G12)

Promedios = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6, 
             PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]
PromediosFrecuencias = np.array(Promedios); 
PromediosFrecuencias = np.transpose(PromediosFrecuencias)

DF = pd.DataFrame(PromediosFrecuencias)
#DF.to_csv("Promedios Paciente 7 Canal 5.csv")

PromediosG1 = np.array(PromediosG1)
PromediosG1 = np.transpose(PromediosG1)

PromediosG2 = np.array(PromediosG2)
PromediosG2 = np.transpose(PromediosG2)

PromediosG3 = np.array(PromediosG3)
PromediosG3 = np.transpose(PromediosG3)

PromediosG4 = np.array(PromediosG4)
PromediosG4 = np.transpose(PromediosG4)

PromediosG5 = np.array(PromediosG5)
PromediosG5 = np.transpose(PromediosG5)

PromediosG6 = np.array(PromediosG6)
PromediosG6 = np.transpose(PromediosG6)

PromediosG7 = np.array(PromediosG7)
PromediosG7 = np.transpose(PromediosG7)

PromediosG8 = np.array(PromediosG8)
PromediosG8 = np.transpose(PromediosG8)

PromediosG9 = np.array(PromediosG9)
PromediosG9 = np.transpose(PromediosG9)

PromediosG10 = np.array(PromediosG10)
PromediosG10 = np.transpose(PromediosG10)

PromediosG11 = np.array(PromediosG11)
PromediosG11 = np.transpose(PromediosG11)

PromediosG12 = np.array(PromediosG12)
PromediosG12 = np.transpose(PromediosG12)

data = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6,
        PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]

#Secuencia Consecutiva de Matrices 
SPromediosG1 = list(itertools.chain(PromediosG1, PromediosG7))
SPromediosG1 = np.array(SPromediosG1)
SPromediosG2 = list(itertools.chain(PromediosG2, PromediosG8))
SPromediosG2 = np.array(SPromediosG2)
SPromediosG3 = list(itertools.chain(PromediosG3, PromediosG9))
SPromediosG3 = np.array(SPromediosG3)
SPromediosG4 = list(itertools.chain(PromediosG4, PromediosG10))
SPromediosG4 = np.array(SPromediosG4)
SPromediosG5 = list(itertools.chain(PromediosG5, PromediosG11))
SPromediosG5 = np.array(SPromediosG5)
SPromediosG6 = list(itertools.chain(PromediosG6, PromediosG12))
SPromediosG6 = np.array(SPromediosG6)
SPromedios = [SPromediosG1, SPromediosG2, SPromediosG3, SPromediosG4, SPromediosG5, SPromediosG6]

SPromediosU0 = list(itertools.chain(SPromedios[0], SPromedios[1]))
SPromediosU1 = list(itertools.chain(SPromediosU0, SPromedios[2]))
SPromediosU2 = list(itertools.chain(SPromediosU1, SPromedios[3]))
SPromediosU3 = list(itertools.chain(SPromediosU2, SPromedios[4]))
SPromediosU4 = list(itertools.chain(SPromediosU3, SPromedios[5]))
SPromediosU = np.array(SPromediosU4)

#Histograma
plt.figure(figsize =(16, 10))
plt.hist(SPromediosU)
plt.grid()
plt.title('Histograma de la Secuencia Consecutiva de Promedios')
plt.xlabel('Valor de la variable')
plt.ylabel('Conteo')
plt.savefig('Histograma Promedios Gesto 1 P3C6.png', dpi = 201)
plt.show()

#scipy stats Wilcoxon 
WCresultados = []
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[1], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[2], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)


stat, p_value = wilcoxon(SPromedios[3], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[4], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[5], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)

WCresultados = np.array(WCresultados)
WCresultados = np.reshape(WCresultados,[6,6])

DF = pd.DataFrame(WCresultados) 
DF.to_csv("Wilcoxon Resultados entre gestos distintos P7C5.csv")

emg_funcional = np.array(voltajesP08[3])
#Lectura archivo 
samplerate = 1e3 #Frecuencia de muestreo
length = emg_funcional.shape[0] / samplerate #Duración total Electromiografía en segundos
#print("Duración Total = {length}s")

#Parametros CWT 
num = 4000 #Número de sampleo generado
wavelet = 'cmor1.0-1.5' # Bugs Fixed https://pywavelets.readthedocs.io/en/latest/release.1.1.0.html 
maxesc = np.log2(1500)
widths = np.logspace(0,maxesc, num=num, base=2,dtype='int')
#print("Escalas Usadas: ", widths)
dt = 1/samplerate # diferencia de intervalos de tiempo
frequencies = pywt.scale2frequency(wavelet, widths) / dt # Obtención de frecuencias correspondientes a escalas Modificar aquí precisi´´on
#print("Freqs asociadas a Escalas: ", frequencies)
# Calculo transformación continua de ondículas de los anchos de matriz numpy
wavelet_coeffs, freqs = pywt.cwt(emg_funcional, widths, wavelet=wavelet, sampling_period=samplerate, method='fft')
#print("Tamaño de wavelet transform: ", wavelet_coeffs.shape)
#print("Freqs asociadas a wavelet transform: ", freqs)

#Valores Absolutos de la CWT
abscwt = abs(wavelet_coeffs) # Devolución de ABS de los números complejos es la magnitud
gesto_1 = abscwt[:,1353:3231]
gesto_2 = abscwt[:,5097:6666]
gesto_3 = abscwt[:,11253:13064]
gesto_4 = abscwt[:,16259:17682]
gesto_5 = abscwt[:,20703:22186]
gesto_6 = abscwt[:,25257:26874]
gesto_7 = abscwt[:,29946:31247]
gesto_8 = abscwt[:,32785:34313]
gesto_9 = abscwt[:,37774:39098]
gesto_10 = abscwt[:,42421:44085]
gesto_11 = abscwt[:,46489:48174]
gesto_12 = abscwt[:,50529:52031]

normxgsts = abscwt
#División de segmentos desde 0 a 57,69 Hz

#Segmentación G1
normxgstsg1freqs1 = normxgsts[3800:4000,1352:3232]
normxgstsg1freqs2 = normxgsts[3600:3800,1352:3232]
normxgstsg1freqs3 = normxgsts[3400:3600,1352:3232]
normxgstsg1freqs4 = normxgsts[3200:3400,1352:3232]
normxgstsg1freqs5 = normxgsts[3000:3200,1352:3232]
normxgstsg1freqs6 = normxgsts[2800:3000,1352:3232]
normxgstsg1freqs7 = normxgsts[2600:2800,1352:3232]
normxgstsg1freqs8 = normxgsts[2400:2600,1352:3232]
normxgstsg1freqs9 = normxgsts[2200:2400,1352:3232]
normxgstsg1freqs10 = normxgsts[2000:2200,1352:3232]
normxgstsg1freqs11 = normxgsts[1800:2000,1352:3232]
#Transformación matrices de NumPy en vectores por columna
lista_F1G1 = normxgstsg1freqs1.flatten(order='F')
lista_F2G1 = normxgstsg1freqs2.flatten(order='F')
lista_F3G1 = normxgstsg1freqs3.flatten(order='F')
lista_F4G1 = normxgstsg1freqs4.flatten(order='F')
lista_F5G1 = normxgstsg1freqs5.flatten(order='F')
lista_F6G1 = normxgstsg1freqs6.flatten(order='F')
lista_F7G1 = normxgstsg1freqs7.flatten(order='F')
lista_F8G1 = normxgstsg1freqs8.flatten(order='F')
lista_F9G1 = normxgstsg1freqs9.flatten(order='F')
lista_F10G1 = normxgstsg1freqs10.flatten(order='F')
lista_F11G1 = normxgstsg1freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G1 = mean(lista_F1G1)
Promedio_lista_F2G1 = mean(lista_F2G1)
Promedio_lista_F3G1 = mean(lista_F3G1)
Promedio_lista_F4G1 = mean(lista_F4G1)
Promedio_lista_F5G1 = mean(lista_F5G1)
Promedio_lista_F6G1 = mean(lista_F6G1)
Promedio_lista_F7G1 = mean(lista_F7G1)
Promedio_lista_F8G1 = mean(lista_F8G1)
Promedio_lista_F9G1 = mean(lista_F9G1)
Promedio_lista_F10G1 = mean(lista_F10G1)
Promedio_lista_F11G1 = mean(lista_F11G1)
#Lista de Promedios
PromediosG1 = []
PromediosG1.append(Promedio_lista_F1G1)
PromediosG1.append(Promedio_lista_F2G1)
PromediosG1.append(Promedio_lista_F3G1)
PromediosG1.append(Promedio_lista_F4G1)
PromediosG1.append(Promedio_lista_F5G1)
PromediosG1.append(Promedio_lista_F6G1)
PromediosG1.append(Promedio_lista_F7G1)
PromediosG1.append(Promedio_lista_F8G1)
PromediosG1.append(Promedio_lista_F9G1)
PromediosG1.append(Promedio_lista_F10G1)
PromediosG1.append(Promedio_lista_F11G1)

#Segmentación G2
normxgstsg2freqs1 = normxgsts[3800:4000,5096:6667]
normxgstsg2freqs2 = normxgsts[3600:3800,5096:6667]
normxgstsg2freqs3 = normxgsts[3400:3600,5096:6667]
normxgstsg2freqs4 = normxgsts[3200:3400,5096:6667]
normxgstsg2freqs5 = normxgsts[3000:3200,5096:6667]
normxgstsg2freqs6 = normxgsts[2800:3000,5096:6667]
normxgstsg2freqs7 = normxgsts[2600:2800,5096:6667]
normxgstsg2freqs8 = normxgsts[2400:2600,5096:6667]
normxgstsg2freqs9 = normxgsts[2200:2400,5096:6667]
normxgstsg2freqs10 = normxgsts[2000:2200,5096:6667]
normxgstsg2freqs11 = normxgsts[1800:2000,5096:6667]
#Transformación matrices de NumPy en vectores por columna
lista_F1G2 = normxgstsg2freqs1.flatten(order='F')
lista_F2G2 = normxgstsg2freqs2.flatten(order='F')
lista_F3G2 = normxgstsg2freqs3.flatten(order='F')
lista_F4G2 = normxgstsg2freqs4.flatten(order='F')
lista_F5G2 = normxgstsg2freqs5.flatten(order='F')
lista_F6G2 = normxgstsg2freqs6.flatten(order='F')
lista_F7G2 = normxgstsg2freqs7.flatten(order='F')
lista_F8G2 = normxgstsg2freqs8.flatten(order='F')
lista_F9G2 = normxgstsg2freqs9.flatten(order='F')
lista_F10G2 = normxgstsg2freqs10.flatten(order='F')
lista_F11G2 = normxgstsg2freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G2 = mean(lista_F1G2)
Promedio_lista_F2G2 = mean(lista_F2G2)
Promedio_lista_F3G2 = mean(lista_F3G2)
Promedio_lista_F4G2 = mean(lista_F4G2)
Promedio_lista_F5G2 = mean(lista_F5G2)
Promedio_lista_F6G2 = mean(lista_F6G2)
Promedio_lista_F7G2 = mean(lista_F7G2)
Promedio_lista_F8G2 = mean(lista_F8G2)
Promedio_lista_F9G2 = mean(lista_F9G2)
Promedio_lista_F10G2 = mean(lista_F10G2)
Promedio_lista_F11G2 = mean(lista_F11G2)
#Lista de Promedios
PromediosG2 = []
PromediosG2.append(Promedio_lista_F1G2)
PromediosG2.append(Promedio_lista_F2G2)
PromediosG2.append(Promedio_lista_F3G2)
PromediosG2.append(Promedio_lista_F4G2)
PromediosG2.append(Promedio_lista_F5G2)
PromediosG2.append(Promedio_lista_F6G2)
PromediosG2.append(Promedio_lista_F7G2)
PromediosG2.append(Promedio_lista_F8G2)
PromediosG2.append(Promedio_lista_F9G2)
PromediosG2.append(Promedio_lista_F10G2)
PromediosG2.append(Promedio_lista_F11G2)
 
#Segmentación G3
normxgstsg3freqs1 = normxgsts[3800:4000,11252:13065]
normxgstsg3freqs2 = normxgsts[3600:3800,11252:13065]
normxgstsg3freqs3 = normxgsts[3400:3600,11252:13065]
normxgstsg3freqs4 = normxgsts[3200:3400,11252:13065]
normxgstsg3freqs5 = normxgsts[3000:3200,11252:13065]
normxgstsg3freqs6 = normxgsts[2800:3000,11252:13065]
normxgstsg3freqs7 = normxgsts[2600:2800,11252:13065]
normxgstsg3freqs8 = normxgsts[2400:2600,11252:13065]
normxgstsg3freqs9 = normxgsts[2200:2400,11252:13065]
normxgstsg3freqs10 = normxgsts[2000:2200,11252:13065]
normxgstsg3freqs11 = normxgsts[1800:2000,11252:13065]
#Transformación matrices de NumPy en vectores por columna
lista_F1G3 = normxgstsg3freqs1.flatten(order='F')
lista_F2G3 = normxgstsg3freqs2.flatten(order='F')
lista_F3G3 = normxgstsg3freqs3.flatten(order='F')
lista_F4G3 = normxgstsg3freqs4.flatten(order='F')
lista_F5G3 = normxgstsg3freqs5.flatten(order='F')
lista_F6G3 = normxgstsg3freqs6.flatten(order='F')
lista_F7G3 = normxgstsg3freqs7.flatten(order='F')
lista_F8G3 = normxgstsg3freqs8.flatten(order='F')
lista_F9G3 = normxgstsg3freqs9.flatten(order='F')
lista_F10G3 = normxgstsg3freqs10.flatten(order='F')
lista_F11G3 = normxgstsg3freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G3 = mean(lista_F1G3)
Promedio_lista_F2G3 = mean(lista_F2G3)
Promedio_lista_F3G3 = mean(lista_F3G3)
Promedio_lista_F4G3 = mean(lista_F4G3)
Promedio_lista_F5G3 = mean(lista_F5G3)
Promedio_lista_F6G3 = mean(lista_F6G3)
Promedio_lista_F7G3 = mean(lista_F7G3)
Promedio_lista_F8G3 = mean(lista_F8G3)
Promedio_lista_F9G3 = mean(lista_F9G3)
Promedio_lista_F10G3 = mean(lista_F10G3)
Promedio_lista_F11G3 = mean(lista_F11G3)
#Lista de Promedios
PromediosG3 = []
PromediosG3.append(Promedio_lista_F1G3)
PromediosG3.append(Promedio_lista_F2G3)
PromediosG3.append(Promedio_lista_F3G3)
PromediosG3.append(Promedio_lista_F4G3)
PromediosG3.append(Promedio_lista_F5G3)
PromediosG3.append(Promedio_lista_F6G3)
PromediosG3.append(Promedio_lista_F7G3)
PromediosG3.append(Promedio_lista_F8G3)
PromediosG3.append(Promedio_lista_F9G3)
PromediosG3.append(Promedio_lista_F10G3)
PromediosG3.append(Promedio_lista_F11G3)  
 
#Segmentación G4
normxgstsg4freqs1 = normxgsts[3800:4000,16258:17683]
normxgstsg4freqs2 = normxgsts[3600:3800,16258:17683]
normxgstsg4freqs3 = normxgsts[3400:3600,16258:17683]
normxgstsg4freqs4 = normxgsts[3200:3400,16258:17683]
normxgstsg4freqs5 = normxgsts[3000:3200,16258:17683]
normxgstsg4freqs6 = normxgsts[2800:3000,16258:17683]
normxgstsg4freqs7 = normxgsts[2600:2800,16258:17683]
normxgstsg4freqs8 = normxgsts[2400:2600,16258:17683]
normxgstsg4freqs9 = normxgsts[2200:2400,16258:17683]
normxgstsg4freqs10 = normxgsts[2000:2200,16258:17683]
normxgstsg4freqs11 = normxgsts[1800:2000,16258:17683]
#Transformación matrices de NumPy en vectores por columna
lista_F1G4 = normxgstsg4freqs1.flatten(order='F')
lista_F2G4 = normxgstsg4freqs2.flatten(order='F')
lista_F3G4 = normxgstsg4freqs3.flatten(order='F')
lista_F4G4 = normxgstsg4freqs4.flatten(order='F')
lista_F5G4 = normxgstsg4freqs5.flatten(order='F')
lista_F6G4 = normxgstsg4freqs6.flatten(order='F')
lista_F7G4 = normxgstsg4freqs7.flatten(order='F')
lista_F8G4 = normxgstsg4freqs8.flatten(order='F')
lista_F9G4 = normxgstsg4freqs9.flatten(order='F')
lista_F10G4 = normxgstsg4freqs10.flatten(order='F')
lista_F11G4 = normxgstsg4freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G4 = mean(lista_F1G4)
Promedio_lista_F2G4 = mean(lista_F2G4)
Promedio_lista_F3G4 = mean(lista_F3G4)
Promedio_lista_F4G4 = mean(lista_F4G4)
Promedio_lista_F5G4 = mean(lista_F5G4)
Promedio_lista_F6G4 = mean(lista_F6G4)
Promedio_lista_F7G4 = mean(lista_F7G4)
Promedio_lista_F8G4 = mean(lista_F8G4)
Promedio_lista_F9G4 = mean(lista_F9G4)
Promedio_lista_F10G4 = mean(lista_F10G4)
Promedio_lista_F11G4 = mean(lista_F11G4)
#Lista de Promedios
PromediosG4 = []
PromediosG4.append(Promedio_lista_F1G4)
PromediosG4.append(Promedio_lista_F2G4)
PromediosG4.append(Promedio_lista_F3G4)
PromediosG4.append(Promedio_lista_F4G4)
PromediosG4.append(Promedio_lista_F5G4)
PromediosG4.append(Promedio_lista_F6G4)
PromediosG4.append(Promedio_lista_F7G4)
PromediosG4.append(Promedio_lista_F8G4)
PromediosG4.append(Promedio_lista_F9G4)
PromediosG4.append(Promedio_lista_F10G4)
PromediosG4.append(Promedio_lista_F11G4)      
 
#Segmentación G5
normxgstsg5freqs1 = normxgsts[3800:4000,20702:22187]
normxgstsg5freqs2 = normxgsts[3600:3800,20702:22187]
normxgstsg5freqs3 = normxgsts[3400:3600,20702:22187]
normxgstsg5freqs4 = normxgsts[3200:3400,20702:22187]
normxgstsg5freqs5 = normxgsts[3000:3200,20702:22187]
normxgstsg5freqs6 = normxgsts[2800:3000,20702:22187]
normxgstsg5freqs7 = normxgsts[2600:2800,20702:22187]
normxgstsg5freqs8 = normxgsts[2400:2600,20702:22187]
normxgstsg5freqs9 = normxgsts[2200:2400,20702:22187]
normxgstsg5freqs10 = normxgsts[2000:2200,20702:22187]
normxgstsg5freqs11 = normxgsts[1800:2000,20702:22187]
#Transformación matrices de NumPy en vectores por columna
lista_F1G5 = normxgstsg5freqs1.flatten(order='F')
lista_F2G5 = normxgstsg5freqs2.flatten(order='F')
lista_F3G5 = normxgstsg5freqs3.flatten(order='F')
lista_F4G5 = normxgstsg5freqs4.flatten(order='F')
lista_F5G5 = normxgstsg5freqs5.flatten(order='F')
lista_F6G5 = normxgstsg5freqs6.flatten(order='F')
lista_F7G5 = normxgstsg5freqs7.flatten(order='F')
lista_F8G5 = normxgstsg5freqs8.flatten(order='F')
lista_F9G5 = normxgstsg5freqs9.flatten(order='F')
lista_F10G5 = normxgstsg5freqs10.flatten(order='F')
lista_F11G5 = normxgstsg5freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G5 = mean(lista_F1G5)
Promedio_lista_F2G5 = mean(lista_F2G5)
Promedio_lista_F3G5 = mean(lista_F3G5)
Promedio_lista_F4G5 = mean(lista_F4G5)
Promedio_lista_F5G5 = mean(lista_F5G5)
Promedio_lista_F6G5 = mean(lista_F6G5)
Promedio_lista_F7G5 = mean(lista_F7G5)
Promedio_lista_F8G5 = mean(lista_F8G5)
Promedio_lista_F9G5 = mean(lista_F9G5)
Promedio_lista_F10G5 = mean(lista_F10G5)
Promedio_lista_F11G5 = mean(lista_F11G5)
#Lista de Promedios
PromediosG5 = []
PromediosG5.append(Promedio_lista_F1G5)
PromediosG5.append(Promedio_lista_F2G5)
PromediosG5.append(Promedio_lista_F3G5)
PromediosG5.append(Promedio_lista_F4G5)
PromediosG5.append(Promedio_lista_F5G5)
PromediosG5.append(Promedio_lista_F6G5)
PromediosG5.append(Promedio_lista_F7G5)
PromediosG5.append(Promedio_lista_F8G5)
PromediosG5.append(Promedio_lista_F9G5)
PromediosG5.append(Promedio_lista_F10G5)
PromediosG5.append(Promedio_lista_F11G5)
 
#Segmentación G6
normxgstsg6freqs1 = normxgsts[3800:4000,25256:26875]
normxgstsg6freqs2 = normxgsts[3600:3800,25256:26875]
normxgstsg6freqs3 = normxgsts[3400:3600,25256:26875]
normxgstsg6freqs4 = normxgsts[3200:3400,25256:26875]
normxgstsg6freqs5 = normxgsts[3000:3200,25256:26875]
normxgstsg6freqs6 = normxgsts[2800:3000,25256:26875]
normxgstsg6freqs7 = normxgsts[2600:2800,25256:26875]
normxgstsg6freqs8 = normxgsts[2400:2600,25256:26875]
normxgstsg6freqs9 = normxgsts[2200:2400,25256:26875]
normxgstsg6freqs10 = normxgsts[2000:2200,25256:26875]
normxgstsg6freqs11 = normxgsts[1800:2000,25256:26875]
#Transformación matrices de NumPy en vectores por columna
lista_F1G6 = normxgstsg6freqs1.flatten(order='F')
lista_F2G6 = normxgstsg6freqs2.flatten(order='F')
lista_F3G6 = normxgstsg6freqs3.flatten(order='F')
lista_F4G6 = normxgstsg6freqs4.flatten(order='F')
lista_F5G6 = normxgstsg6freqs5.flatten(order='F')
lista_F6G6 = normxgstsg6freqs6.flatten(order='F')
lista_F7G6 = normxgstsg6freqs7.flatten(order='F')
lista_F8G6 = normxgstsg6freqs8.flatten(order='F')
lista_F9G6 = normxgstsg6freqs9.flatten(order='F')
lista_F10G6 = normxgstsg6freqs10.flatten(order='F')
lista_F11G6 = normxgstsg6freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G6 = mean(lista_F1G6)
Promedio_lista_F2G6 = mean(lista_F2G6)
Promedio_lista_F3G6 = mean(lista_F3G6)
Promedio_lista_F4G6 = mean(lista_F4G6)
Promedio_lista_F5G6 = mean(lista_F5G6)
Promedio_lista_F6G6 = mean(lista_F6G6)
Promedio_lista_F7G6 = mean(lista_F7G6)
Promedio_lista_F8G6 = mean(lista_F8G6)
Promedio_lista_F9G6 = mean(lista_F9G6)
Promedio_lista_F10G6 = mean(lista_F10G6)
Promedio_lista_F11G6 = mean(lista_F11G6)
#Lista de Promedios
PromediosG6 = []
PromediosG6.append(Promedio_lista_F1G6)
PromediosG6.append(Promedio_lista_F2G6)
PromediosG6.append(Promedio_lista_F3G6)
PromediosG6.append(Promedio_lista_F4G6)
PromediosG6.append(Promedio_lista_F5G6)
PromediosG6.append(Promedio_lista_F6G6)
PromediosG6.append(Promedio_lista_F7G6)
PromediosG6.append(Promedio_lista_F8G6)
PromediosG6.append(Promedio_lista_F9G6)
PromediosG6.append(Promedio_lista_F10G6)
PromediosG6.append(Promedio_lista_F11G6)
 
#Segmentación G7
normxgstsg7freqs1 = normxgsts[3800:4000,29945:31248]
normxgstsg7freqs2 = normxgsts[3600:3800,29945:31248]
normxgstsg7freqs3 = normxgsts[3400:3600,29945:31248]
normxgstsg7freqs4 = normxgsts[3200:3400,29945:31248]
normxgstsg7freqs5 = normxgsts[3000:3200,29945:31248]
normxgstsg7freqs6 = normxgsts[2800:3000,29945:31248]
normxgstsg7freqs7 = normxgsts[2600:2800,29945:31248]
normxgstsg7freqs8 = normxgsts[2400:2600,29945:31248]
normxgstsg7freqs9 = normxgsts[2200:2400,29945:31248]
normxgstsg7freqs10 = normxgsts[2000:2200,29945:31248]
normxgstsg7freqs11 = normxgsts[1800:2000,29945:31248]
#Transformación matrices de NumPy en vectores por columna
lista_F1G7 = normxgstsg7freqs1.flatten(order='F')
lista_F2G7 = normxgstsg7freqs2.flatten(order='F')
lista_F3G7 = normxgstsg7freqs3.flatten(order='F')
lista_F4G7 = normxgstsg7freqs4.flatten(order='F')
lista_F5G7 = normxgstsg7freqs5.flatten(order='F')
lista_F6G7 = normxgstsg7freqs6.flatten(order='F')
lista_F7G7 = normxgstsg7freqs7.flatten(order='F')
lista_F8G7 = normxgstsg7freqs8.flatten(order='F')
lista_F9G7 = normxgstsg7freqs9.flatten(order='F')
lista_F10G7 = normxgstsg7freqs10.flatten(order='F')
lista_F11G7 = normxgstsg7freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G7 = mean(lista_F1G7)
Promedio_lista_F2G7 = mean(lista_F2G7)
Promedio_lista_F3G7 = mean(lista_F3G7)
Promedio_lista_F4G7 = mean(lista_F4G7)
Promedio_lista_F5G7 = mean(lista_F5G7)
Promedio_lista_F6G7 = mean(lista_F6G7)
Promedio_lista_F7G7 = mean(lista_F7G7)
Promedio_lista_F8G7 = mean(lista_F8G7)
Promedio_lista_F9G7 = mean(lista_F9G7)
Promedio_lista_F10G7 = mean(lista_F10G7)
Promedio_lista_F11G7 = mean(lista_F11G7)
#Lista de Promedios
PromediosG7 = []
PromediosG7.append(Promedio_lista_F1G7)
PromediosG7.append(Promedio_lista_F2G7)
PromediosG7.append(Promedio_lista_F3G7)
PromediosG7.append(Promedio_lista_F4G7)
PromediosG7.append(Promedio_lista_F5G7)
PromediosG7.append(Promedio_lista_F6G7)
PromediosG7.append(Promedio_lista_F7G7)
PromediosG7.append(Promedio_lista_F8G7)
PromediosG7.append(Promedio_lista_F9G7)
PromediosG7.append(Promedio_lista_F10G7)
PromediosG7.append(Promedio_lista_F11G7)
 
#Segmentación G8
normxgstsg8freqs1 = normxgsts[3800:4000,32784:34314] 
normxgstsg8freqs2 = normxgsts[3600:3800,32784:34314] 
normxgstsg8freqs3 = normxgsts[3400:3600,32784:34314] 
normxgstsg8freqs4 = normxgsts[3200:3400,32784:34314] 
normxgstsg8freqs5 = normxgsts[3000:3200,32784:34314] 
normxgstsg8freqs6 = normxgsts[2800:3000,32784:34314] 
normxgstsg8freqs7 = normxgsts[2600:2800,32784:34314] 
normxgstsg8freqs8 = normxgsts[2400:2600,32784:34314] 
normxgstsg8freqs9 = normxgsts[2200:2400,32784:34314] 
normxgstsg8freqs10 = normxgsts[2000:2200,32784:34314] 
normxgstsg8freqs11 = normxgsts[1800:2000,32784:34314] 
#Transformación matrices de NumPy en vectores por columna
lista_F1G8 = normxgstsg8freqs1.flatten(order='F')
lista_F2G8 = normxgstsg8freqs2.flatten(order='F')
lista_F3G8 = normxgstsg8freqs3.flatten(order='F')
lista_F4G8 = normxgstsg8freqs4.flatten(order='F')
lista_F5G8 = normxgstsg8freqs5.flatten(order='F')
lista_F6G8 = normxgstsg8freqs6.flatten(order='F')
lista_F7G8 = normxgstsg8freqs7.flatten(order='F')
lista_F8G8 = normxgstsg8freqs8.flatten(order='F')
lista_F9G8 = normxgstsg8freqs9.flatten(order='F')
lista_F10G8 = normxgstsg8freqs10.flatten(order='F')
lista_F11G8 = normxgstsg8freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G8 = mean(lista_F1G8)
Promedio_lista_F2G8 = mean(lista_F2G8)
Promedio_lista_F3G8 = mean(lista_F3G8)
Promedio_lista_F4G8 = mean(lista_F4G8)
Promedio_lista_F5G8 = mean(lista_F5G8)
Promedio_lista_F6G8 = mean(lista_F6G8)
Promedio_lista_F7G8 = mean(lista_F7G8)
Promedio_lista_F8G8 = mean(lista_F8G8)
Promedio_lista_F9G8 = mean(lista_F9G8)
Promedio_lista_F10G8 = mean(lista_F10G8)
Promedio_lista_F11G8 = mean(lista_F11G8)
#Lista de Promedios
PromediosG8 = []
PromediosG8.append(Promedio_lista_F1G8)
PromediosG8.append(Promedio_lista_F2G8)
PromediosG8.append(Promedio_lista_F3G8)
PromediosG8.append(Promedio_lista_F4G8)
PromediosG8.append(Promedio_lista_F5G8)
PromediosG8.append(Promedio_lista_F6G8)
PromediosG8.append(Promedio_lista_F7G8)
PromediosG8.append(Promedio_lista_F8G8)
PromediosG8.append(Promedio_lista_F9G8)
PromediosG8.append(Promedio_lista_F10G8)
PromediosG8.append(Promedio_lista_F11G8)
 
#Segmentación G9
normxgstsg9freqs1 = normxgsts[3800:4000,37773:39099]
normxgstsg9freqs2 = normxgsts[3600:3800,37773:39099]
normxgstsg9freqs3 = normxgsts[3400:3600,37773:39099]
normxgstsg9freqs4 = normxgsts[3200:3400,37773:39099]
normxgstsg9freqs5 = normxgsts[3000:3200,37773:39099]
normxgstsg9freqs6 = normxgsts[2800:3000,37773:39099]
normxgstsg9freqs7 = normxgsts[2600:2800,37773:39099]
normxgstsg9freqs8 = normxgsts[2400:2600,37773:39099]
normxgstsg9freqs9 = normxgsts[2200:2400,37773:39099]
normxgstsg9freqs10 = normxgsts[2000:2200,37773:39099]
normxgstsg9freqs11 = normxgsts[1800:2000,37773:39099]
#Transformación matrices de NumPy en vectores por columna
lista_F1G9 = normxgstsg9freqs1.flatten(order='F')
lista_F2G9 = normxgstsg9freqs2.flatten(order='F')
lista_F3G9 = normxgstsg9freqs3.flatten(order='F')
lista_F4G9 = normxgstsg9freqs4.flatten(order='F')
lista_F5G9 = normxgstsg9freqs5.flatten(order='F')
lista_F6G9 = normxgstsg9freqs6.flatten(order='F')
lista_F7G9 = normxgstsg9freqs7.flatten(order='F')
lista_F8G9 = normxgstsg9freqs8.flatten(order='F')
lista_F9G9 = normxgstsg9freqs9.flatten(order='F')
lista_F10G9 = normxgstsg9freqs10.flatten(order='F')
lista_F11G9 = normxgstsg9freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G9 = mean(lista_F1G9)
Promedio_lista_F2G9 = mean(lista_F2G9)
Promedio_lista_F3G9 = mean(lista_F3G9)
Promedio_lista_F4G9 = mean(lista_F4G9)
Promedio_lista_F5G9 = mean(lista_F5G9)
Promedio_lista_F6G9 = mean(lista_F6G9)
Promedio_lista_F7G9 = mean(lista_F7G9)
Promedio_lista_F8G9 = mean(lista_F8G9)
Promedio_lista_F9G9 = mean(lista_F9G9)
Promedio_lista_F10G9 = mean(lista_F10G9)
Promedio_lista_F11G9 = mean(lista_F11G9)
#Lista de Promedios
PromediosG9 = []
PromediosG9.append(Promedio_lista_F1G9)
PromediosG9.append(Promedio_lista_F2G9)
PromediosG9.append(Promedio_lista_F3G9)
PromediosG9.append(Promedio_lista_F4G9)
PromediosG9.append(Promedio_lista_F5G9)
PromediosG9.append(Promedio_lista_F6G9)
PromediosG9.append(Promedio_lista_F7G9)
PromediosG9.append(Promedio_lista_F8G9)
PromediosG9.append(Promedio_lista_F9G9)
PromediosG9.append(Promedio_lista_F10G9)
PromediosG9.append(Promedio_lista_F11G9)
 
#Segmentación G10
normxgstsg10freqs1 = normxgsts[3800:4000,42420:44086]
normxgstsg10freqs2 = normxgsts[3600:3800,42420:44086]
normxgstsg10freqs3 = normxgsts[3400:3600,42420:44086]
normxgstsg10freqs4 = normxgsts[3200:3400,42420:44086]
normxgstsg10freqs5 = normxgsts[3000:3200,42420:44086]
normxgstsg10freqs6 = normxgsts[2800:3000,42420:44086]
normxgstsg10freqs7 = normxgsts[2600:2800,42420:44086]
normxgstsg10freqs8 = normxgsts[2400:2600,42420:44086]
normxgstsg10freqs9 = normxgsts[2200:2400,42420:44086]
normxgstsg10freqs10 = normxgsts[2000:2200,42420:44086]
normxgstsg10freqs11 = normxgsts[1800:2000,42420:44086]
#Transformación matrices de NumPy en vectores por columna
lista_F1G10 = normxgstsg10freqs1.flatten(order='F')
lista_F2G10 = normxgstsg10freqs2.flatten(order='F')
lista_F3G10 = normxgstsg10freqs3.flatten(order='F')
lista_F4G10 = normxgstsg10freqs4.flatten(order='F')
lista_F5G10 = normxgstsg10freqs5.flatten(order='F')
lista_F6G10 = normxgstsg10freqs6.flatten(order='F')
lista_F7G10 = normxgstsg10freqs7.flatten(order='F')
lista_F8G10 = normxgstsg10freqs8.flatten(order='F')
lista_F9G10 = normxgstsg10freqs9.flatten(order='F')
lista_F10G10 = normxgstsg10freqs10.flatten(order='F')
lista_F11G10 = normxgstsg10freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G10 = mean(lista_F1G10)
Promedio_lista_F2G10 = mean(lista_F2G10)
Promedio_lista_F3G10 = mean(lista_F3G10)
Promedio_lista_F4G10 = mean(lista_F4G10)
Promedio_lista_F5G10 = mean(lista_F5G10)
Promedio_lista_F6G10 = mean(lista_F6G10)
Promedio_lista_F7G10 = mean(lista_F7G10)
Promedio_lista_F8G10 = mean(lista_F8G10)
Promedio_lista_F9G10 = mean(lista_F9G10)
Promedio_lista_F10G10 = mean(lista_F10G10)
Promedio_lista_F11G10 = mean(lista_F11G10)
#Lista de Promedios
PromediosG10 = []
PromediosG10.append(Promedio_lista_F1G10)
PromediosG10.append(Promedio_lista_F2G10)
PromediosG10.append(Promedio_lista_F3G10)
PromediosG10.append(Promedio_lista_F4G10)
PromediosG10.append(Promedio_lista_F5G10)
PromediosG10.append(Promedio_lista_F6G10)
PromediosG10.append(Promedio_lista_F7G10)
PromediosG10.append(Promedio_lista_F8G10)
PromediosG10.append(Promedio_lista_F9G10)
PromediosG10.append(Promedio_lista_F10G10)
PromediosG10.append(Promedio_lista_F11G10)
 
#Segmentación G11
normxgstsg11freqs1 = normxgsts[3800:4000,46488:48175]
normxgstsg11freqs2 = normxgsts[3600:3800,46488:48175]
normxgstsg11freqs3 = normxgsts[3400:3600,46488:48175]
normxgstsg11freqs4 = normxgsts[3200:3400,46488:48175]
normxgstsg11freqs5 = normxgsts[3000:3200,46488:48175]
normxgstsg11freqs6 = normxgsts[2800:3000,46488:48175]
normxgstsg11freqs7 = normxgsts[2600:2800,46488:48175]
normxgstsg11freqs8 = normxgsts[2400:2600,46488:48175]
normxgstsg11freqs9 = normxgsts[2200:2400,46488:48175]
normxgstsg11freqs10 = normxgsts[2000:2200,46488:48175]
normxgstsg11freqs11 = normxgsts[1800:2000,46488:48175]
#Transformación matrices de NumPy en vectores por columna
lista_F1G11 = normxgstsg11freqs1.flatten(order='F')
lista_F2G11 = normxgstsg11freqs2.flatten(order='F')
lista_F3G11 = normxgstsg11freqs3.flatten(order='F')
lista_F4G11 = normxgstsg11freqs4.flatten(order='F')
lista_F5G11 = normxgstsg11freqs5.flatten(order='F')
lista_F6G11 = normxgstsg11freqs6.flatten(order='F')
lista_F7G11 = normxgstsg11freqs7.flatten(order='F')
lista_F8G11 = normxgstsg11freqs8.flatten(order='F')
lista_F9G11 = normxgstsg11freqs9.flatten(order='F')
lista_F10G11 = normxgstsg11freqs10.flatten(order='F')
lista_F11G11 = normxgstsg11freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G11 = mean(lista_F1G11)
Promedio_lista_F2G11 = mean(lista_F2G11)
Promedio_lista_F3G11 = mean(lista_F3G11)
Promedio_lista_F4G11 = mean(lista_F4G11)
Promedio_lista_F5G11 = mean(lista_F5G11)
Promedio_lista_F6G11 = mean(lista_F6G11)
Promedio_lista_F7G11 = mean(lista_F7G11)
Promedio_lista_F8G11 = mean(lista_F8G11)
Promedio_lista_F9G11 = mean(lista_F9G11)
Promedio_lista_F10G11 = mean(lista_F10G11)
Promedio_lista_F11G11 = mean(lista_F11G11)
#Lista de Promedios
PromediosG11 = []
PromediosG11.append(Promedio_lista_F1G11)
PromediosG11.append(Promedio_lista_F2G11)
PromediosG11.append(Promedio_lista_F3G11)
PromediosG11.append(Promedio_lista_F4G11)
PromediosG11.append(Promedio_lista_F5G11)
PromediosG11.append(Promedio_lista_F6G11)
PromediosG11.append(Promedio_lista_F7G11)
PromediosG11.append(Promedio_lista_F8G11)
PromediosG11.append(Promedio_lista_F9G11)
PromediosG11.append(Promedio_lista_F10G11)
PromediosG11.append(Promedio_lista_F11G11)
 
#Segmentación G12
normxgstsg12freqs1 = normxgsts[3800:4000,50528:52032]
normxgstsg12freqs2 = normxgsts[3600:3800,50528:52032]
normxgstsg12freqs3 = normxgsts[3400:3600,50528:52032]
normxgstsg12freqs4 = normxgsts[3200:3400,50528:52032]
normxgstsg12freqs5 = normxgsts[3000:3200,50528:52032]
normxgstsg12freqs6 = normxgsts[2800:3000,50528:52032]
normxgstsg12freqs7 = normxgsts[2600:2800,50528:52032]
normxgstsg12freqs8 = normxgsts[2400:2600,50528:52032]
normxgstsg12freqs9 = normxgsts[2200:2400,50528:52032]
normxgstsg12freqs10 = normxgsts[2000:2200,50528:52032]
normxgstsg12freqs11 = normxgsts[1800:2000,50528:52032]
#Transformación matrices de NumPy en vectores por columna
lista_F1G12 = normxgstsg12freqs1.flatten(order='F')
lista_F2G12 = normxgstsg12freqs2.flatten(order='F')
lista_F3G12 = normxgstsg12freqs3.flatten(order='F')
lista_F4G12 = normxgstsg12freqs4.flatten(order='F')
lista_F5G12 = normxgstsg12freqs5.flatten(order='F')
lista_F6G12 = normxgstsg12freqs6.flatten(order='F')
lista_F7G12 = normxgstsg12freqs7.flatten(order='F')
lista_F8G12 = normxgstsg12freqs8.flatten(order='F')
lista_F9G12 = normxgstsg12freqs9.flatten(order='F')
lista_F10G12 = normxgstsg12freqs10.flatten(order='F')
lista_F11G12 = normxgstsg12freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G12 = mean(lista_F1G12)
Promedio_lista_F2G12 = mean(lista_F2G12)
Promedio_lista_F3G12 = mean(lista_F3G12)
Promedio_lista_F4G12 = mean(lista_F4G12)
Promedio_lista_F5G12 = mean(lista_F5G12)
Promedio_lista_F6G12 = mean(lista_F6G12)
Promedio_lista_F7G12 = mean(lista_F7G12)
Promedio_lista_F8G12 = mean(lista_F8G12)
Promedio_lista_F9G12 = mean(lista_F9G12)
Promedio_lista_F10G12 = mean(lista_F10G12)
Promedio_lista_F11G12 = mean(lista_F11G12)
#Lista de Promedios
PromediosG12 = []
PromediosG12.append(Promedio_lista_F1G12)
PromediosG12.append(Promedio_lista_F2G12)
PromediosG12.append(Promedio_lista_F3G12)
PromediosG12.append(Promedio_lista_F4G12)
PromediosG12.append(Promedio_lista_F5G12)
PromediosG12.append(Promedio_lista_F6G12)
PromediosG12.append(Promedio_lista_F7G12)
PromediosG12.append(Promedio_lista_F8G12)
PromediosG12.append(Promedio_lista_F9G12)
PromediosG12.append(Promedio_lista_F10G12)
PromediosG12.append(Promedio_lista_F11G12)

Promedios = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6, 
             PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]
PromediosFrecuencias = np.array(Promedios); 
PromediosFrecuencias = np.transpose(PromediosFrecuencias)

DF = pd.DataFrame(PromediosFrecuencias)
#DF.to_csv("Promedios Paciente 8 Canal 4.csv")

PromediosG1 = np.array(PromediosG1)
PromediosG1 = np.transpose(PromediosG1)

PromediosG2 = np.array(PromediosG2)
PromediosG2 = np.transpose(PromediosG2)

PromediosG3 = np.array(PromediosG3)
PromediosG3 = np.transpose(PromediosG3)

PromediosG4 = np.array(PromediosG4)
PromediosG4 = np.transpose(PromediosG4)

PromediosG5 = np.array(PromediosG5)
PromediosG5 = np.transpose(PromediosG5)

PromediosG6 = np.array(PromediosG6)
PromediosG6 = np.transpose(PromediosG6)

PromediosG7 = np.array(PromediosG7)
PromediosG7 = np.transpose(PromediosG7)

PromediosG8 = np.array(PromediosG8)
PromediosG8 = np.transpose(PromediosG8)

PromediosG9 = np.array(PromediosG9)
PromediosG9 = np.transpose(PromediosG9)

PromediosG10 = np.array(PromediosG10)
PromediosG10 = np.transpose(PromediosG10)

PromediosG11 = np.array(PromediosG11)
PromediosG11 = np.transpose(PromediosG11)

PromediosG12 = np.array(PromediosG12)
PromediosG12 = np.transpose(PromediosG12)

data = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6,
        PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]

#Secuencia Consecutiva de Matrices 
SPromediosG1 = list(itertools.chain(PromediosG1, PromediosG7))
SPromediosG1 = np.array(SPromediosG1)
SPromediosG2 = list(itertools.chain(PromediosG2, PromediosG8))
SPromediosG2 = np.array(SPromediosG2)
SPromediosG3 = list(itertools.chain(PromediosG3, PromediosG9))
SPromediosG3 = np.array(SPromediosG3)
SPromediosG4 = list(itertools.chain(PromediosG4, PromediosG10))
SPromediosG4 = np.array(SPromediosG4)
SPromediosG5 = list(itertools.chain(PromediosG5, PromediosG11))
SPromediosG5 = np.array(SPromediosG5)
SPromediosG6 = list(itertools.chain(PromediosG6, PromediosG12))
SPromediosG6 = np.array(SPromediosG6)
SPromedios = [SPromediosG1, SPromediosG2, SPromediosG3, SPromediosG4, SPromediosG5, SPromediosG6]

SPromediosU0 = list(itertools.chain(SPromedios[0], SPromedios[1]))
SPromediosU1 = list(itertools.chain(SPromediosU0, SPromedios[2]))
SPromediosU2 = list(itertools.chain(SPromediosU1, SPromedios[3]))
SPromediosU3 = list(itertools.chain(SPromediosU2, SPromedios[4]))
SPromediosU4 = list(itertools.chain(SPromediosU3, SPromedios[5]))
SPromediosU = np.array(SPromediosU4)

#Histograma
plt.figure(figsize =(16, 10))
plt.hist(SPromediosU)
plt.grid()
plt.title('Histograma de la Secuencia Consecutiva de Promedios')
plt.xlabel('Valor de la variable')
plt.ylabel('Conteo')
plt.savefig('Histograma Promedios Gesto 1 P3C6.png', dpi = 201)
plt.show()

#scipy stats Wilcoxon 
WCresultados = []
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[1], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[2], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)


stat, p_value = wilcoxon(SPromedios[3], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[4], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[5], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)

WCresultados = np.array(WCresultados)
WCresultados = np.reshape(WCresultados,[6,6])

DF = pd.DataFrame(WCresultados) 
DF.to_csv("Wilcoxon Resultados entre gestos distintos P8C4.csv")

emg_funcional = np.array(voltajesP09[1])
#Lectura archivo 
samplerate = 1e3 #Frecuencia de muestreo
length = emg_funcional.shape[0] / samplerate #Duración total Electromiografía en segundos
#print("Duración Total = {length}s")

#Parametros CWT 
num = 4000 #Número de sampleo generado
wavelet = 'cmor1.0-1.5' # Bugs Fixed https://pywavelets.readthedocs.io/en/latest/release.1.1.0.html 
maxesc = np.log2(1500)
widths = np.logspace(0,maxesc, num=num, base=2,dtype='int')
#print("Escalas Usadas: ", widths)
dt = 1/samplerate # diferencia de intervalos de tiempo
frequencies = pywt.scale2frequency(wavelet, widths) / dt # Obtención de frecuencias correspondientes a escalas Modificar aquí precisi´´on
#print("Freqs asociadas a Escalas: ", frequencies)
# Calculo transformación continua de ondículas de los anchos de matriz numpy
wavelet_coeffs, freqs = pywt.cwt(emg_funcional, widths, wavelet=wavelet, sampling_period=samplerate, method='fft')
#print("Tamaño de wavelet transform: ", wavelet_coeffs.shape)
#print("Freqs asociadas a wavelet transform: ", freqs)

#Valores Absolutos de la CWT
abscwt = abs(wavelet_coeffs) # Devolución de ABS de los números complejos es la magnitud
gesto_1 = abscwt[:,1565:3051]
gesto_2 = abscwt[:,4989:6769]
gesto_3 = abscwt[:,11214:13040]
gesto_4 = abscwt[:,17561:19223]
gesto_5 = abscwt[:,23395:25044]
gesto_6 = abscwt[:,29598:31035]
gesto_7 = abscwt[:,34044:35612]
gesto_8 = abscwt[:,37689:39624]
gesto_9 = abscwt[:,43805:45576]
gesto_10 = abscwt[:,49214:50578]
gesto_11 = abscwt[:,54124:55690]
gesto_12 = abscwt[:,59188:60808]

normxgsts = abscwt
#División de segmentos desde 0 a 57,69 Hz

#Segmentación G1
normxgstsg1freqs1 = normxgsts[3800:4000,1564:3052]
normxgstsg1freqs2 = normxgsts[3600:3800,1564:3052]
normxgstsg1freqs3 = normxgsts[3400:3600,1564:3052]
normxgstsg1freqs4 = normxgsts[3200:3400,1564:3052]
normxgstsg1freqs5 = normxgsts[3000:3200,1564:3052]
normxgstsg1freqs6 = normxgsts[2800:3000,1564:3052]
normxgstsg1freqs7 = normxgsts[2600:2800,1564:3052]
normxgstsg1freqs8 = normxgsts[2400:2600,1564:3052]
normxgstsg1freqs9 = normxgsts[2200:2400,1564:3052]
normxgstsg1freqs10 = normxgsts[2000:2200,1564:3052]
normxgstsg1freqs11 = normxgsts[1800:2000,1564:3052]
#Transformación matrices de NumPy en vectores por columna
lista_F1G1 = normxgstsg1freqs1.flatten(order='F')
lista_F2G1 = normxgstsg1freqs2.flatten(order='F')
lista_F3G1 = normxgstsg1freqs3.flatten(order='F')
lista_F4G1 = normxgstsg1freqs4.flatten(order='F')
lista_F5G1 = normxgstsg1freqs5.flatten(order='F')
lista_F6G1 = normxgstsg1freqs6.flatten(order='F')
lista_F7G1 = normxgstsg1freqs7.flatten(order='F')
lista_F8G1 = normxgstsg1freqs8.flatten(order='F')
lista_F9G1 = normxgstsg1freqs9.flatten(order='F')
lista_F10G1 = normxgstsg1freqs10.flatten(order='F')
lista_F11G1 = normxgstsg1freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G1 = mean(lista_F1G1)
Promedio_lista_F2G1 = mean(lista_F2G1)
Promedio_lista_F3G1 = mean(lista_F3G1)
Promedio_lista_F4G1 = mean(lista_F4G1)
Promedio_lista_F5G1 = mean(lista_F5G1)
Promedio_lista_F6G1 = mean(lista_F6G1)
Promedio_lista_F7G1 = mean(lista_F7G1)
Promedio_lista_F8G1 = mean(lista_F8G1)
Promedio_lista_F9G1 = mean(lista_F9G1)
Promedio_lista_F10G1 = mean(lista_F10G1)
Promedio_lista_F11G1 = mean(lista_F11G1)
#Lista de Promedios
PromediosG1 = []
PromediosG1.append(Promedio_lista_F1G1)
PromediosG1.append(Promedio_lista_F2G1)
PromediosG1.append(Promedio_lista_F3G1)
PromediosG1.append(Promedio_lista_F4G1)
PromediosG1.append(Promedio_lista_F5G1)
PromediosG1.append(Promedio_lista_F6G1)
PromediosG1.append(Promedio_lista_F7G1)
PromediosG1.append(Promedio_lista_F8G1)
PromediosG1.append(Promedio_lista_F9G1)
PromediosG1.append(Promedio_lista_F10G1)
PromediosG1.append(Promedio_lista_F11G1)
 
#Segmentación G2
normxgstsg2freqs1 = normxgsts[3800:4000,4988:6770]
normxgstsg2freqs2 = normxgsts[3600:3800,4988:6770]
normxgstsg2freqs3 = normxgsts[3400:3600,4988:6770]
normxgstsg2freqs4 = normxgsts[3200:3400,4988:6770]
normxgstsg2freqs5 = normxgsts[3000:3200,4988:6770]
normxgstsg2freqs6 = normxgsts[2800:3000,4988:6770]
normxgstsg2freqs7 = normxgsts[2600:2800,4988:6770]
normxgstsg2freqs8 = normxgsts[2400:2600,4988:6770]
normxgstsg2freqs9 = normxgsts[2200:2400,4988:6770]
normxgstsg2freqs10 = normxgsts[2000:2200,4988:6770]
normxgstsg2freqs11 = normxgsts[1800:2000,4988:6770]
#Transformación matrices de NumPy en vectores por columna
lista_F1G2 = normxgstsg2freqs1.flatten(order='F')
lista_F2G2 = normxgstsg2freqs2.flatten(order='F')
lista_F3G2 = normxgstsg2freqs3.flatten(order='F')
lista_F4G2 = normxgstsg2freqs4.flatten(order='F')
lista_F5G2 = normxgstsg2freqs5.flatten(order='F')
lista_F6G2 = normxgstsg2freqs6.flatten(order='F')
lista_F7G2 = normxgstsg2freqs7.flatten(order='F')
lista_F8G2 = normxgstsg2freqs8.flatten(order='F')
lista_F9G2 = normxgstsg2freqs9.flatten(order='F')
lista_F10G2 = normxgstsg2freqs10.flatten(order='F')
lista_F11G2 = normxgstsg2freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G2 = mean(lista_F1G2)
Promedio_lista_F2G2 = mean(lista_F2G2)
Promedio_lista_F3G2 = mean(lista_F3G2)
Promedio_lista_F4G2 = mean(lista_F4G2)
Promedio_lista_F5G2 = mean(lista_F5G2)
Promedio_lista_F6G2 = mean(lista_F6G2)
Promedio_lista_F7G2 = mean(lista_F7G2)
Promedio_lista_F8G2 = mean(lista_F8G2)
Promedio_lista_F9G2 = mean(lista_F9G2)
Promedio_lista_F10G2 = mean(lista_F10G2)
Promedio_lista_F11G2 = mean(lista_F11G2)
#Lista de Promedios
PromediosG2 = []
PromediosG2.append(Promedio_lista_F1G2)
PromediosG2.append(Promedio_lista_F2G2)
PromediosG2.append(Promedio_lista_F3G2)
PromediosG2.append(Promedio_lista_F4G2)
PromediosG2.append(Promedio_lista_F5G2)
PromediosG2.append(Promedio_lista_F6G2)
PromediosG2.append(Promedio_lista_F7G2)
PromediosG2.append(Promedio_lista_F8G2)
PromediosG2.append(Promedio_lista_F9G2)
PromediosG2.append(Promedio_lista_F10G2)
PromediosG2.append(Promedio_lista_F11G2)
 
#Segmentación G3
normxgstsg3freqs1 = normxgsts[3800:4000,11213:13041]
normxgstsg3freqs2 = normxgsts[3600:3800,11213:13041]
normxgstsg3freqs3 = normxgsts[3400:3600,11213:13041]
normxgstsg3freqs4 = normxgsts[3200:3400,11213:13041]
normxgstsg3freqs5 = normxgsts[3000:3200,11213:13041]
normxgstsg3freqs6 = normxgsts[2800:3000,11213:13041]
normxgstsg3freqs7 = normxgsts[2600:2800,11213:13041]
normxgstsg3freqs8 = normxgsts[2400:2600,11213:13041]
normxgstsg3freqs9 = normxgsts[2200:2400,11213:13041]
normxgstsg3freqs10 = normxgsts[2000:2200,11213:13041]
normxgstsg3freqs11 = normxgsts[1800:2000,11213:13041]
#Transformación matrices de NumPy en vectores por columna
lista_F1G3 = normxgstsg3freqs1.flatten(order='F')
lista_F2G3 = normxgstsg3freqs2.flatten(order='F')
lista_F3G3 = normxgstsg3freqs3.flatten(order='F')
lista_F4G3 = normxgstsg3freqs4.flatten(order='F')
lista_F5G3 = normxgstsg3freqs5.flatten(order='F')
lista_F6G3 = normxgstsg3freqs6.flatten(order='F')
lista_F7G3 = normxgstsg3freqs7.flatten(order='F')
lista_F8G3 = normxgstsg3freqs8.flatten(order='F')
lista_F9G3 = normxgstsg3freqs9.flatten(order='F')
lista_F10G3 = normxgstsg3freqs10.flatten(order='F')
lista_F11G3 = normxgstsg3freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G3 = mean(lista_F1G3)
Promedio_lista_F2G3 = mean(lista_F2G3)
Promedio_lista_F3G3 = mean(lista_F3G3)
Promedio_lista_F4G3 = mean(lista_F4G3)
Promedio_lista_F5G3 = mean(lista_F5G3)
Promedio_lista_F6G3 = mean(lista_F6G3)
Promedio_lista_F7G3 = mean(lista_F7G3)
Promedio_lista_F8G3 = mean(lista_F8G3)
Promedio_lista_F9G3 = mean(lista_F9G3)
Promedio_lista_F10G3 = mean(lista_F10G3)
Promedio_lista_F11G3 = mean(lista_F11G3)
#Lista de Promedios
PromediosG3 = []
PromediosG3.append(Promedio_lista_F1G3)
PromediosG3.append(Promedio_lista_F2G3)
PromediosG3.append(Promedio_lista_F3G3)
PromediosG3.append(Promedio_lista_F4G3)
PromediosG3.append(Promedio_lista_F5G3)
PromediosG3.append(Promedio_lista_F6G3)
PromediosG3.append(Promedio_lista_F7G3)
PromediosG3.append(Promedio_lista_F8G3)
PromediosG3.append(Promedio_lista_F9G3)
PromediosG3.append(Promedio_lista_F10G3)
PromediosG3.append(Promedio_lista_F11G3)  
 
#Segmentación G4
normxgstsg4freqs1 = normxgsts[3800:4000,17560:19224]
normxgstsg4freqs2 = normxgsts[3600:3800,17560:19224]
normxgstsg4freqs3 = normxgsts[3400:3600,17560:19224]
normxgstsg4freqs4 = normxgsts[3200:3400,17560:19224]
normxgstsg4freqs5 = normxgsts[3000:3200,17560:19224]
normxgstsg4freqs6 = normxgsts[2800:3000,17560:19224]
normxgstsg4freqs7 = normxgsts[2600:2800,17560:19224]
normxgstsg4freqs8 = normxgsts[2400:2600,17560:19224]
normxgstsg4freqs9 = normxgsts[2200:2400,17560:19224]
normxgstsg4freqs10 = normxgsts[2000:2200,17560:19224]
normxgstsg4freqs11 = normxgsts[1800:2000,17560:19224]
#Transformación matrices de NumPy en vectores por columna
lista_F1G4 = normxgstsg4freqs1.flatten(order='F')
lista_F2G4 = normxgstsg4freqs2.flatten(order='F')
lista_F3G4 = normxgstsg4freqs3.flatten(order='F')
lista_F4G4 = normxgstsg4freqs4.flatten(order='F')
lista_F5G4 = normxgstsg4freqs5.flatten(order='F')
lista_F6G4 = normxgstsg4freqs6.flatten(order='F')
lista_F7G4 = normxgstsg4freqs7.flatten(order='F')
lista_F8G4 = normxgstsg4freqs8.flatten(order='F')
lista_F9G4 = normxgstsg4freqs9.flatten(order='F')
lista_F10G4 = normxgstsg4freqs10.flatten(order='F')
lista_F11G4 = normxgstsg4freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G4 = mean(lista_F1G4)
Promedio_lista_F2G4 = mean(lista_F2G4)
Promedio_lista_F3G4 = mean(lista_F3G4)
Promedio_lista_F4G4 = mean(lista_F4G4)
Promedio_lista_F5G4 = mean(lista_F5G4)
Promedio_lista_F6G4 = mean(lista_F6G4)
Promedio_lista_F7G4 = mean(lista_F7G4)
Promedio_lista_F8G4 = mean(lista_F8G4)
Promedio_lista_F9G4 = mean(lista_F9G4)
Promedio_lista_F10G4 = mean(lista_F10G4)
Promedio_lista_F11G4 = mean(lista_F11G4)
#Lista de Promedios
PromediosG4 = []
PromediosG4.append(Promedio_lista_F1G4)
PromediosG4.append(Promedio_lista_F2G4)
PromediosG4.append(Promedio_lista_F3G4)
PromediosG4.append(Promedio_lista_F4G4)
PromediosG4.append(Promedio_lista_F5G4)
PromediosG4.append(Promedio_lista_F6G4)
PromediosG4.append(Promedio_lista_F7G4)
PromediosG4.append(Promedio_lista_F8G4)
PromediosG4.append(Promedio_lista_F9G4)
PromediosG4.append(Promedio_lista_F10G4)
PromediosG4.append(Promedio_lista_F11G4)      
 
#Segmentación G5
normxgstsg5freqs1 = normxgsts[3800:4000,23394:25045] 
normxgstsg5freqs2 = normxgsts[3600:3800,23394:25045] 
normxgstsg5freqs3 = normxgsts[3400:3600,23394:25045] 
normxgstsg5freqs4 = normxgsts[3200:3400,23394:25045] 
normxgstsg5freqs5 = normxgsts[3000:3200,23394:25045] 
normxgstsg5freqs6 = normxgsts[2800:3000,23394:25045] 
normxgstsg5freqs7 = normxgsts[2600:2800,23394:25045] 
normxgstsg5freqs8 = normxgsts[2400:2600,23394:25045] 
normxgstsg5freqs9 = normxgsts[2200:2400,23394:25045] 
normxgstsg5freqs10 = normxgsts[2000:2200,23394:25045] 
normxgstsg5freqs11 = normxgsts[1800:2000,23394:25045] 
#Transformación matrices de NumPy en vectores por columna
lista_F1G5 = normxgstsg5freqs1.flatten(order='F')
lista_F2G5 = normxgstsg5freqs2.flatten(order='F')
lista_F3G5 = normxgstsg5freqs3.flatten(order='F')
lista_F4G5 = normxgstsg5freqs4.flatten(order='F')
lista_F5G5 = normxgstsg5freqs5.flatten(order='F')
lista_F6G5 = normxgstsg5freqs6.flatten(order='F')
lista_F7G5 = normxgstsg5freqs7.flatten(order='F')
lista_F8G5 = normxgstsg5freqs8.flatten(order='F')
lista_F9G5 = normxgstsg5freqs9.flatten(order='F')
lista_F10G5 = normxgstsg5freqs10.flatten(order='F')
lista_F11G5 = normxgstsg5freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G5 = mean(lista_F1G5)
Promedio_lista_F2G5 = mean(lista_F2G5)
Promedio_lista_F3G5 = mean(lista_F3G5)
Promedio_lista_F4G5 = mean(lista_F4G5)
Promedio_lista_F5G5 = mean(lista_F5G5)
Promedio_lista_F6G5 = mean(lista_F6G5)
Promedio_lista_F7G5 = mean(lista_F7G5)
Promedio_lista_F8G5 = mean(lista_F8G5)
Promedio_lista_F9G5 = mean(lista_F9G5)
Promedio_lista_F10G5 = mean(lista_F10G5)
Promedio_lista_F11G5 = mean(lista_F11G5)
#Lista de Promedios
PromediosG5 = []
PromediosG5.append(Promedio_lista_F1G5)
PromediosG5.append(Promedio_lista_F2G5)
PromediosG5.append(Promedio_lista_F3G5)
PromediosG5.append(Promedio_lista_F4G5)
PromediosG5.append(Promedio_lista_F5G5)
PromediosG5.append(Promedio_lista_F6G5)
PromediosG5.append(Promedio_lista_F7G5)
PromediosG5.append(Promedio_lista_F8G5)
PromediosG5.append(Promedio_lista_F9G5)
PromediosG5.append(Promedio_lista_F10G5)
PromediosG5.append(Promedio_lista_F11G5)
 
#Segmentación G6
normxgstsg6freqs1 = normxgsts[3800:4000,29597:31036]
normxgstsg6freqs2 = normxgsts[3600:3800,29597:31036]
normxgstsg6freqs3 = normxgsts[3400:3600,29597:31036]
normxgstsg6freqs4 = normxgsts[3200:3400,29597:31036]
normxgstsg6freqs5 = normxgsts[3000:3200,29597:31036]
normxgstsg6freqs6 = normxgsts[2800:3000,29597:31036]
normxgstsg6freqs7 = normxgsts[2600:2800,29597:31036]
normxgstsg6freqs8 = normxgsts[2400:2600,29597:31036]
normxgstsg6freqs9 = normxgsts[2200:2400,29597:31036]
normxgstsg6freqs10 = normxgsts[2000:2200,29597:31036]
normxgstsg6freqs11 = normxgsts[1800:2000,29597:31036]
#Transformación matrices de NumPy en vectores por columna
lista_F1G6 = normxgstsg6freqs1.flatten(order='F')
lista_F2G6 = normxgstsg6freqs2.flatten(order='F')
lista_F3G6 = normxgstsg6freqs3.flatten(order='F')
lista_F4G6 = normxgstsg6freqs4.flatten(order='F')
lista_F5G6 = normxgstsg6freqs5.flatten(order='F')
lista_F6G6 = normxgstsg6freqs6.flatten(order='F')
lista_F7G6 = normxgstsg6freqs7.flatten(order='F')
lista_F8G6 = normxgstsg6freqs8.flatten(order='F')
lista_F9G6 = normxgstsg6freqs9.flatten(order='F')
lista_F10G6 = normxgstsg6freqs10.flatten(order='F')
lista_F11G6 = normxgstsg6freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G6 = mean(lista_F1G6)
Promedio_lista_F2G6 = mean(lista_F2G6)
Promedio_lista_F3G6 = mean(lista_F3G6)
Promedio_lista_F4G6 = mean(lista_F4G6)
Promedio_lista_F5G6 = mean(lista_F5G6)
Promedio_lista_F6G6 = mean(lista_F6G6)
Promedio_lista_F7G6 = mean(lista_F7G6)
Promedio_lista_F8G6 = mean(lista_F8G6)
Promedio_lista_F9G6 = mean(lista_F9G6)
Promedio_lista_F10G6 = mean(lista_F10G6)
Promedio_lista_F11G6 = mean(lista_F11G6)
#Lista de Promedios
PromediosG6 = []
PromediosG6.append(Promedio_lista_F1G6)
PromediosG6.append(Promedio_lista_F2G6)
PromediosG6.append(Promedio_lista_F3G6)
PromediosG6.append(Promedio_lista_F4G6)
PromediosG6.append(Promedio_lista_F5G6)
PromediosG6.append(Promedio_lista_F6G6)
PromediosG6.append(Promedio_lista_F7G6)
PromediosG6.append(Promedio_lista_F8G6)
PromediosG6.append(Promedio_lista_F9G6)
PromediosG6.append(Promedio_lista_F10G6)
PromediosG6.append(Promedio_lista_F11G6)
 
#Segmentación G7
normxgstsg7freqs1 = normxgsts[3800:4000,34043:35613]
normxgstsg7freqs2 = normxgsts[3600:3800,34043:35613]
normxgstsg7freqs3 = normxgsts[3400:3600,34043:35613]
normxgstsg7freqs4 = normxgsts[3200:3400,34043:35613]
normxgstsg7freqs5 = normxgsts[3000:3200,34043:35613]
normxgstsg7freqs6 = normxgsts[2800:3000,34043:35613]
normxgstsg7freqs7 = normxgsts[2600:2800,34043:35613]
normxgstsg7freqs8 = normxgsts[2400:2600,34043:35613]
normxgstsg7freqs9 = normxgsts[2200:2400,34043:35613]
normxgstsg7freqs10 = normxgsts[2000:2200,34043:35613]
normxgstsg7freqs11 = normxgsts[1800:2000,34043:35613]
#Transformación matrices de NumPy en vectores por columna
lista_F1G7 = normxgstsg7freqs1.flatten(order='F')
lista_F2G7 = normxgstsg7freqs2.flatten(order='F')
lista_F3G7 = normxgstsg7freqs3.flatten(order='F')
lista_F4G7 = normxgstsg7freqs4.flatten(order='F')
lista_F5G7 = normxgstsg7freqs5.flatten(order='F')
lista_F6G7 = normxgstsg7freqs6.flatten(order='F')
lista_F7G7 = normxgstsg7freqs7.flatten(order='F')
lista_F8G7 = normxgstsg7freqs8.flatten(order='F')
lista_F9G7 = normxgstsg7freqs9.flatten(order='F')
lista_F10G7 = normxgstsg7freqs10.flatten(order='F')
lista_F11G7 = normxgstsg7freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G7 = mean(lista_F1G7)
Promedio_lista_F2G7 = mean(lista_F2G7)
Promedio_lista_F3G7 = mean(lista_F3G7)
Promedio_lista_F4G7 = mean(lista_F4G7)
Promedio_lista_F5G7 = mean(lista_F5G7)
Promedio_lista_F6G7 = mean(lista_F6G7)
Promedio_lista_F7G7 = mean(lista_F7G7)
Promedio_lista_F8G7 = mean(lista_F8G7)
Promedio_lista_F9G7 = mean(lista_F9G7)
Promedio_lista_F10G7 = mean(lista_F10G7)
Promedio_lista_F11G7 = mean(lista_F11G7)
#Lista de Promedios
PromediosG7 = []
PromediosG7.append(Promedio_lista_F1G7)
PromediosG7.append(Promedio_lista_F2G7)
PromediosG7.append(Promedio_lista_F3G7)
PromediosG7.append(Promedio_lista_F4G7)
PromediosG7.append(Promedio_lista_F5G7)
PromediosG7.append(Promedio_lista_F6G7)
PromediosG7.append(Promedio_lista_F7G7)
PromediosG7.append(Promedio_lista_F8G7)
PromediosG7.append(Promedio_lista_F9G7)
PromediosG7.append(Promedio_lista_F10G7)
PromediosG7.append(Promedio_lista_F11G7)
 
#Segmentación G8
normxgstsg8freqs1 = normxgsts[3800:4000,37688:39625] 
normxgstsg8freqs2 = normxgsts[3600:3800,37688:39625] 
normxgstsg8freqs3 = normxgsts[3400:3600,37688:39625] 
normxgstsg8freqs4 = normxgsts[3200:3400,37688:39625] 
normxgstsg8freqs5 = normxgsts[3000:3200,37688:39625] 
normxgstsg8freqs6 = normxgsts[2800:3000,37688:39625] 
normxgstsg8freqs7 = normxgsts[2600:2800,37688:39625] 
normxgstsg8freqs8 = normxgsts[2400:2600,37688:39625] 
normxgstsg8freqs9 = normxgsts[2200:2400,37688:39625] 
normxgstsg8freqs10 = normxgsts[2000:2200,37688:39625] 
normxgstsg8freqs11 = normxgsts[1800:2000,37688:39625] 
#Transformación matrices de NumPy en vectores por columna
lista_F1G8 = normxgstsg8freqs1.flatten(order='F')
lista_F2G8 = normxgstsg8freqs2.flatten(order='F')
lista_F3G8 = normxgstsg8freqs3.flatten(order='F')
lista_F4G8 = normxgstsg8freqs4.flatten(order='F')
lista_F5G8 = normxgstsg8freqs5.flatten(order='F')
lista_F6G8 = normxgstsg8freqs6.flatten(order='F')
lista_F7G8 = normxgstsg8freqs7.flatten(order='F')
lista_F8G8 = normxgstsg8freqs8.flatten(order='F')
lista_F9G8 = normxgstsg8freqs9.flatten(order='F')
lista_F10G8 = normxgstsg8freqs10.flatten(order='F')
lista_F11G8 = normxgstsg8freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G8 = mean(lista_F1G8)
Promedio_lista_F2G8 = mean(lista_F2G8)
Promedio_lista_F3G8 = mean(lista_F3G8)
Promedio_lista_F4G8 = mean(lista_F4G8)
Promedio_lista_F5G8 = mean(lista_F5G8)
Promedio_lista_F6G8 = mean(lista_F6G8)
Promedio_lista_F7G8 = mean(lista_F7G8)
Promedio_lista_F8G8 = mean(lista_F8G8)
Promedio_lista_F9G8 = mean(lista_F9G8)
Promedio_lista_F10G8 = mean(lista_F10G8)
Promedio_lista_F11G8 = mean(lista_F11G8)
#Lista de Promedios
PromediosG8 = []
PromediosG8.append(Promedio_lista_F1G8)
PromediosG8.append(Promedio_lista_F2G8)
PromediosG8.append(Promedio_lista_F3G8)
PromediosG8.append(Promedio_lista_F4G8)
PromediosG8.append(Promedio_lista_F5G8)
PromediosG8.append(Promedio_lista_F6G8)
PromediosG8.append(Promedio_lista_F7G8)
PromediosG8.append(Promedio_lista_F8G8)
PromediosG8.append(Promedio_lista_F9G8)
PromediosG8.append(Promedio_lista_F10G8)
PromediosG8.append(Promedio_lista_F11G8)
 
#Segmentación G9
normxgstsg9freqs1 = normxgsts[3800:4000,43804:45577]
normxgstsg9freqs2 = normxgsts[3600:3800,43804:45577]
normxgstsg9freqs3 = normxgsts[3400:3600,43804:45577]
normxgstsg9freqs4 = normxgsts[3200:3400,43804:45577]
normxgstsg9freqs5 = normxgsts[3000:3200,43804:45577]
normxgstsg9freqs6 = normxgsts[2800:3000,43804:45577]
normxgstsg9freqs7 = normxgsts[2600:2800,43804:45577]
normxgstsg9freqs8 = normxgsts[2400:2600,43804:45577]
normxgstsg9freqs9 = normxgsts[2200:2400,43804:45577]
normxgstsg9freqs10 = normxgsts[2000:2200,43804:45577]
normxgstsg9freqs11 = normxgsts[1800:2000,43804:45577]
#Transformación matrices de NumPy en vectores por columna
lista_F1G9 = normxgstsg9freqs1.flatten(order='F')
lista_F2G9 = normxgstsg9freqs2.flatten(order='F')
lista_F3G9 = normxgstsg9freqs3.flatten(order='F')
lista_F4G9 = normxgstsg9freqs4.flatten(order='F')
lista_F5G9 = normxgstsg9freqs5.flatten(order='F')
lista_F6G9 = normxgstsg9freqs6.flatten(order='F')
lista_F7G9 = normxgstsg9freqs7.flatten(order='F')
lista_F8G9 = normxgstsg9freqs8.flatten(order='F')
lista_F9G9 = normxgstsg9freqs9.flatten(order='F')
lista_F10G9 = normxgstsg9freqs10.flatten(order='F')
lista_F11G9 = normxgstsg9freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G9 = mean(lista_F1G9)
Promedio_lista_F2G9 = mean(lista_F2G9)
Promedio_lista_F3G9 = mean(lista_F3G9)
Promedio_lista_F4G9 = mean(lista_F4G9)
Promedio_lista_F5G9 = mean(lista_F5G9)
Promedio_lista_F6G9 = mean(lista_F6G9)
Promedio_lista_F7G9 = mean(lista_F7G9)
Promedio_lista_F8G9 = mean(lista_F8G9)
Promedio_lista_F9G9 = mean(lista_F9G9)
Promedio_lista_F10G9 = mean(lista_F10G9)
Promedio_lista_F11G9 = mean(lista_F11G9)
#Lista de Promedios
PromediosG9 = []
PromediosG9.append(Promedio_lista_F1G9)
PromediosG9.append(Promedio_lista_F2G9)
PromediosG9.append(Promedio_lista_F3G9)
PromediosG9.append(Promedio_lista_F4G9)
PromediosG9.append(Promedio_lista_F5G9)
PromediosG9.append(Promedio_lista_F6G9)
PromediosG9.append(Promedio_lista_F7G9)
PromediosG9.append(Promedio_lista_F8G9)
PromediosG9.append(Promedio_lista_F9G9)
PromediosG9.append(Promedio_lista_F10G9)
PromediosG9.append(Promedio_lista_F11G9)
 
#Segmentación G10
normxgstsg10freqs1 = normxgsts[3800:4000,49213:50579]
normxgstsg10freqs2 = normxgsts[3600:3800,49213:50579]
normxgstsg10freqs3 = normxgsts[3400:3600,49213:50579]
normxgstsg10freqs4 = normxgsts[3200:3400,49213:50579]
normxgstsg10freqs5 = normxgsts[3000:3200,49213:50579]
normxgstsg10freqs6 = normxgsts[2800:3000,49213:50579]
normxgstsg10freqs7 = normxgsts[2600:2800,49213:50579]
normxgstsg10freqs8 = normxgsts[2400:2600,49213:50579]
normxgstsg10freqs9 = normxgsts[2200:2400,49213:50579]
normxgstsg10freqs10 = normxgsts[2000:2200,49213:50579]
normxgstsg10freqs11 = normxgsts[1800:2000,49213:50579]
#Transformación matrices de NumPy en vectores por columna
lista_F1G10 = normxgstsg10freqs1.flatten(order='F')
lista_F2G10 = normxgstsg10freqs2.flatten(order='F')
lista_F3G10 = normxgstsg10freqs3.flatten(order='F')
lista_F4G10 = normxgstsg10freqs4.flatten(order='F')
lista_F5G10 = normxgstsg10freqs5.flatten(order='F')
lista_F6G10 = normxgstsg10freqs6.flatten(order='F')
lista_F7G10 = normxgstsg10freqs7.flatten(order='F')
lista_F8G10 = normxgstsg10freqs8.flatten(order='F')
lista_F9G10 = normxgstsg10freqs9.flatten(order='F')
lista_F10G10 = normxgstsg10freqs10.flatten(order='F')
lista_F11G10 = normxgstsg10freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G10 = mean(lista_F1G10)
Promedio_lista_F2G10 = mean(lista_F2G10)
Promedio_lista_F3G10 = mean(lista_F3G10)
Promedio_lista_F4G10 = mean(lista_F4G10)
Promedio_lista_F5G10 = mean(lista_F5G10)
Promedio_lista_F6G10 = mean(lista_F6G10)
Promedio_lista_F7G10 = mean(lista_F7G10)
Promedio_lista_F8G10 = mean(lista_F8G10)
Promedio_lista_F9G10 = mean(lista_F9G10)
Promedio_lista_F10G10 = mean(lista_F10G10)
Promedio_lista_F11G10 = mean(lista_F11G10)
#Lista de Promedios
PromediosG10 = []
PromediosG10.append(Promedio_lista_F1G10)
PromediosG10.append(Promedio_lista_F2G10)
PromediosG10.append(Promedio_lista_F3G10)
PromediosG10.append(Promedio_lista_F4G10)
PromediosG10.append(Promedio_lista_F5G10)
PromediosG10.append(Promedio_lista_F6G10)
PromediosG10.append(Promedio_lista_F7G10)
PromediosG10.append(Promedio_lista_F8G10)
PromediosG10.append(Promedio_lista_F9G10)
PromediosG10.append(Promedio_lista_F10G10)
PromediosG10.append(Promedio_lista_F11G10)
 
#Segmentación G11
normxgstsg11freqs1 = normxgsts[3800:4000,54123:55691]
normxgstsg11freqs2 = normxgsts[3600:3800,54123:55691]
normxgstsg11freqs3 = normxgsts[3400:3600,54123:55691]
normxgstsg11freqs4 = normxgsts[3200:3400,54123:55691]
normxgstsg11freqs5 = normxgsts[3000:3200,54123:55691]
normxgstsg11freqs6 = normxgsts[2800:3000,54123:55691]
normxgstsg11freqs7 = normxgsts[2600:2800,54123:55691]
normxgstsg11freqs8 = normxgsts[2400:2600,54123:55691]
normxgstsg11freqs9 = normxgsts[2200:2400,54123:55691]
normxgstsg11freqs10 = normxgsts[2000:2200,54123:55691]
normxgstsg11freqs11 = normxgsts[1800:2000,54123:55691]
#Transformación matrices de NumPy en vectores por columna
lista_F1G11 = normxgstsg11freqs1.flatten(order='F')
lista_F2G11 = normxgstsg11freqs2.flatten(order='F')
lista_F3G11 = normxgstsg11freqs3.flatten(order='F')
lista_F4G11 = normxgstsg11freqs4.flatten(order='F')
lista_F5G11 = normxgstsg11freqs5.flatten(order='F')
lista_F6G11 = normxgstsg11freqs6.flatten(order='F')
lista_F7G11 = normxgstsg11freqs7.flatten(order='F')
lista_F8G11 = normxgstsg11freqs8.flatten(order='F')
lista_F9G11 = normxgstsg11freqs9.flatten(order='F')
lista_F10G11 = normxgstsg11freqs10.flatten(order='F')
lista_F11G11 = normxgstsg11freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G11 = mean(lista_F1G11)
Promedio_lista_F2G11 = mean(lista_F2G11)
Promedio_lista_F3G11 = mean(lista_F3G11)
Promedio_lista_F4G11 = mean(lista_F4G11)
Promedio_lista_F5G11 = mean(lista_F5G11)
Promedio_lista_F6G11 = mean(lista_F6G11)
Promedio_lista_F7G11 = mean(lista_F7G11)
Promedio_lista_F8G11 = mean(lista_F8G11)
Promedio_lista_F9G11 = mean(lista_F9G11)
Promedio_lista_F10G11 = mean(lista_F10G11)
Promedio_lista_F11G11 = mean(lista_F11G11)
#Lista de Promedios
PromediosG11 = []
PromediosG11.append(Promedio_lista_F1G11)
PromediosG11.append(Promedio_lista_F2G11)
PromediosG11.append(Promedio_lista_F3G11)
PromediosG11.append(Promedio_lista_F4G11)
PromediosG11.append(Promedio_lista_F5G11)
PromediosG11.append(Promedio_lista_F6G11)
PromediosG11.append(Promedio_lista_F7G11)
PromediosG11.append(Promedio_lista_F8G11)
PromediosG11.append(Promedio_lista_F9G11)
PromediosG11.append(Promedio_lista_F10G11)
PromediosG11.append(Promedio_lista_F11G11)
 
#Segmentación G12
normxgstsg12freqs1 = normxgsts[3800:4000,59187:60809]
normxgstsg12freqs2 = normxgsts[3600:3800,59187:60809]
normxgstsg12freqs3 = normxgsts[3400:3600,59187:60809]
normxgstsg12freqs4 = normxgsts[3200:3400,59187:60809]
normxgstsg12freqs5 = normxgsts[3000:3200,59187:60809]
normxgstsg12freqs6 = normxgsts[2800:3000,59187:60809]
normxgstsg12freqs7 = normxgsts[2600:2800,59187:60809]
normxgstsg12freqs8 = normxgsts[2400:2600,59187:60809]
normxgstsg12freqs9 = normxgsts[2200:2400,59187:60809]
normxgstsg12freqs10 = normxgsts[2000:2200,59187:60809]
normxgstsg12freqs11 = normxgsts[1800:2000,59187:60809]
#Transformación matrices de NumPy en vectores por columna
lista_F1G12 = normxgstsg12freqs1.flatten(order='F')
lista_F2G12 = normxgstsg12freqs2.flatten(order='F')
lista_F3G12 = normxgstsg12freqs3.flatten(order='F')
lista_F4G12 = normxgstsg12freqs4.flatten(order='F')
lista_F5G12 = normxgstsg12freqs5.flatten(order='F')
lista_F6G12 = normxgstsg12freqs6.flatten(order='F')
lista_F7G12 = normxgstsg12freqs7.flatten(order='F')
lista_F8G12 = normxgstsg12freqs8.flatten(order='F')
lista_F9G12 = normxgstsg12freqs9.flatten(order='F')
lista_F10G12 = normxgstsg12freqs10.flatten(order='F')
lista_F11G12 = normxgstsg12freqs11.flatten(order='F')
#Valor Promedio
Promedio_lista_F1G12 = mean(lista_F1G12)
Promedio_lista_F2G12 = mean(lista_F2G12)
Promedio_lista_F3G12 = mean(lista_F3G12)
Promedio_lista_F4G12 = mean(lista_F4G12)
Promedio_lista_F5G12 = mean(lista_F5G12)
Promedio_lista_F6G12 = mean(lista_F6G12)
Promedio_lista_F7G12 = mean(lista_F7G12)
Promedio_lista_F8G12 = mean(lista_F8G12)
Promedio_lista_F9G12 = mean(lista_F9G12)
Promedio_lista_F10G12 = mean(lista_F10G12)
Promedio_lista_F11G12 = mean(lista_F11G12)
#Lista de Promedios
PromediosG12 = []
PromediosG12.append(Promedio_lista_F1G12)
PromediosG12.append(Promedio_lista_F2G12)
PromediosG12.append(Promedio_lista_F3G12)
PromediosG12.append(Promedio_lista_F4G12)
PromediosG12.append(Promedio_lista_F5G12)
PromediosG12.append(Promedio_lista_F6G12)
PromediosG12.append(Promedio_lista_F7G12)
PromediosG12.append(Promedio_lista_F8G12)
PromediosG12.append(Promedio_lista_F9G12)
PromediosG12.append(Promedio_lista_F10G12)
PromediosG12.append(Promedio_lista_F11G12)

Promedios = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6, 
             PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]
PromediosFrecuencias = np.array(Promedios); 
PromediosFrecuencias = np.transpose(PromediosFrecuencias)

DF = pd.DataFrame(PromediosFrecuencias)
#DF.to_csv("Promedios Paciente 9 Canal 2.csv")

PromediosG1 = np.array(PromediosG1)
PromediosG1 = np.transpose(PromediosG1)

PromediosG2 = np.array(PromediosG2)
PromediosG2 = np.transpose(PromediosG2)

PromediosG3 = np.array(PromediosG3)
PromediosG3 = np.transpose(PromediosG3)

PromediosG4 = np.array(PromediosG4)
PromediosG4 = np.transpose(PromediosG4)

PromediosG5 = np.array(PromediosG5)
PromediosG5 = np.transpose(PromediosG5)

PromediosG6 = np.array(PromediosG6)
PromediosG6 = np.transpose(PromediosG6)

PromediosG7 = np.array(PromediosG7)
PromediosG7 = np.transpose(PromediosG7)

PromediosG8 = np.array(PromediosG8)
PromediosG8 = np.transpose(PromediosG8)

PromediosG9 = np.array(PromediosG9)
PromediosG9 = np.transpose(PromediosG9)

PromediosG10 = np.array(PromediosG10)
PromediosG10 = np.transpose(PromediosG10)

PromediosG11 = np.array(PromediosG11)
PromediosG11 = np.transpose(PromediosG11)

PromediosG12 = np.array(PromediosG12)
PromediosG12 = np.transpose(PromediosG12)

data = [PromediosG1, PromediosG2, PromediosG3, PromediosG4, PromediosG5, PromediosG6,
        PromediosG7, PromediosG8, PromediosG9, PromediosG10, PromediosG11, PromediosG12]

#Secuencia Consecutiva de Matrices 
SPromediosG1 = list(itertools.chain(PromediosG1, PromediosG7))
SPromediosG1 = np.array(SPromediosG1)
SPromediosG2 = list(itertools.chain(PromediosG2, PromediosG8))
SPromediosG2 = np.array(SPromediosG2)
SPromediosG3 = list(itertools.chain(PromediosG3, PromediosG9))
SPromediosG3 = np.array(SPromediosG3)
SPromediosG4 = list(itertools.chain(PromediosG4, PromediosG10))
SPromediosG4 = np.array(SPromediosG4)
SPromediosG5 = list(itertools.chain(PromediosG5, PromediosG11))
SPromediosG5 = np.array(SPromediosG5)
SPromediosG6 = list(itertools.chain(PromediosG6, PromediosG12))
SPromediosG6 = np.array(SPromediosG6)
SPromedios = [SPromediosG1, SPromediosG2, SPromediosG3, SPromediosG4, SPromediosG5, SPromediosG6]

SPromediosU0 = list(itertools.chain(SPromedios[0], SPromedios[1]))
SPromediosU1 = list(itertools.chain(SPromediosU0, SPromedios[2]))
SPromediosU2 = list(itertools.chain(SPromediosU1, SPromedios[3]))
SPromediosU3 = list(itertools.chain(SPromediosU2, SPromedios[4]))
SPromediosU4 = list(itertools.chain(SPromediosU3, SPromedios[5]))
SPromediosU = np.array(SPromediosU4)

#Histograma
plt.figure(figsize =(16, 10))
plt.hist(SPromediosU)
plt.grid()
plt.title('Histograma de la Secuencia Consecutiva de Promedios')
plt.xlabel('Valor de la variable')
plt.ylabel('Conteo')
plt.savefig('Histograma Promedios Gesto 1 P3C6.png', dpi = 201)
plt.show()

#scipy stats Wilcoxon 
WCresultados = []
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[0], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[1], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[1], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[2], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[2], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)


stat, p_value = wilcoxon(SPromedios[3], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[3], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[4], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)
stat, p_value = wilcoxon(SPromedios[4], SPromedios[5], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)

stat, p_value = wilcoxon(SPromedios[5], SPromedios[0], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[1], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[2], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[3], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
stat, p_value = wilcoxon(SPromedios[5], SPromedios[4], zero_method='wilcox', alternative= 'two-sided')
WCresultados.append(p_value)
MCr = ''
WCresultados.append(MCr)

WCresultados = np.array(WCresultados)
WCresultados = np.reshape(WCresultados,[6,6])

DF = pd.DataFrame(WCresultados) 
DF.to_csv("Wilcoxon Resultados entre gestos distintos P9C2.csv")
