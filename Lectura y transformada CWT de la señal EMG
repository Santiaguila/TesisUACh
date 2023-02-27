# TRABAJO DE TÍTULO - SANTIAGO GUZMÁN
# LECTURA DE LA SEÑAL EMG
# PROCESAMIENTO DE SEÑAL - CWT

# 1. Librerias
import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
import glob
import os
import heapq
#Recorrido de la fila completa por columna
def obtener_columna(matriz, indice_columna):
	return [fila[indice_columna] if indice_columna < len(fila) else None for fila in matriz]

#Valores de cada columna para su posterior Normalización
def mat2gray(vector):
        V = vector - np.min(vector)
        V = V / np.max(vector)
        return V

# 2. Lectura de la señal EMG 
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

# 3. Diccionario de señales EMG
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

# 4.1. Electromiogragías Paciente 3
emgpacientes03 = [voltajesP03[1], voltajesP03[4], voltajesP03[5], voltajesP03[7]]
n = 4

emg_CanalPaciente = []
for k in range(len(emgpacientes03[0:n])):
    emg_funcional = np.array(emgpacientes03[k])
    emg_CanalPaciente.append(emg_funcional)
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
    #Separación de gestos por el tiempo
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
      
    #Saturación de valores Máximos
    porcentajedevalores = 0.001
    lista_G1 = gesto_1.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G2 = gesto_2.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G3 = gesto_3.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G4 = gesto_4.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G5 = gesto_5.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G6 = gesto_6.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G7 = gesto_7.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G8 = gesto_8.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G9 = gesto_9.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G10 = gesto_10.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G11 = gesto_11.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G12 = gesto_12.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G = [lista_G1, lista_G2, lista_G3, lista_G4, lista_G5, lista_G6,
               lista_G7, lista_G8, lista_G9, lista_G10, lista_G11, lista_G12]
    
    nrodevalores_G1 = int(len(lista_G[0])*porcentajedevalores)
    nrodevalores_G2 = int(len(lista_G[1])*porcentajedevalores)
    nrodevalores_G3 = int(len(lista_G[2])*porcentajedevalores)
    nrodevalores_G4 = int(len(lista_G[3])*porcentajedevalores)
    nrodevalores_G5 = int(len(lista_G[4])*porcentajedevalores)
    nrodevalores_G6 = int(len(lista_G[5])*porcentajedevalores)
    nrodevalores_G7 = int(len(lista_G[6])*porcentajedevalores)
    nrodevalores_G8 = int(len(lista_G[7])*porcentajedevalores)
    nrodevalores_G9 = int(len(lista_G[8])*porcentajedevalores)
    nrodevalores_G10 = int(len(lista_G[9])*porcentajedevalores)
    nrodevalores_G11 = int(len(lista_G[10])*porcentajedevalores)
    nrodevalores_G12 = int(len(lista_G[11])*porcentajedevalores)
    nrodevalores_G = [nrodevalores_G1, nrodevalores_G2, nrodevalores_G3, nrodevalores_G4, nrodevalores_G5, nrodevalores_G6,
                      nrodevalores_G7, nrodevalores_G8,nrodevalores_G9, nrodevalores_G10, nrodevalores_G11, nrodevalores_G12]
    
    busquedadevalores_G1 = heapq.nlargest(nrodevalores_G[0], lista_G[0])
    busquedadevalores_G2 = heapq.nlargest(nrodevalores_G[1], lista_G[1])
    busquedadevalores_G3 = heapq.nlargest(nrodevalores_G[2], lista_G[2])
    busquedadevalores_G4 = heapq.nlargest(nrodevalores_G[3], lista_G[3])
    busquedadevalores_G5 = heapq.nlargest(nrodevalores_G[4], lista_G[4])
    busquedadevalores_G6 = heapq.nlargest(nrodevalores_G[5], lista_G[5])
    busquedadevalores_G7 = heapq.nlargest(nrodevalores_G[6], lista_G[6])
    busquedadevalores_G8 = heapq.nlargest(nrodevalores_G[7], lista_G[7])
    busquedadevalores_G9 = heapq.nlargest(nrodevalores_G[8], lista_G[8])
    busquedadevalores_G10 = heapq.nlargest(nrodevalores_G[9], lista_G[9])
    busquedadevalores_G11 = heapq.nlargest(nrodevalores_G[10], lista_G[10])
    busquedadevalores_G12 = heapq.nlargest(nrodevalores_G[11], lista_G[11])
    busquedadevalores_G = [busquedadevalores_G1, busquedadevalores_G2, busquedadevalores_G3, busquedadevalores_G4, busquedadevalores_G5, busquedadevalores_G6,
                           busquedadevalores_G7, busquedadevalores_G8, busquedadevalores_G9, busquedadevalores_G10, busquedadevalores_G11, busquedadevalores_G12]

    lista_valoresmayores_G1 = np.array(busquedadevalores_G[0])
    lista_valoresmayores_G2 = np.array(busquedadevalores_G[1])
    lista_valoresmayores_G3 = np.array(busquedadevalores_G[2])
    lista_valoresmayores_G4 = np.array(busquedadevalores_G[3])
    lista_valoresmayores_G5 = np.array(busquedadevalores_G[4])
    lista_valoresmayores_G6 = np.array(busquedadevalores_G[5])
    lista_valoresmayores_G7 = np.array(busquedadevalores_G[6])
    lista_valoresmayores_G8 = np.array(busquedadevalores_G[7])
    lista_valoresmayores_G9 = np.array(busquedadevalores_G[8])
    lista_valoresmayores_G10 = np.array(busquedadevalores_G[9])
    lista_valoresmayores_G11 = np.array(busquedadevalores_G[10])
    lista_valoresmayores_G12 = np.array(busquedadevalores_G[11])
    lista_valoresmayores_G = [lista_valoresmayores_G1, lista_valoresmayores_G2, lista_valoresmayores_G3, lista_valoresmayores_G4, lista_valoresmayores_G5, lista_valoresmayores_G6,
                              lista_valoresmayores_G7, lista_valoresmayores_G8, lista_valoresmayores_G9, lista_valoresmayores_G10, lista_valoresmayores_G11, lista_valoresmayores_G12]
    
    valormin_G1 = np.min(lista_valoresmayores_G[0])
    valormin_G2 = np.min(lista_valoresmayores_G[1])
    valormin_G3 = np.min(lista_valoresmayores_G[2])
    valormin_G4 = np.min(lista_valoresmayores_G[3])
    valormin_G5 = np.min(lista_valoresmayores_G[4])
    valormin_G6 = np.min(lista_valoresmayores_G[5])
    valormin_G7 = np.min(lista_valoresmayores_G[6])
    valormin_G8 = np.min(lista_valoresmayores_G[7])
    valormin_G9 = np.min(lista_valoresmayores_G[8])
    valormin_G10 = np.min(lista_valoresmayores_G[9])
    valormin_G11 = np.min(lista_valoresmayores_G[10])
    valormin_G12 = np.min(lista_valoresmayores_G[11])
    valormin_G = [valormin_G1, valormin_G2, valormin_G3, valormin_G4, valormin_G5, valormin_G6, 
                  valormin_G7, valormin_G8, valormin_G9, valormin_G10, valormin_G11, valormin_G12]
    
    gesto_1[gesto_1 >= valormin_G[0]] = valormin_G[0]
    gesto_2[gesto_2 >= valormin_G[1]] = valormin_G[1]
    gesto_3[gesto_3 >= valormin_G[2]] = valormin_G[2]
    gesto_4[gesto_4 >= valormin_G[3]] = valormin_G[3]
    gesto_5[gesto_5 >= valormin_G[4]] = valormin_G[4]
    gesto_6[gesto_6 >= valormin_G[5]] = valormin_G[5]
    gesto_7[gesto_7 >= valormin_G[6]] = valormin_G[6]
    gesto_8[gesto_8 >= valormin_G[7]] = valormin_G[7]
    gesto_9[gesto_9 >= valormin_G[8]] = valormin_G[8]
    gesto_10[gesto_10 >= valormin_G[9]] = valormin_G[9]
    gesto_11[gesto_11 >= valormin_G[10]] = valormin_G[10]
    gesto_12[gesto_12 >= valormin_G[11]] = valormin_G[11]

    normxgsts = mat2gray(abscwt)
    normxgsts[:,0:762], normxgsts[:,2243:4044], normxgsts[:,5629:9069], normxgsts[:,11028:14567], normxgsts[:,16475:19794], normxgsts[:,22042:25666], normxgsts[:,27749:30557], normxgsts[:,32233:34108], normxgsts[:,35929:39641], normxgsts[:,41660:45362], normxgsts[:,47548:50807], normxgsts[:,52752:56003], normxgsts[:,57709:59107] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    normxgsts[:,763:2242] = mat2gray(gesto_1) 
    normxgsts[:,4045:5628] = mat2gray(gesto_2)
    normxgsts[:,9070:11027] = mat2gray(gesto_3)
    normxgsts[:,14568:16474] = mat2gray(gesto_4)
    normxgsts[:,19795:22041] = mat2gray(gesto_5)
    normxgsts[:,25667:27748] = mat2gray(gesto_6)
    normxgsts[:,30558:32232] = mat2gray(gesto_7)
    normxgsts[:,34109:35928] = mat2gray(gesto_8)
    normxgsts[:,39642:41659] = mat2gray(gesto_9)
    normxgsts[:,45363:47547] = mat2gray(gesto_10)
    normxgsts[:,50808:52751] = mat2gray(gesto_11)
    normxgsts[:,56004:57708] = mat2gray(gesto_12)
    
    fs = samplerate
    
    #Nombre de cada Gráfica
    pacientes = ['paciente 01 ','paciente 02 ','paciente 03 ','paciente 04 ','paciente 05 ','paciente 06 ','paciente 07 ','paciente 08 ','paciente 09 ','paciente 10 ']
    canales = ['canal 2 ','canal 5 ','canal 6 ','canal 8 ']    
    file_emg = 'gráfica EMG'
    file_cwt = 'gráfica CWT cmor1.0-1.5'
    file_end = 'gráfica CWT cmor1.0-1.5.png'
    file_end_2 = 'gráfica EMG.png'
    
    #Título de la Gr+afica
    title_emg = canales[k] + pacientes[2] + file_emg
    title_cwt = canales[k] + pacientes[2] + file_cwt
    
    #Nombre del archivo PNG
    file_name = canales[k] + pacientes[2] + file_end
    file_name_2 = canales[k] + pacientes[2] + file_end_2
    
    # Vectores de tiempo
    timeemg = np.arange(0, len(emg_funcional) / fs, 1 / fs)
    bdt03 = [0.763, 2.242, 4.045, 5.628, 9.07, 11.027, 14.568, 16.474, 19.795, 22.041, 25.667, 27.748,
             30.558, 32.232, 34.109, 35.928, 39.642, 41.659, 45.363, 47.547, 50.808, 52.751, 56.004, 57.708]
    # Gráficas EMG
    plt.figure(figsize=(16,10))
    plt.plot(timeemg, emg_funcional, 'b', label='Señal EMG RAW')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Voltaje [V]")
    plt.title(title_emg)
    plt.xlim(0, timeemg.max())
    plt.grid()
    #plt.axhline(y=0.01,color='k')
    plt.axvline(x=bdt03[0], color='c', linewidth=1.2, label="G1 - Mano en Reposo")
    plt.axvline(x=bdt03[1], color='c', linewidth=1.2)
    plt.axvline(x=bdt03[2], color='m', linewidth=1.2, label="G2 - Mano Cerrada en Puño")
    plt.axvline(x=bdt03[3], color='m', linewidth=1.2)
    plt.axvline(x=bdt03[4], color='y', linewidth=1.2, label="G3 - Flexión de la Muñeca")
    plt.axvline(x=bdt03[5], color='y', linewidth=1.2)
    plt.axvline(x=bdt03[6], color='r', linewidth=1.2, label="G4 - extensión de la Muñeca")
    plt.axvline(x=bdt03[7], color='r', linewidth=1.2)
    plt.axvline(x=bdt03[8], color='g', linewidth=1.2, label="G5 - Desviación Radial")
    plt.axvline(x=bdt03[9], color='g', linewidth=1.2)
    plt.axvline(x=bdt03[10], color='k', linewidth=1.2, label="G6 - Desviación Cubital")
    plt.axvline(x=bdt03[11], color='k', linewidth=1.2)
    plt.axvline(x=bdt03[12], color='c', linewidth=1.2)
    plt.axvline(x=bdt03[13], color='c', linewidth=1.2)
    plt.axvline(x=bdt03[14], color='m', linewidth=1.2)
    plt.axvline(x=bdt03[15], color='m', linewidth=1.2)
    plt.axvline(x=bdt03[16], color='y', linewidth=1.2)
    plt.axvline(x=bdt03[17], color='y', linewidth=1.2)
    plt.axvline(x=bdt03[18], color='r', linewidth=1.2)
    plt.axvline(x=bdt03[19], color='r', linewidth=1.2)
    plt.axvline(x=bdt03[20], color='g', linewidth=1.2)
    plt.axvline(x=bdt03[21], color='g', linewidth=1.2)
    plt.axvline(x=bdt03[22], color='k', linewidth=1.2)
    plt.axvline(x=bdt03[23], color='k', linewidth=1.2)
    plt.legend(loc='lower left', fontsize='x-small', borderpad=None)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useOffset=True)
    plt.savefig(file_name_2, dpi = 201)  
    plt.show()
    
    #Parametros Matemáticos Cambio de eje Tiempo
    reduction_to= int(np.floor(len(abscwt) / 4)) #Intervalos de Ejes
    time_axis = np.linspace(0.0, emg_funcional.shape[0] * np.array(dt), emg_funcional.shape[0]+1)
    time = np.arange(0, time_axis.shape[0] + 1, int(reduction_to*2.5))
    output_time_axis = time_axis[time]
    outtime = np.round(output_time_axis, 0)
    
    #Parametros Matemáticos Cambio de eje Frecuencia
    scales = range(0,num-1)
    step = 200 
    outscales = scales[0:num:step]
    outfreqs = np.round(frequencies[0:num:step], 2)

    # Gráfica de CWT Normalizada
    plt.figure(figsize=(16,10))
    plt.text(int(bdt03[0]*1000), int(len(outscales)+1500), 'G1', color='w')
    plt.text(int(bdt03[2]*1000), int(len(outscales)+1500), 'G2', color='w')
    plt.text(int(bdt03[4]*1000), int(len(outscales)+1500), 'G3', color='w')
    plt.text(int(bdt03[6]*1000), int(len(outscales)+1500), 'G4', color='w')
    plt.text(int(bdt03[8]*1000), int(len(outscales)+1500), 'G5', color='w')
    plt.text(int(bdt03[10]*1000), int(len(outscales)+1500), 'G6', color='w')
    plt.text(int(bdt03[12]*1000), int(len(outscales)+1500), 'G1', color='w')
    plt.text(int(bdt03[14]*1000), int(len(outscales)+1500), 'G2', color='w')
    plt.text(int(bdt03[16]*1000), int(len(outscales)+1500), 'G3', color='w')
    plt.text(int(bdt03[18]*1000), int(len(outscales)+1500), 'G4', color='w')
    plt.text(int(bdt03[20]*1000), int(len(outscales)+1500), 'G5', color='w')
    plt.text(int(bdt03[22]*1000), int(len(outscales)+1500), 'G6', color='w')
    plt.imshow(normxgsts, cmap='jet', aspect ='auto')
    plt.xticks(time, outtime)
    plt.yticks(outscales,outfreqs)
    plt.ylim(top=1100)
    #plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.title(title_cwt)
    plt.savefig(file_name, dpi = 201)
    plt.show()

# 4.2. Electromiogragías Paciente 5
emgpacientes05 = [voltajesP05[3], voltajesP05[5]]
n = 2
emg_CanalPaciente = []
for k in range(len(emgpacientes05[0:n])):
    emg_funcional = np.array(emgpacientes05[k])
    emg_CanalPaciente.append(emg_funcional)
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
    #Separación de gestos por el tiempo
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
    
    #Saturación de valores Máximos
    porcentajedevalores = 0.001
    lista_G1 = gesto_1.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G2 = gesto_2.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G3 = gesto_3.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G4 = gesto_4.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G5 = gesto_5.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G6 = gesto_6.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G7 = gesto_7.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G8 = gesto_8.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G9 = gesto_9.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G10 = gesto_10.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G11 = gesto_11.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G12 = gesto_12.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G13 = gesto_13.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G14 = gesto_14.flatten(order='F') # transformar matrices de NumPy en vectores por columna
    lista_G = [lista_G1, lista_G2, lista_G3, lista_G4, lista_G5, lista_G6,
               lista_G7, lista_G8, lista_G9, lista_G10, lista_G11, lista_G12,
               lista_G13, lista_G14]
    
    nrodevalores_G1 = int(len(lista_G[0])*porcentajedevalores)
    nrodevalores_G2 = int(len(lista_G[1])*porcentajedevalores)
    nrodevalores_G3 = int(len(lista_G[2])*porcentajedevalores)
    nrodevalores_G4 = int(len(lista_G[3])*porcentajedevalores)
    nrodevalores_G5 = int(len(lista_G[4])*porcentajedevalores)
    nrodevalores_G6 = int(len(lista_G[5])*porcentajedevalores)
    nrodevalores_G7 = int(len(lista_G[6])*porcentajedevalores)
    nrodevalores_G8 = int(len(lista_G[7])*porcentajedevalores)
    nrodevalores_G9 = int(len(lista_G[8])*porcentajedevalores)
    nrodevalores_G10 = int(len(lista_G[9])*porcentajedevalores)
    nrodevalores_G11 = int(len(lista_G[10])*porcentajedevalores)
    nrodevalores_G12 = int(len(lista_G[11])*porcentajedevalores)
    nrodevalores_G13 = int(len(lista_G[12])*porcentajedevalores)
    nrodevalores_G14 = int(len(lista_G[13])*porcentajedevalores)
    
    nrodevalores_G = [nrodevalores_G1, nrodevalores_G2, nrodevalores_G3, nrodevalores_G4, nrodevalores_G5, nrodevalores_G6,
                      nrodevalores_G7, nrodevalores_G8,nrodevalores_G9, nrodevalores_G10, nrodevalores_G11, nrodevalores_G12,
                      nrodevalores_G13, nrodevalores_G14]
    
    busquedadevalores_G1 = heapq.nlargest(nrodevalores_G[0], lista_G[0])
    busquedadevalores_G2 = heapq.nlargest(nrodevalores_G[1], lista_G[1])
    busquedadevalores_G3 = heapq.nlargest(nrodevalores_G[2], lista_G[2])
    busquedadevalores_G4 = heapq.nlargest(nrodevalores_G[3], lista_G[3])
    busquedadevalores_G5 = heapq.nlargest(nrodevalores_G[4], lista_G[4])
    busquedadevalores_G6 = heapq.nlargest(nrodevalores_G[5], lista_G[5])
    busquedadevalores_G7 = heapq.nlargest(nrodevalores_G[6], lista_G[6])
    busquedadevalores_G8 = heapq.nlargest(nrodevalores_G[7], lista_G[7])
    busquedadevalores_G9 = heapq.nlargest(nrodevalores_G[8], lista_G[8])
    busquedadevalores_G10 = heapq.nlargest(nrodevalores_G[9], lista_G[9])
    busquedadevalores_G11 = heapq.nlargest(nrodevalores_G[10], lista_G[10])
    busquedadevalores_G12 = heapq.nlargest(nrodevalores_G[11], lista_G[11])
    busquedadevalores_G13 = heapq.nlargest(nrodevalores_G[12], lista_G[12])
    busquedadevalores_G14 = heapq.nlargest(nrodevalores_G[13], lista_G[13])
    busquedadevalores_G = [busquedadevalores_G1, busquedadevalores_G2, busquedadevalores_G3, busquedadevalores_G4, busquedadevalores_G5, busquedadevalores_G6,
                           busquedadevalores_G7, busquedadevalores_G8, busquedadevalores_G9, busquedadevalores_G10, busquedadevalores_G11, busquedadevalores_G12,
                           busquedadevalores_G13, busquedadevalores_G14]

    lista_valoresmayores_G1 = np.array(busquedadevalores_G[0])
    lista_valoresmayores_G2 = np.array(busquedadevalores_G[1])
    lista_valoresmayores_G3 = np.array(busquedadevalores_G[2])
    lista_valoresmayores_G4 = np.array(busquedadevalores_G[3])
    lista_valoresmayores_G5 = np.array(busquedadevalores_G[4])
    lista_valoresmayores_G6 = np.array(busquedadevalores_G[5])
    lista_valoresmayores_G7 = np.array(busquedadevalores_G[6])
    lista_valoresmayores_G8 = np.array(busquedadevalores_G[7])
    lista_valoresmayores_G9 = np.array(busquedadevalores_G[8])
    lista_valoresmayores_G10 = np.array(busquedadevalores_G[9])
    lista_valoresmayores_G11 = np.array(busquedadevalores_G[10])
    lista_valoresmayores_G12 = np.array(busquedadevalores_G[11])
    lista_valoresmayores_G13 = np.array(busquedadevalores_G[12])
    lista_valoresmayores_G14 = np.array(busquedadevalores_G[13])
    lista_valoresmayores_G = [lista_valoresmayores_G1, lista_valoresmayores_G2, lista_valoresmayores_G3, lista_valoresmayores_G4, lista_valoresmayores_G5, lista_valoresmayores_G6,
                              lista_valoresmayores_G7, lista_valoresmayores_G8, lista_valoresmayores_G9, lista_valoresmayores_G10, lista_valoresmayores_G11, lista_valoresmayores_G12,
                              lista_valoresmayores_G13, lista_valoresmayores_G14]
    
    valormin_G1 = np.min(lista_valoresmayores_G[0])
    valormin_G2 = np.min(lista_valoresmayores_G[1])
    valormin_G3 = np.min(lista_valoresmayores_G[2])
    valormin_G4 = np.min(lista_valoresmayores_G[3])
    valormin_G5 = np.min(lista_valoresmayores_G[4])
    valormin_G6 = np.min(lista_valoresmayores_G[5])
    valormin_G7 = np.min(lista_valoresmayores_G[6])
    valormin_G8 = np.min(lista_valoresmayores_G[7])
    valormin_G9 = np.min(lista_valoresmayores_G[8])
    valormin_G10 = np.min(lista_valoresmayores_G[9])
    valormin_G11 = np.min(lista_valoresmayores_G[10])
    valormin_G12 = np.min(lista_valoresmayores_G[11])
    valormin_G13 = np.min(lista_valoresmayores_G[12])
    valormin_G14 = np.min(lista_valoresmayores_G[13])
    valormin_G = [valormin_G1, valormin_G2, valormin_G3, valormin_G4, valormin_G5, valormin_G6, 
                  valormin_G7, valormin_G8, valormin_G9, valormin_G10, valormin_G11, valormin_G12,
                  valormin_G13, valormin_G14]
    
    gesto_1[gesto_1 >= valormin_G[0]] = valormin_G[0]
    gesto_2[gesto_2 >= valormin_G[1]] = valormin_G[1]
    gesto_3[gesto_3 >= valormin_G[2]] = valormin_G[2]
    gesto_4[gesto_4 >= valormin_G[3]] = valormin_G[3]
    gesto_5[gesto_5 >= valormin_G[4]] = valormin_G[4]
    gesto_6[gesto_6 >= valormin_G[5]] = valormin_G[5]
    gesto_7[gesto_7 >= valormin_G[6]] = valormin_G[6]
    gesto_8[gesto_8 >= valormin_G[7]] = valormin_G[7]
    gesto_9[gesto_9 >= valormin_G[8]] = valormin_G[8]
    gesto_10[gesto_10 >= valormin_G[9]] = valormin_G[9]
    gesto_11[gesto_11 >= valormin_G[10]] = valormin_G[10]
    gesto_12[gesto_12 >= valormin_G[11]] = valormin_G[11]
    gesto_13[gesto_13 >= valormin_G[12]] = valormin_G[12]
    gesto_14[gesto_14 >= valormin_G[13]] = valormin_G[13]
             
    normxgsts = mat2gray(abscwt)
    normxgsts[:,0:1868], normxgsts[:,3845:5799], normxgsts[:,7642:11453], normxgsts[:,13232:16878], normxgsts[:,18595:22381], normxgsts[:,24009:27710], normxgsts[:,29313:33285], normxgsts[:,35048:37774], normxgsts[:,39819:41664], normxgsts[:,43446:47529], normxgsts[:,49203:53230], normxgsts[:,55022:58950], normxgsts[:,60722:64905], normxgsts[:,66761:70096], normxgsts[:,71993:74680] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    normxgsts[:,1869:3844] = mat2gray(gesto_1) 
    normxgsts[:,5800:7641] = mat2gray(gesto_2)
    normxgsts[:,11454:13231] = mat2gray(gesto_3)
    normxgsts[:,16878:18594] = mat2gray(gesto_4)
    normxgsts[:,22382:24008] = mat2gray(gesto_5)
    normxgsts[:,27711:29312] = mat2gray(gesto_6)
    normxgsts[:,33286:35047] = mat2gray(gesto_7)
    normxgsts[:,37775:39818] = mat2gray(gesto_8)
    normxgsts[:,41665:43445] = mat2gray(gesto_9)
    normxgsts[:,47530:49202] = mat2gray(gesto_10)
    normxgsts[:,53231:55021] = mat2gray(gesto_11)
    normxgsts[:,58951:60721] = mat2gray(gesto_12)
    normxgsts[:,64906:66760] = mat2gray(gesto_13)
    normxgsts[:,70097:71992] = mat2gray(gesto_14)
    
    fs = samplerate
    
    #Nombre de cada Gráfica
    pacientes = ['paciente 01 ','paciente 02 ','paciente 03 ','paciente 04 ','paciente 05 ','paciente 06 ','paciente 07 ','paciente 08 ','paciente 09 ','paciente 10 ']
    canales = ['canal 4 ','canal 6 ']    
    file_emg = 'gráfica EMG'
    file_cwt = 'gráfica CWT cmor1.0-1.5'
    file_end = 'gráfica CWT cmor1.0-1.5.png'
    file_end_2 = 'gráfica EMG.png'
    
    #Título de la Gr+afica
    title_emg = canales[k] + pacientes[4] + file_emg
    title_cwt = canales[k] + pacientes[4] + file_cwt
    
    #Nombre del archivo PNG
    file_name = canales[k] + pacientes[4] + file_end
    file_name_2 = canales[k] + pacientes[4] + file_end_2
    
    # Vectores de tiempo
    timeemg = np.arange(0, len(emg_funcional) / fs, 1 / fs)
    bdt05 = [1.869, 3.844, 5.80, 7.641, 11.454, 13.231, 16.878, 18.594, 22.382, 24.008, 27.711, 29.312, 33.286, 35.047, 
             37.775, 39.818, 41.665, 43.445, 47.530, 49.202, 53.231, 55.021, 58.951, 60.721, 64.906, 66.760, 70.097, 71.992]
    # Gráficas EMG
    plt.figure(figsize=(16,10))
    plt.plot(timeemg, emg_funcional, 'b', label='Señal EMG RAW')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Voltaje [V]")
    plt.title(title_emg)
    plt.xlim(0, timeemg.max())
    plt.grid()
    #plt.axhline(y=0.01,color='k')
    plt.axvline(x=bdt05[0], color='c', linewidth=1.2, label="G1 - Mano en Reposo")
    plt.axvline(x=bdt05[1], color='c', linewidth=1.2)
    plt.axvline(x=bdt05[2], color='m', linewidth=1.2, label="G2 - Mano Cerrada en Puño")
    plt.axvline(x=bdt05[3], color='m', linewidth=1.2)
    plt.axvline(x=bdt05[4], color='y', linewidth=1.2, label="G3 - Flexión de la Muñeca")
    plt.axvline(x=bdt05[5], color='y', linewidth=1.2)
    plt.axvline(x=bdt05[6], color='r', linewidth=1.2, label="G4 - extensión de la Muñeca")
    plt.axvline(x=bdt05[7], color='r', linewidth=1.2)
    plt.axvline(x=bdt05[8], color='g', linewidth=1.2, label="G5 - Desviación Radial")
    plt.axvline(x=bdt05[9], color='g', linewidth=1.2)
    plt.axvline(x=bdt05[10], color='k', linewidth=1.2, label="G6 - Desviación Cubital")
    plt.axvline(x=bdt05[11], color='k', linewidth=1.2)
    plt.axvline(x=bdt05[12], color='orange', linewidth=1.2, label="G7 - Palma Extendida")
    plt.axvline(x=bdt05[13], color='orange', linewidth=1.2)
    plt.axvline(x=bdt05[14], color='c', linewidth=1.2)
    plt.axvline(x=bdt05[15], color='c', linewidth=1.2)
    plt.axvline(x=bdt05[16], color='m', linewidth=1.2)
    plt.axvline(x=bdt05[17], color='m', linewidth=1.2)
    plt.axvline(x=bdt05[18], color='y', linewidth=1.2)
    plt.axvline(x=bdt05[19], color='y', linewidth=1.2)
    plt.axvline(x=bdt05[20], color='r', linewidth=1.2)
    plt.axvline(x=bdt05[21], color='r', linewidth=1.2)
    plt.axvline(x=bdt05[22], color='g', linewidth=1.2)
    plt.axvline(x=bdt05[23], color='g', linewidth=1.2)
    plt.axvline(x=bdt05[24], color='k', linewidth=1.2)
    plt.axvline(x=bdt05[25], color='k', linewidth=1.2)
    plt.axvline(x=bdt05[26], color='orange', linewidth=1.2)
    plt.axvline(x=bdt05[27], color='orange', linewidth=1.2)
    plt.legend(loc='lower left', fontsize='x-small', borderpad=None)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useOffset=True)
    plt.savefig(file_name_2, dpi = 201)  
    plt.show()
    
    #Parametros Matemáticos Cambio de eje Tiempo
    reduction_to= int(np.floor(len(abscwt) / 4)) #Intervalos de Ejes
    time_axis = np.linspace(0.0, emg_funcional.shape[0] * np.array(dt), emg_funcional.shape[0]+1)
    time = np.arange(0, time_axis.shape[0] + 1, int(reduction_to*2.5))
    output_time_axis = time_axis[time]
    outtime = np.round(output_time_axis, 0)
    
    #Parametros Matemáticos Cambio de eje Frecuencia
    scales = range(0,num-1)
    step = 200 
    outscales = scales[0:num:step]
    outfreqs = np.round(frequencies[0:num:step], 2)

    # Gráfica de CWT Normalizada
    plt.figure(figsize=(16,10))
    plt.text(int(bdt05[0]*1000), int(len(outscales)+1500), 'G1', color='w')
    plt.text(int(bdt05[2]*1000), int(len(outscales)+1500), 'G2', color='w')
    plt.text(int(bdt05[4]*1000), int(len(outscales)+1500), 'G3', color='w')
    plt.text(int(bdt05[6]*1000), int(len(outscales)+1500), 'G4', color='w')
    plt.text(int(bdt05[8]*1000), int(len(outscales)+1500), 'G5', color='w')
    plt.text(int(bdt05[10]*1000), int(len(outscales)+1500), 'G6', color='w')
    plt.text(int(bdt05[12]*1000), int(len(outscales)+1500), 'G7', color='w')
    plt.text(int(bdt05[14]*1000), int(len(outscales)+1500), 'G1', color='w')
    plt.text(int(bdt05[16]*1000), int(len(outscales)+1500), 'G2', color='w')
    plt.text(int(bdt05[18]*1000), int(len(outscales)+1500), 'G3', color='w')
    plt.text(int(bdt05[20]*1000), int(len(outscales)+1500), 'G4', color='w')
    plt.text(int(bdt05[22]*1000), int(len(outscales)+1500), 'G5', color='w')
    plt.text(int(bdt05[24]*1000), int(len(outscales)+1500), 'G6', color='w')
    plt.text(int(bdt05[26]*1000), int(len(outscales)+1500), 'G7', color='w')
    plt.imshow(normxgsts, cmap='jet', aspect ='auto')
    plt.xticks(time, outtime)
    plt.yticks(outscales,outfreqs)
    plt.ylim(top=1100)
    #plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.title(title_cwt)
    plt.savefig(file_name, dpi = 201)
    plt.show()
    
# 4.3. Electromiogragías Paciente 6
emgpacientes06 = [voltajesP06[7]] 
emg_funcional = np.array(emgpacientes06)
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
#Separación de gestos por el tiempo
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


#Saturación de valores Máximos
porcentajedevalores = 0.001
lista_G1 = gesto_1.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G2 = gesto_2.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G3 = gesto_3.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G4 = gesto_4.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G5 = gesto_5.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G6 = gesto_6.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G7 = gesto_7.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G8 = gesto_8.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G9 = gesto_9.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G10 = gesto_10.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G11 = gesto_11.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G12 = gesto_12.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G = [lista_G1, lista_G2, lista_G3, lista_G4, lista_G5, lista_G6,
       lista_G7, lista_G8, lista_G9, lista_G10, lista_G11, lista_G12]

nrodevalores_G1 = int(len(lista_G[0])*porcentajedevalores)
nrodevalores_G2 = int(len(lista_G[1])*porcentajedevalores)
nrodevalores_G3 = int(len(lista_G[2])*porcentajedevalores)
nrodevalores_G4 = int(len(lista_G[3])*porcentajedevalores)
nrodevalores_G5 = int(len(lista_G[4])*porcentajedevalores)
nrodevalores_G6 = int(len(lista_G[5])*porcentajedevalores)
nrodevalores_G7 = int(len(lista_G[6])*porcentajedevalores)
nrodevalores_G8 = int(len(lista_G[7])*porcentajedevalores)
nrodevalores_G9 = int(len(lista_G[8])*porcentajedevalores)
nrodevalores_G10 = int(len(lista_G[9])*porcentajedevalores)
nrodevalores_G11 = int(len(lista_G[10])*porcentajedevalores)
nrodevalores_G12 = int(len(lista_G[11])*porcentajedevalores)
nrodevalores_G = [nrodevalores_G1, nrodevalores_G2, nrodevalores_G3, nrodevalores_G4, nrodevalores_G5, nrodevalores_G6,
              nrodevalores_G7, nrodevalores_G8,nrodevalores_G9, nrodevalores_G10, nrodevalores_G11, nrodevalores_G12]

busquedadevalores_G1 = heapq.nlargest(nrodevalores_G[0], lista_G[0])
busquedadevalores_G2 = heapq.nlargest(nrodevalores_G[1], lista_G[1])
busquedadevalores_G3 = heapq.nlargest(nrodevalores_G[2], lista_G[2])
busquedadevalores_G4 = heapq.nlargest(nrodevalores_G[3], lista_G[3])
busquedadevalores_G5 = heapq.nlargest(nrodevalores_G[4], lista_G[4])
busquedadevalores_G6 = heapq.nlargest(nrodevalores_G[5], lista_G[5])
busquedadevalores_G7 = heapq.nlargest(nrodevalores_G[6], lista_G[6])
busquedadevalores_G8 = heapq.nlargest(nrodevalores_G[7], lista_G[7])
busquedadevalores_G9 = heapq.nlargest(nrodevalores_G[8], lista_G[8])
busquedadevalores_G10 = heapq.nlargest(nrodevalores_G[9], lista_G[9])
busquedadevalores_G11 = heapq.nlargest(nrodevalores_G[10], lista_G[10])
busquedadevalores_G12 = heapq.nlargest(nrodevalores_G[11], lista_G[11])
busquedadevalores_G = [busquedadevalores_G1, busquedadevalores_G2, busquedadevalores_G3, busquedadevalores_G4, busquedadevalores_G5, busquedadevalores_G6,
                   busquedadevalores_G7, busquedadevalores_G8, busquedadevalores_G9, busquedadevalores_G10, busquedadevalores_G11, busquedadevalores_G12]

lista_valoresmayores_G1 = np.array(busquedadevalores_G[0])
lista_valoresmayores_G2 = np.array(busquedadevalores_G[1])
lista_valoresmayores_G3 = np.array(busquedadevalores_G[2])
lista_valoresmayores_G4 = np.array(busquedadevalores_G[3])
lista_valoresmayores_G5 = np.array(busquedadevalores_G[4])
lista_valoresmayores_G6 = np.array(busquedadevalores_G[5])
lista_valoresmayores_G7 = np.array(busquedadevalores_G[6])
lista_valoresmayores_G8 = np.array(busquedadevalores_G[7])
lista_valoresmayores_G9 = np.array(busquedadevalores_G[8])
lista_valoresmayores_G10 = np.array(busquedadevalores_G[9])
lista_valoresmayores_G11 = np.array(busquedadevalores_G[10])
lista_valoresmayores_G12 = np.array(busquedadevalores_G[11])
lista_valoresmayores_G = [lista_valoresmayores_G1, lista_valoresmayores_G2, lista_valoresmayores_G3, lista_valoresmayores_G4, lista_valoresmayores_G5, lista_valoresmayores_G6,
                      lista_valoresmayores_G7, lista_valoresmayores_G8, lista_valoresmayores_G9, lista_valoresmayores_G10, lista_valoresmayores_G11, lista_valoresmayores_G12]

valormin_G1 = np.min(lista_valoresmayores_G[0])
valormin_G2 = np.min(lista_valoresmayores_G[1])
valormin_G3 = np.min(lista_valoresmayores_G[2])
valormin_G4 = np.min(lista_valoresmayores_G[3])
valormin_G5 = np.min(lista_valoresmayores_G[4])
valormin_G6 = np.min(lista_valoresmayores_G[5])
valormin_G7 = np.min(lista_valoresmayores_G[6])
valormin_G8 = np.min(lista_valoresmayores_G[7])
valormin_G9 = np.min(lista_valoresmayores_G[8])
valormin_G10 = np.min(lista_valoresmayores_G[9])
valormin_G11 = np.min(lista_valoresmayores_G[10])
valormin_G12 = np.min(lista_valoresmayores_G[11])
valormin_G = [valormin_G1, valormin_G2, valormin_G3, valormin_G4, valormin_G5, valormin_G6, 
          valormin_G7, valormin_G8, valormin_G9, valormin_G10, valormin_G11, valormin_G12]

gesto_1[gesto_1 >= valormin_G[0]] = valormin_G[0]
gesto_2[gesto_2 >= valormin_G[1]] = valormin_G[1]
gesto_3[gesto_3 >= valormin_G[2]] = valormin_G[2]
gesto_4[gesto_4 >= valormin_G[3]] = valormin_G[3]
gesto_5[gesto_5 >= valormin_G[4]] = valormin_G[4]
gesto_6[gesto_6 >= valormin_G[5]] = valormin_G[5]
gesto_7[gesto_7 >= valormin_G[6]] = valormin_G[6]
gesto_8[gesto_8 >= valormin_G[7]] = valormin_G[7]
gesto_9[gesto_9 >= valormin_G[8]] = valormin_G[8]
gesto_10[gesto_10 >= valormin_G[9]] = valormin_G[9]
gesto_11[gesto_11 >= valormin_G[10]] = valormin_G[10]
gesto_12[gesto_12 >= valormin_G[11]] = valormin_G[11]


normxgsts = mat2gray(abscwt)
normxgsts[:,0:1885], normxgsts[:,3665:5865], normxgsts[:,7600:11635], normxgsts[:,13372:17021], normxgsts[:,18831:24637], normxgsts[:,26768:30391], normxgsts[:,32345:34753], normxgsts[:,36512:38418], normxgsts[:,40172:43903], normxgsts[:,45711:49401], normxgsts[:,51243:55607], normxgsts[:,57366:61999], normxgsts[:,63679:65920] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
normxgsts[:,1886:3664] = mat2gray(gesto_1) 
normxgsts[:,5866:7599] = mat2gray(gesto_2)
normxgsts[:,11636:13371] = mat2gray(gesto_3)
normxgsts[:,17022:18830] = mat2gray(gesto_4)
normxgsts[:,24638:26767] = mat2gray(gesto_5)
normxgsts[:,30392:32344] = mat2gray(gesto_6)
normxgsts[:,34754:36511] = mat2gray(gesto_7)
normxgsts[:,38419:40171] = mat2gray(gesto_8)
normxgsts[:,43904:45710] = mat2gray(gesto_9)
normxgsts[:,49402:51242] = mat2gray(gesto_10)
normxgsts[:,55608:57365] = mat2gray(gesto_11)
normxgsts[:,62000:63678] = mat2gray(gesto_12)

fs = samplerate

#Nombre de cada Gráfica
pacientes = ['paciente 01 ','paciente 02 ','paciente 03 ','paciente 04 ','paciente 05 ','paciente 06 ','paciente 07 ','paciente 08 ','paciente 09 ','paciente 10 ']
canales = ['canal 1 ','canal 2 ','canal 3 ','canal 4 ','canal 5 ','canal 6 ','canal 7 ','canal 8 ']    
file_emg = 'gráfica EMG'
file_cwt = 'gráfica CWT cmor1.0-1.5'
file_end = 'gráfica CWT cmor1.0-1.5.png'
file_end_2 = 'gráfica EMG.png'

#Título de la Gr+afica
title_emg = canales[7] + pacientes[5] + file_emg
title_cwt = canales[7] + pacientes[5] + file_cwt

#Nombre del archivo PNG
file_name = canales[7] + pacientes[5] + file_end
file_name_2 = canales[7] + pacientes[5] + file_end_2

# Vectores de tiempo
timeemg = np.arange(0, len(emg_funcional) / fs, 1 / fs)
bdt06 = [1.886, 3.664, 5.866, 7.599, 11.636, 13.371, 17.022, 18.830, 24.638, 26.767, 30.392, 32.344, 
     34.754, 36.511, 38.419, 40.171, 43.904, 45.710, 49.402, 51.242, 55.608, 57.365, 62.000, 63.678]

# Gráficas EMG
plt.figure(figsize=(16,10))
plt.plot(timeemg, emg_funcional, 'b', label='Señal EMG RAW')
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.title(title_emg)
plt.xlim(0, timeemg.max())
plt.grid()
#plt.axhline(y=0.01,color='k')
plt.axvline(x=bdt06[0], color='c', linewidth=1.2, label="G1 - Mano en Reposo")
plt.axvline(x=bdt06[1], color='c', linewidth=1.2)
plt.axvline(x=bdt06[2], color='m', linewidth=1.2, label="G2 - Mano Cerrada en Puño")
plt.axvline(x=bdt06[3], color='m', linewidth=1.2)
plt.axvline(x=bdt06[4], color='y', linewidth=1.2, label="G3 - Flexión de la Muñeca")
plt.axvline(x=bdt06[5], color='y', linewidth=1.2)
plt.axvline(x=bdt06[6], color='r', linewidth=1.2, label="G4 - extensión de la Muñeca")
plt.axvline(x=bdt06[7], color='r', linewidth=1.2)
plt.axvline(x=bdt06[8], color='g', linewidth=1.2, label="G5 - Desviación Radial")
plt.axvline(x=bdt06[9], color='g', linewidth=1.2)
plt.axvline(x=bdt06[10], color='k', linewidth=1.2, label="G6 - Desviación Cubital")
plt.axvline(x=bdt06[11], color='k', linewidth=1.2)
plt.axvline(x=bdt06[12], color='c', linewidth=1.2)
plt.axvline(x=bdt06[13], color='c', linewidth=1.2)
plt.axvline(x=bdt06[14], color='m', linewidth=1.2)
plt.axvline(x=bdt06[15], color='m', linewidth=1.2)
plt.axvline(x=bdt06[16], color='y', linewidth=1.2)
plt.axvline(x=bdt06[17], color='y', linewidth=1.2)
plt.axvline(x=bdt06[18], color='r', linewidth=1.2)
plt.axvline(x=bdt06[19], color='r', linewidth=1.2)
plt.axvline(x=bdt06[20], color='g', linewidth=1.2)
plt.axvline(x=bdt06[21], color='g', linewidth=1.2)
plt.axvline(x=bdt06[22], color='k', linewidth=1.2)
plt.axvline(x=bdt06[23], color='k', linewidth=1.2)
plt.legend(loc='lower left', fontsize='x-small', borderpad=None)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useOffset=True)
plt.savefig(file_name_2, dpi = 201)  
plt.show()

#Parametros Matemáticos Cambio de eje Tiempo
reduction_to= int(np.floor(len(abscwt) / 4)) #Intervalos de Ejes
time_axis = np.linspace(0.0, emg_funcional.shape[0] * np.array(dt), emg_funcional.shape[0]+1)
time = np.arange(0, time_axis.shape[0] + 1, int(reduction_to*2.5))
output_time_axis = time_axis[time]
outtime = np.round(output_time_axis, 0)

#Parametros Matemáticos Cambio de eje Frecuencia
scales = range(0,num-1)
step = 200 
outscales = scales[0:num:step]
outfreqs = np.round(frequencies[0:num:step], 2)

# Gráfica de CWT Normalizada
plt.figure(figsize=(16,10))
plt.text(int(bdt06[0]*1000), int(len(outscales)+1500), 'G1', color='w')
plt.text(int(bdt06[2]*1000), int(len(outscales)+1500), 'G2', color='w')
plt.text(int(bdt06[4]*1000), int(len(outscales)+1500), 'G3', color='w')
plt.text(int(bdt06[6]*1000), int(len(outscales)+1500), 'G4', color='w')
plt.text(int(bdt06[8]*1000), int(len(outscales)+1500), 'G5', color='w')
plt.text(int(bdt06[10]*1000), int(len(outscales)+1500), 'G6', color='w')
plt.text(int(bdt06[12]*1000), int(len(outscales)+1500), 'G1', color='w')
plt.text(int(bdt06[14]*1000), int(len(outscales)+1500), 'G2', color='w')
plt.text(int(bdt06[16]*1000), int(len(outscales)+1500), 'G3', color='w')
plt.text(int(bdt06[18]*1000), int(len(outscales)+1500), 'G4', color='w')
plt.text(int(bdt06[20]*1000), int(len(outscales)+1500), 'G5', color='w')
plt.text(int(bdt06[22]*1000), int(len(outscales)+1500), 'G6', color='w')
plt.imshow(normxgsts, cmap='jet', aspect ='auto')
plt.xticks(time, outtime)
plt.yticks(outscales,outfreqs)
plt.ylim(top=1100)
#plt.gca().invert_yaxis()
plt.colorbar()
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.title(title_cwt)
plt.savefig(file_name, dpi = 201)
plt.show()

# 4.4. Electromiogragías Paciente 7
emgpacientes07 = [voltajesP07[4]]
emg_funcional = np.array(emgpacientes07)
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
#Separación de gestos por el tiempo
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

#Saturación de valores Máximos
porcentajedevalores = 0.001
lista_G1 = gesto_1.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G2 = gesto_2.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G3 = gesto_3.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G4 = gesto_4.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G5 = gesto_5.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G6 = gesto_6.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G7 = gesto_7.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G8 = gesto_8.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G9 = gesto_9.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G10 = gesto_10.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G11 = gesto_11.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G12 = gesto_12.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G = [lista_G1, lista_G2, lista_G3, lista_G4, lista_G5, lista_G6,
           lista_G7, lista_G8, lista_G9, lista_G10, lista_G11, lista_G12]

nrodevalores_G1 = int(len(lista_G[0])*porcentajedevalores)
nrodevalores_G2 = int(len(lista_G[1])*porcentajedevalores)
nrodevalores_G3 = int(len(lista_G[2])*porcentajedevalores)
nrodevalores_G4 = int(len(lista_G[3])*porcentajedevalores)
nrodevalores_G5 = int(len(lista_G[4])*porcentajedevalores)
nrodevalores_G6 = int(len(lista_G[5])*porcentajedevalores)
nrodevalores_G7 = int(len(lista_G[6])*porcentajedevalores)
nrodevalores_G8 = int(len(lista_G[7])*porcentajedevalores)
nrodevalores_G9 = int(len(lista_G[8])*porcentajedevalores)
nrodevalores_G10 = int(len(lista_G[9])*porcentajedevalores)
nrodevalores_G11 = int(len(lista_G[10])*porcentajedevalores)
nrodevalores_G12 = int(len(lista_G[11])*porcentajedevalores)
nrodevalores_G = [nrodevalores_G1, nrodevalores_G2, nrodevalores_G3, nrodevalores_G4, nrodevalores_G5, nrodevalores_G6,
                  nrodevalores_G7, nrodevalores_G8,nrodevalores_G9, nrodevalores_G10, nrodevalores_G11, nrodevalores_G12]

busquedadevalores_G1 = heapq.nlargest(nrodevalores_G[0], lista_G[0])
busquedadevalores_G2 = heapq.nlargest(nrodevalores_G[1], lista_G[1])
busquedadevalores_G3 = heapq.nlargest(nrodevalores_G[2], lista_G[2])
busquedadevalores_G4 = heapq.nlargest(nrodevalores_G[3], lista_G[3])
busquedadevalores_G5 = heapq.nlargest(nrodevalores_G[4], lista_G[4])
busquedadevalores_G6 = heapq.nlargest(nrodevalores_G[5], lista_G[5])
busquedadevalores_G7 = heapq.nlargest(nrodevalores_G[6], lista_G[6])
busquedadevalores_G8 = heapq.nlargest(nrodevalores_G[7], lista_G[7])
busquedadevalores_G9 = heapq.nlargest(nrodevalores_G[8], lista_G[8])
busquedadevalores_G10 = heapq.nlargest(nrodevalores_G[9], lista_G[9])
busquedadevalores_G11 = heapq.nlargest(nrodevalores_G[10], lista_G[10])
busquedadevalores_G12 = heapq.nlargest(nrodevalores_G[11], lista_G[11])
busquedadevalores_G = [busquedadevalores_G1, busquedadevalores_G2, busquedadevalores_G3, busquedadevalores_G4, busquedadevalores_G5, busquedadevalores_G6,
                       busquedadevalores_G7, busquedadevalores_G8, busquedadevalores_G9, busquedadevalores_G10, busquedadevalores_G11, busquedadevalores_G12]

lista_valoresmayores_G1 = np.array(busquedadevalores_G[0])
lista_valoresmayores_G2 = np.array(busquedadevalores_G[1])
lista_valoresmayores_G3 = np.array(busquedadevalores_G[2])
lista_valoresmayores_G4 = np.array(busquedadevalores_G[3])
lista_valoresmayores_G5 = np.array(busquedadevalores_G[4])
lista_valoresmayores_G6 = np.array(busquedadevalores_G[5])
lista_valoresmayores_G7 = np.array(busquedadevalores_G[6])
lista_valoresmayores_G8 = np.array(busquedadevalores_G[7])
lista_valoresmayores_G9 = np.array(busquedadevalores_G[8])
lista_valoresmayores_G10 = np.array(busquedadevalores_G[9])
lista_valoresmayores_G11 = np.array(busquedadevalores_G[10])
lista_valoresmayores_G12 = np.array(busquedadevalores_G[11])
lista_valoresmayores_G = [lista_valoresmayores_G1, lista_valoresmayores_G2, lista_valoresmayores_G3, lista_valoresmayores_G4, lista_valoresmayores_G5, lista_valoresmayores_G6,
                          lista_valoresmayores_G7, lista_valoresmayores_G8, lista_valoresmayores_G9, lista_valoresmayores_G10, lista_valoresmayores_G11, lista_valoresmayores_G12]

valormin_G1 = np.min(lista_valoresmayores_G[0])
valormin_G2 = np.min(lista_valoresmayores_G[1])
valormin_G3 = np.min(lista_valoresmayores_G[2])
valormin_G4 = np.min(lista_valoresmayores_G[3])
valormin_G5 = np.min(lista_valoresmayores_G[4])
valormin_G6 = np.min(lista_valoresmayores_G[5])
valormin_G7 = np.min(lista_valoresmayores_G[6])
valormin_G8 = np.min(lista_valoresmayores_G[7])
valormin_G9 = np.min(lista_valoresmayores_G[8])
valormin_G10 = np.min(lista_valoresmayores_G[9])
valormin_G11 = np.min(lista_valoresmayores_G[10])
valormin_G12 = np.min(lista_valoresmayores_G[11])
valormin_G = [valormin_G1, valormin_G2, valormin_G3, valormin_G4, valormin_G5, valormin_G6, 
              valormin_G7, valormin_G8, valormin_G9, valormin_G10, valormin_G11, valormin_G12]

gesto_1[gesto_1 >= valormin_G[0]] = valormin_G[0]
gesto_2[gesto_2 >= valormin_G[1]] = valormin_G[1]
gesto_3[gesto_3 >= valormin_G[2]] = valormin_G[2]
gesto_4[gesto_4 >= valormin_G[3]] = valormin_G[3]
gesto_5[gesto_5 >= valormin_G[4]] = valormin_G[4]
gesto_6[gesto_6 >= valormin_G[5]] = valormin_G[5]
gesto_7[gesto_7 >= valormin_G[6]] = valormin_G[6]
gesto_8[gesto_8 >= valormin_G[7]] = valormin_G[7]
gesto_9[gesto_9 >= valormin_G[8]] = valormin_G[8]
gesto_10[gesto_10 >= valormin_G[9]] = valormin_G[9]
gesto_11[gesto_11 >= valormin_G[10]] = valormin_G[10]
gesto_12[gesto_12 >= valormin_G[11]] = valormin_G[11]

normxgsts = mat2gray(abscwt)
normxgsts[:,3883:6166], normxgsts[:,8162:12655], normxgsts[:,14943:20752], normxgsts[:,23405:27657], normxgsts[:,30644:34211], normxgsts[:,36657:39280], normxgsts[:,41151:43230], normxgsts[:,45592:49301], normxgsts[:,52048:57023], normxgsts[:,59580:63843], normxgsts[:,66522:70997], normxgsts[:,73703:75676] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
normxgsts[:,0:3882] = mat2gray(gesto_1) 
normxgsts[:,6167:8161] = mat2gray(gesto_2)
normxgsts[:,12656:14942] = mat2gray(gesto_3)
normxgsts[:,20753:23405] = mat2gray(gesto_4)
normxgsts[:,27657:30644] = mat2gray(gesto_5)
normxgsts[:,34211:36657] = mat2gray(gesto_6)
normxgsts[:,39280:41151] = mat2gray(gesto_7)
normxgsts[:,43230:45592] = mat2gray(gesto_8)
normxgsts[:,49301:52048] = mat2gray(gesto_9)
normxgsts[:,57023:59580] = mat2gray(gesto_10)
normxgsts[:,63843:66522] = mat2gray(gesto_11)
normxgsts[:,70997:73703] = mat2gray(gesto_12)

fs = samplerate

#Nombre de cada Gráfica
pacientes = ['paciente 01 ','paciente 02 ','paciente 03 ','paciente 04 ','paciente 05 ','paciente 06 ','paciente 07 ','paciente 08 ','paciente 09 ','paciente 10 ']
canales = ['canal 1 ','canal 2 ','canal 3 ','canal 4 ','canal 5 ','canal 6 ','canal 7 ','canal 8 ']    
file_emg = 'gráfica EMG'
file_cwt = 'gráfica CWT cmor1.0-1.5'
file_end = 'gráfica CWT cmor1.0-1.5.png'
file_end_2 = 'gráfica EMG.png'

#Título de la Gr+afica
title_emg = canales[4] + pacientes[6] + file_emg
title_cwt = canales[4] + pacientes[6] + file_cwt

#Nombre del archivo PNG
file_name = canales[4] + pacientes[6] + file_end
file_name_2 = canales[4] + pacientes[6] + file_end_2

# Vectores de tiempo
timeemg = np.arange(0, len(emg_funcional) / fs, 1 / fs)
bdt07 = [0.01, 3.882, 6.167, 8.161, 12.656, 14.942, 20.753, 23.405, 27.657, 30.644, 34.211, 36.657, 
         39.280, 41.151, 43.230, 45.592, 49.301, 52.048, 57.023, 59.580, 63.843, 66.522, 70.997, 73.703]

# Gráficas EMG
plt.figure(figsize=(16,10))
plt.plot(timeemg, emg_funcional, 'b', label='Señal EMG RAW')
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.title(title_emg)
plt.xlim(0, timeemg.max())
plt.grid()
#plt.axhline(y=0.01,color='k')
plt.axvline(x=bdt07[0], color='c', linewidth=1.2, label="G1 - Mano en Reposo")
plt.axvline(x=bdt07[1], color='c', linewidth=1.2)
plt.axvline(x=bdt07[2], color='m', linewidth=1.2, label="G2 - Mano Cerrada en Puño")
plt.axvline(x=bdt07[3], color='m', linewidth=1.2)
plt.axvline(x=bdt07[4], color='y', linewidth=1.2, label="G3 - Flexión de la Muñeca")
plt.axvline(x=bdt07[5], color='y', linewidth=1.2)
plt.axvline(x=bdt07[6], color='r', linewidth=1.2, label="G4 - extensión de la Muñeca")
plt.axvline(x=bdt07[7], color='r', linewidth=1.2)
plt.axvline(x=bdt07[8], color='g', linewidth=1.2, label="G5 - Desviación Radial")
plt.axvline(x=bdt07[9], color='g', linewidth=1.2)
plt.axvline(x=bdt07[10], color='k', linewidth=1.2, label="G6 - Desviación Cubital")
plt.axvline(x=bdt07[11], color='k', linewidth=1.2)
plt.axvline(x=bdt07[12], color='c', linewidth=1.2)
plt.axvline(x=bdt07[13], color='c', linewidth=1.2)
plt.axvline(x=bdt07[14], color='m', linewidth=1.2)
plt.axvline(x=bdt07[15], color='m', linewidth=1.2)
plt.axvline(x=bdt07[16], color='y', linewidth=1.2)
plt.axvline(x=bdt07[17], color='y', linewidth=1.2)
plt.axvline(x=bdt07[18], color='r', linewidth=1.2)
plt.axvline(x=bdt07[19], color='r', linewidth=1.2)
plt.axvline(x=bdt07[20], color='g', linewidth=1.2)
plt.axvline(x=bdt07[21], color='g', linewidth=1.2)
plt.axvline(x=bdt07[22], color='k', linewidth=1.2)
plt.axvline(x=bdt07[23], color='k', linewidth=1.2)
plt.legend(loc='lower left', fontsize='x-small', borderpad=None)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useOffset=True)
plt.savefig(file_name_2, dpi = 201)  
plt.show()

#Parametros Matemáticos Cambio de eje Tiempo
reduction_to= int(np.floor(len(abscwt) / 4)) #Intervalos de Ejes
time_axis = np.linspace(0.0, emg_funcional.shape[0] * np.array(dt), emg_funcional.shape[0]+1)
time = np.arange(0, time_axis.shape[0] + 1, int(reduction_to*2.5))
output_time_axis = time_axis[time]
outtime = np.round(output_time_axis, 0)

#Parametros Matemáticos Cambio de eje Frecuencia
scales = range(0,num-1)
step = 200 
outscales = scales[0:num:step]
outfreqs = np.round(frequencies[0:num:step], 2)

# Gráfica de CWT Normalizada
plt.figure(figsize=(16,10))
plt.text(int(bdt07[0]*1100), int(len(outscales)+1500), 'G1', color='w')
plt.text(int(bdt07[2]*1000), int(len(outscales)+1500), 'G2', color='w')
plt.text(int(bdt07[4]*1000), int(len(outscales)+1500), 'G3', color='w')
plt.text(int(bdt07[6]*1000), int(len(outscales)+1500), 'G4', color='w')
plt.text(int(bdt07[8]*1000), int(len(outscales)+1500), 'G5', color='w')
plt.text(int(bdt07[10]*1000), int(len(outscales)+1500), 'G6', color='w')
plt.text(int(bdt07[12]*1000), int(len(outscales)+1500), 'G1', color='w')
plt.text(int(bdt07[14]*1000), int(len(outscales)+1500), 'G2', color='w')
plt.text(int(bdt07[16]*1000), int(len(outscales)+1500), 'G3', color='w')
plt.text(int(bdt07[18]*1000), int(len(outscales)+1500), 'G4', color='w')
plt.text(int(bdt07[20]*1000), int(len(outscales)+1500), 'G5', color='w')
plt.text(int(bdt07[22]*1000), int(len(outscales)+1500), 'G6', color='w')
plt.imshow(normxgsts, cmap='jet', aspect ='auto')
plt.xticks(time, outtime)
plt.yticks(outscales,outfreqs)
plt.ylim(top=1100)
#plt.gca().invert_yaxis()
plt.colorbar()
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.title(title_cwt)
plt.savefig(file_name, dpi = 201)
plt.show()

# 4.6. Electromiogragías Paciente 8
emgpacientes08 = [voltajesP08[3]]        
emg_funcional = np.array(emgpacientes08)
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
#Separación de gestos por el tiempo
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

#Saturación de valores Máximos
porcentajedevalores = 0.001
lista_G1 = gesto_1.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G2 = gesto_2.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G3 = gesto_3.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G4 = gesto_4.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G5 = gesto_5.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G6 = gesto_6.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G7 = gesto_7.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G8 = gesto_8.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G9 = gesto_9.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G10 = gesto_10.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G11 = gesto_11.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G12 = gesto_12.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G = [lista_G1, lista_G2, lista_G3, lista_G4, lista_G5, lista_G6,
           lista_G7, lista_G8, lista_G9, lista_G10, lista_G11, lista_G12]

nrodevalores_G1 = int(len(lista_G[0])*porcentajedevalores)
nrodevalores_G2 = int(len(lista_G[1])*porcentajedevalores)
nrodevalores_G3 = int(len(lista_G[2])*porcentajedevalores)
nrodevalores_G4 = int(len(lista_G[3])*porcentajedevalores)
nrodevalores_G5 = int(len(lista_G[4])*porcentajedevalores)
nrodevalores_G6 = int(len(lista_G[5])*porcentajedevalores)
nrodevalores_G7 = int(len(lista_G[6])*porcentajedevalores)
nrodevalores_G8 = int(len(lista_G[7])*porcentajedevalores)
nrodevalores_G9 = int(len(lista_G[8])*porcentajedevalores)
nrodevalores_G10 = int(len(lista_G[9])*porcentajedevalores)
nrodevalores_G11 = int(len(lista_G[10])*porcentajedevalores)
nrodevalores_G12 = int(len(lista_G[11])*porcentajedevalores)
nrodevalores_G = [nrodevalores_G1, nrodevalores_G2, nrodevalores_G3, nrodevalores_G4, nrodevalores_G5, nrodevalores_G6,
                  nrodevalores_G7, nrodevalores_G8,nrodevalores_G9, nrodevalores_G10, nrodevalores_G11, nrodevalores_G12]

busquedadevalores_G1 = heapq.nlargest(nrodevalores_G[0], lista_G[0])
busquedadevalores_G2 = heapq.nlargest(nrodevalores_G[1], lista_G[1])
busquedadevalores_G3 = heapq.nlargest(nrodevalores_G[2], lista_G[2])
busquedadevalores_G4 = heapq.nlargest(nrodevalores_G[3], lista_G[3])
busquedadevalores_G5 = heapq.nlargest(nrodevalores_G[4], lista_G[4])
busquedadevalores_G6 = heapq.nlargest(nrodevalores_G[5], lista_G[5])
busquedadevalores_G7 = heapq.nlargest(nrodevalores_G[6], lista_G[6])
busquedadevalores_G8 = heapq.nlargest(nrodevalores_G[7], lista_G[7])
busquedadevalores_G9 = heapq.nlargest(nrodevalores_G[8], lista_G[8])
busquedadevalores_G10 = heapq.nlargest(nrodevalores_G[9], lista_G[9])
busquedadevalores_G11 = heapq.nlargest(nrodevalores_G[10], lista_G[10])
busquedadevalores_G12 = heapq.nlargest(nrodevalores_G[11], lista_G[11])
busquedadevalores_G = [busquedadevalores_G1, busquedadevalores_G2, busquedadevalores_G3, busquedadevalores_G4, busquedadevalores_G5, busquedadevalores_G6,
                       busquedadevalores_G7, busquedadevalores_G8, busquedadevalores_G9, busquedadevalores_G10, busquedadevalores_G11, busquedadevalores_G12]

lista_valoresmayores_G1 = np.array(busquedadevalores_G[0])
lista_valoresmayores_G2 = np.array(busquedadevalores_G[1])
lista_valoresmayores_G3 = np.array(busquedadevalores_G[2])
lista_valoresmayores_G4 = np.array(busquedadevalores_G[3])
lista_valoresmayores_G5 = np.array(busquedadevalores_G[4])
lista_valoresmayores_G6 = np.array(busquedadevalores_G[5])
lista_valoresmayores_G7 = np.array(busquedadevalores_G[6])
lista_valoresmayores_G8 = np.array(busquedadevalores_G[7])
lista_valoresmayores_G9 = np.array(busquedadevalores_G[8])
lista_valoresmayores_G10 = np.array(busquedadevalores_G[9])
lista_valoresmayores_G11 = np.array(busquedadevalores_G[10])
lista_valoresmayores_G12 = np.array(busquedadevalores_G[11])
lista_valoresmayores_G = [lista_valoresmayores_G1, lista_valoresmayores_G2, lista_valoresmayores_G3, lista_valoresmayores_G4, lista_valoresmayores_G5, lista_valoresmayores_G6,
                          lista_valoresmayores_G7, lista_valoresmayores_G8, lista_valoresmayores_G9, lista_valoresmayores_G10, lista_valoresmayores_G11, lista_valoresmayores_G12]

valormin_G1 = np.min(lista_valoresmayores_G[0])
valormin_G2 = np.min(lista_valoresmayores_G[1])
valormin_G3 = np.min(lista_valoresmayores_G[2])
valormin_G4 = np.min(lista_valoresmayores_G[3])
valormin_G5 = np.min(lista_valoresmayores_G[4])
valormin_G6 = np.min(lista_valoresmayores_G[5])
valormin_G7 = np.min(lista_valoresmayores_G[6])
valormin_G8 = np.min(lista_valoresmayores_G[7])
valormin_G9 = np.min(lista_valoresmayores_G[8])
valormin_G10 = np.min(lista_valoresmayores_G[9])
valormin_G11 = np.min(lista_valoresmayores_G[10])
valormin_G12 = np.min(lista_valoresmayores_G[11])
valormin_G = [valormin_G1, valormin_G2, valormin_G3, valormin_G4, valormin_G5, valormin_G6, 
              valormin_G7, valormin_G8, valormin_G9, valormin_G10, valormin_G11, valormin_G12]

gesto_1[gesto_1 >= valormin_G[0]] = valormin_G[0]
gesto_2[gesto_2 >= valormin_G[1]] = valormin_G[1]
gesto_3[gesto_3 >= valormin_G[2]] = valormin_G[2]
gesto_4[gesto_4 >= valormin_G[3]] = valormin_G[3]
gesto_5[gesto_5 >= valormin_G[4]] = valormin_G[4]
gesto_6[gesto_6 >= valormin_G[5]] = valormin_G[5]
gesto_7[gesto_7 >= valormin_G[6]] = valormin_G[6]
gesto_8[gesto_8 >= valormin_G[7]] = valormin_G[7]
gesto_9[gesto_9 >= valormin_G[8]] = valormin_G[8]
gesto_10[gesto_10 >= valormin_G[9]] = valormin_G[9]
gesto_11[gesto_11 >= valormin_G[10]] = valormin_G[10]
gesto_12[gesto_12 >= valormin_G[11]] = valormin_G[11]

normxgsts = mat2gray(abscwt)
normxgsts[:,0:1352], normxgsts[:,3232:5096], normxgsts[:,6667:11252], normxgsts[:,13065:16258], normxgsts[:,17683:20702], normxgsts[:,22187:25256], normxgsts[:,26875:29945], normxgsts[:,31248:32784], normxgsts[:,34314:37773], normxgsts[:,39099:42420], normxgsts[:,44086:46488], normxgsts[:,48175:50528], normxgsts[:,52032:52820] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
normxgsts[:,1353:3231] = mat2gray(gesto_1) 
normxgsts[:,5097:6666] = mat2gray(gesto_2)
normxgsts[:,11253:13064] = mat2gray(gesto_3)
normxgsts[:,16259:17682] = mat2gray(gesto_4)
normxgsts[:,20703:22186] = mat2gray(gesto_5)
normxgsts[:,25257:26874] = mat2gray(gesto_6)
normxgsts[:,29946:31247] = mat2gray(gesto_7)
normxgsts[:,32785:34313] = mat2gray(gesto_8)
normxgsts[:,37774:39098] = mat2gray(gesto_9)
normxgsts[:,42421:44085] = mat2gray(gesto_10)
normxgsts[:,46489:48174] = mat2gray(gesto_11)
normxgsts[:,50529:52031] = mat2gray(gesto_12)

fs = samplerate

#Nombre de cada Gráfica
pacientes = ['paciente 01 ','paciente 02 ','paciente 03 ','paciente 04 ','paciente 05 ','paciente 06 ','paciente 07 ','paciente 08 ','paciente 09 ','paciente 10 ']
canales = ['canal 1 ','canal 2 ','canal 3 ','canal 4 ','canal 5 ','canal 6 ','canal 7 ','canal 8 ']    
file_emg = 'gráfica EMG'
file_cwt = 'gráfica CWT cmor1.0-1.5'
file_end = 'gráfica CWT cmor1.0-1.5.png'
file_end_2 = 'gráfica EMG.png'

#Título de la Gr+afica
title_emg = canales[3] + pacientes[7] + file_emg
title_cwt = canales[3] + pacientes[7] + file_cwt

#Nombre del archivo PNG
file_name = canales[3] + pacientes[7] + file_end
file_name_2 = canales[3] + pacientes[7] + file_end_2

# Vectores de tiempo
timeemg = np.arange(0, len(emg_funcional) / fs, 1 / fs)
bdt08 = [1.353, 3.231, 5.097, 6.666, 11.253, 13.064, 16.259, 17.682, 20.703, 22.186, 25.257, 26.874, 
         29.946, 31.247, 32.785, 34.313, 37.774, 39.098, 42.421, 44.085, 46.489, 48.174, 50.529, 52.031]

# Gráficas EMG
plt.figure(figsize=(16,10))
plt.plot(timeemg, emg_funcional, 'b', label='Señal EMG RAW')
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.title(title_emg)
plt.xlim(0, timeemg.max())
plt.grid()
#plt.axhline(y=0.01,color='k')
plt.axvline(x=bdt08[0], color='c', linewidth=1.2, label="G1 - Mano en Reposo")
plt.axvline(x=bdt08[1], color='c', linewidth=1.2)
plt.axvline(x=bdt08[2], color='m', linewidth=1.2, label="G2 - Mano Cerrada en Puño")
plt.axvline(x=bdt08[3], color='m', linewidth=1.2)
plt.axvline(x=bdt08[4], color='y', linewidth=1.2, label="G3 - Flexión de la Muñeca")
plt.axvline(x=bdt08[5], color='y', linewidth=1.2)
plt.axvline(x=bdt08[6], color='r', linewidth=1.2, label="G4 - extensión de la Muñeca")
plt.axvline(x=bdt08[7], color='r', linewidth=1.2)
plt.axvline(x=bdt08[8], color='g', linewidth=1.2, label="G5 - Desviación Radial")
plt.axvline(x=bdt08[9], color='g', linewidth=1.2)
plt.axvline(x=bdt08[10], color='k', linewidth=1.2, label="G6 - Desviación Cubital")
plt.axvline(x=bdt08[11], color='k', linewidth=1.2)
plt.axvline(x=bdt08[12], color='c', linewidth=1.2)
plt.axvline(x=bdt08[13], color='c', linewidth=1.2)
plt.axvline(x=bdt08[14], color='m', linewidth=1.2)
plt.axvline(x=bdt08[15], color='m', linewidth=1.2)
plt.axvline(x=bdt08[16], color='y', linewidth=1.2)
plt.axvline(x=bdt08[17], color='y', linewidth=1.2)
plt.axvline(x=bdt08[18], color='r', linewidth=1.2)
plt.axvline(x=bdt08[19], color='r', linewidth=1.2)
plt.axvline(x=bdt08[20], color='g', linewidth=1.2)
plt.axvline(x=bdt08[21], color='g', linewidth=1.2)
plt.axvline(x=bdt08[22], color='k', linewidth=1.2)
plt.axvline(x=bdt08[23], color='k', linewidth=1.2)
plt.legend(loc='lower left', fontsize='x-small', borderpad=None)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useOffset=True)
plt.savefig(file_name_2, dpi = 201)  
plt.show()

#Parametros Matemáticos Cambio de eje Tiempo
reduction_to= int(np.floor(len(abscwt) / 4)) #Intervalos de Ejes
time_axis = np.linspace(0.0, emg_funcional.shape[0] * np.array(dt), emg_funcional.shape[0]+1)
time = np.arange(0, time_axis.shape[0] + 1, int(reduction_to*2.5))
output_time_axis = time_axis[time]
outtime = np.round(output_time_axis, 0)

#Parametros Matemáticos Cambio de eje Frecuencia
scales = range(0,num-1)
step = 200 
outscales = scales[0:num:step]
outfreqs = np.round(frequencies[0:num:step], 2)

# Gráfica de CWT Normalizada
plt.figure(figsize=(16,10))
plt.text(int(bdt08[0]*1000), int(len(outscales)+1500), 'G1', color='w')
plt.text(int(bdt08[2]*1000), int(len(outscales)+1500), 'G2', color='w')
plt.text(int(bdt08[4]*1000), int(len(outscales)+1500), 'G3', color='w')
plt.text(int(bdt08[6]*1000), int(len(outscales)+1500), 'G4', color='w')
plt.text(int(bdt08[8]*1000), int(len(outscales)+1500), 'G5', color='w')
plt.text(int(bdt08[10]*1000), int(len(outscales)+1500), 'G6', color='w')
plt.text(int(bdt08[12]*1000), int(len(outscales)+1500), 'G1', color='w')
plt.text(int(bdt08[14]*1000), int(len(outscales)+1500), 'G2', color='w')
plt.text(int(bdt08[16]*1000), int(len(outscales)+1500), 'G3', color='w')
plt.text(int(bdt08[18]*1000), int(len(outscales)+1500), 'G4', color='w')
plt.text(int(bdt08[20]*1000), int(len(outscales)+1500), 'G5', color='w')
plt.text(int(bdt08[22]*1000), int(len(outscales)+1500), 'G6', color='w')
plt.imshow(normxgsts, cmap='jet', aspect ='auto')
plt.xticks(time, outtime)
plt.yticks(outscales,outfreqs)
plt.ylim(top=1100)
#plt.gca().invert_yaxis()
plt.colorbar()
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.title(title_cwt)
plt.savefig(file_name, dpi = 201)
plt.show()
        
# 4.6. Electromiogragías Paciente 9
emgpacientes09 = [voltajesP09[1]] 
emg_funcional = np.array(emgpacientes09)
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
#Separación de gestos por el tiempo
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

#Saturación de valores Máximos
porcentajedevalores = 0.001
lista_G1 = gesto_1.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G2 = gesto_2.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G3 = gesto_3.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G4 = gesto_4.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G5 = gesto_5.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G6 = gesto_6.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G7 = gesto_7.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G8 = gesto_8.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G9 = gesto_9.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G10 = gesto_10.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G11 = gesto_11.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G12 = gesto_12.flatten(order='F') # transformar matrices de NumPy en vectores por columna
lista_G = [lista_G1, lista_G2, lista_G3, lista_G4, lista_G5, lista_G6,
           lista_G7, lista_G8, lista_G9, lista_G10, lista_G11, lista_G12]

nrodevalores_G1 = int(len(lista_G[0])*porcentajedevalores)
nrodevalores_G2 = int(len(lista_G[1])*porcentajedevalores)
nrodevalores_G3 = int(len(lista_G[2])*porcentajedevalores)
nrodevalores_G4 = int(len(lista_G[3])*porcentajedevalores)
nrodevalores_G5 = int(len(lista_G[4])*porcentajedevalores)
nrodevalores_G6 = int(len(lista_G[5])*porcentajedevalores)
nrodevalores_G7 = int(len(lista_G[6])*porcentajedevalores)
nrodevalores_G8 = int(len(lista_G[7])*porcentajedevalores)
nrodevalores_G9 = int(len(lista_G[8])*porcentajedevalores)
nrodevalores_G10 = int(len(lista_G[9])*porcentajedevalores)
nrodevalores_G11 = int(len(lista_G[10])*porcentajedevalores)
nrodevalores_G12 = int(len(lista_G[11])*porcentajedevalores)
nrodevalores_G = [nrodevalores_G1, nrodevalores_G2, nrodevalores_G3, nrodevalores_G4, nrodevalores_G5, nrodevalores_G6,
                  nrodevalores_G7, nrodevalores_G8,nrodevalores_G9, nrodevalores_G10, nrodevalores_G11, nrodevalores_G12]

busquedadevalores_G1 = heapq.nlargest(nrodevalores_G[0], lista_G[0])
busquedadevalores_G2 = heapq.nlargest(nrodevalores_G[1], lista_G[1])
busquedadevalores_G3 = heapq.nlargest(nrodevalores_G[2], lista_G[2])
busquedadevalores_G4 = heapq.nlargest(nrodevalores_G[3], lista_G[3])
busquedadevalores_G5 = heapq.nlargest(nrodevalores_G[4], lista_G[4])
busquedadevalores_G6 = heapq.nlargest(nrodevalores_G[5], lista_G[5])
busquedadevalores_G7 = heapq.nlargest(nrodevalores_G[6], lista_G[6])
busquedadevalores_G8 = heapq.nlargest(nrodevalores_G[7], lista_G[7])
busquedadevalores_G9 = heapq.nlargest(nrodevalores_G[8], lista_G[8])
busquedadevalores_G10 = heapq.nlargest(nrodevalores_G[9], lista_G[9])
busquedadevalores_G11 = heapq.nlargest(nrodevalores_G[10], lista_G[10])
busquedadevalores_G12 = heapq.nlargest(nrodevalores_G[11], lista_G[11])
busquedadevalores_G = [busquedadevalores_G1, busquedadevalores_G2, busquedadevalores_G3, busquedadevalores_G4, busquedadevalores_G5, busquedadevalores_G6,
                       busquedadevalores_G7, busquedadevalores_G8, busquedadevalores_G9, busquedadevalores_G10, busquedadevalores_G11, busquedadevalores_G12]

lista_valoresmayores_G1 = np.array(busquedadevalores_G[0])
lista_valoresmayores_G2 = np.array(busquedadevalores_G[1])
lista_valoresmayores_G3 = np.array(busquedadevalores_G[2])
lista_valoresmayores_G4 = np.array(busquedadevalores_G[3])
lista_valoresmayores_G5 = np.array(busquedadevalores_G[4])
lista_valoresmayores_G6 = np.array(busquedadevalores_G[5])
lista_valoresmayores_G7 = np.array(busquedadevalores_G[6])
lista_valoresmayores_G8 = np.array(busquedadevalores_G[7])
lista_valoresmayores_G9 = np.array(busquedadevalores_G[8])
lista_valoresmayores_G10 = np.array(busquedadevalores_G[9])
lista_valoresmayores_G11 = np.array(busquedadevalores_G[10])
lista_valoresmayores_G12 = np.array(busquedadevalores_G[11])
lista_valoresmayores_G = [lista_valoresmayores_G1, lista_valoresmayores_G2, lista_valoresmayores_G3, lista_valoresmayores_G4, lista_valoresmayores_G5, lista_valoresmayores_G6,
                          lista_valoresmayores_G7, lista_valoresmayores_G8, lista_valoresmayores_G9, lista_valoresmayores_G10, lista_valoresmayores_G11, lista_valoresmayores_G12]

valormin_G1 = np.min(lista_valoresmayores_G[0])
valormin_G2 = np.min(lista_valoresmayores_G[1])
valormin_G3 = np.min(lista_valoresmayores_G[2])
valormin_G4 = np.min(lista_valoresmayores_G[3])
valormin_G5 = np.min(lista_valoresmayores_G[4])
valormin_G6 = np.min(lista_valoresmayores_G[5])
valormin_G7 = np.min(lista_valoresmayores_G[6])
valormin_G8 = np.min(lista_valoresmayores_G[7])
valormin_G9 = np.min(lista_valoresmayores_G[8])
valormin_G10 = np.min(lista_valoresmayores_G[9])
valormin_G11 = np.min(lista_valoresmayores_G[10])
valormin_G12 = np.min(lista_valoresmayores_G[11])
valormin_G = [valormin_G1, valormin_G2, valormin_G3, valormin_G4, valormin_G5, valormin_G6, 
              valormin_G7, valormin_G8, valormin_G9, valormin_G10, valormin_G11, valormin_G12]

gesto_1[gesto_1 >= valormin_G[0]] = valormin_G[0]
gesto_2[gesto_2 >= valormin_G[1]] = valormin_G[1]
gesto_3[gesto_3 >= valormin_G[2]] = valormin_G[2]
gesto_4[gesto_4 >= valormin_G[3]] = valormin_G[3]
gesto_5[gesto_5 >= valormin_G[4]] = valormin_G[4]
gesto_6[gesto_6 >= valormin_G[5]] = valormin_G[5]
gesto_7[gesto_7 >= valormin_G[6]] = valormin_G[6]
gesto_8[gesto_8 >= valormin_G[7]] = valormin_G[7]
gesto_9[gesto_9 >= valormin_G[8]] = valormin_G[8]
gesto_10[gesto_10 >= valormin_G[9]] = valormin_G[9]
gesto_11[gesto_11 >= valormin_G[10]] = valormin_G[10]
gesto_12[gesto_12 >= valormin_G[11]] = valormin_G[11]

normxgsts = mat2gray(abscwt)
normxgsts[:,0:1564], normxgsts[:,3052:4988], normxgsts[:,6770:11213], normxgsts[:,13041:17560], normxgsts[:,19224:23394], normxgsts[:,25045:29597], normxgsts[:,31036:34043], normxgsts[:,35613:37688], normxgsts[:,39625:43804], normxgsts[:,45577:49213], normxgsts[:,50579:54123], normxgsts[:,55691:59187], normxgsts[:,60809:62365] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
normxgsts[:,1565:3051] = mat2gray(gesto_1) 
normxgsts[:,4989:6769] = mat2gray(gesto_2)
normxgsts[:,11214:13040] = mat2gray(gesto_3)
normxgsts[:,17561:19223] = mat2gray(gesto_4)
normxgsts[:,23395:25044] = mat2gray(gesto_5)
normxgsts[:,29598:31035] = mat2gray(gesto_6)
normxgsts[:,34044:35612] = mat2gray(gesto_7)
normxgsts[:,37689:39624] = mat2gray(gesto_8)
normxgsts[:,43805:45576] = mat2gray(gesto_9)
normxgsts[:,49214:50578] = mat2gray(gesto_10)
normxgsts[:,54124:55690] = mat2gray(gesto_11)
normxgsts[:,59188:60808] = mat2gray(gesto_12)

fs = samplerate

#Nombre de cada Gráfica
pacientes = ['paciente 01 ','paciente 02 ','paciente 03 ','paciente 04 ','paciente 05 ','paciente 06 ','paciente 07 ','paciente 08 ','paciente 09 ','paciente 10 ']
canales = ['canal 1 ','canal 2 ','canal 3 ','canal 4 ','canal 5 ','canal 6 ','canal 7 ','canal 8 ']    
file_emg = 'gráfica EMG'
file_cwt = 'gráfica CWT cmor1.0-1.5'
file_end = 'gráfica CWT cmor1.0-1.5.png'
file_end_2 = 'gráfica EMG.png'

#Título de la Gráfica
title_emg = canales[1] + pacientes[8] + file_emg
title_cwt = canales[1] + pacientes[8] + file_cwt

#Nombre del archivo PNG
file_name = canales[1] + pacientes[8] + file_end
file_name_2 = canales[1] + pacientes[8] + file_end_2

# Vectores de tiempo
timeemg = np.arange(0, len(emg_funcional) / fs, 1 / fs)
bdt09 = [1.565, 3.051, 4.989, 6.769, 11.214, 13.04, 17.561, 19.223, 23.395, 25.044, 29.598, 31.035, 
         34.044, 35.612, 37.689, 39.624, 43.805, 45.576, 49.214, 50.578, 54.124, 55.690, 59.188, 60.808]


# Gráficas EMG
plt.figure(figsize=(16,10))
plt.plot(timeemg, emg_funcional, 'b', label='Señal EMG RAW')
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.title(title_emg)
plt.xlim(0, timeemg.max())
plt.grid()
#plt.axhline(y=0.01,color='k')
plt.axvline(x=bdt09[0], color='c', linewidth=1.2, label="G1 - Mano en Reposo")
plt.axvline(x=bdt09[1], color='c', linewidth=1.2)
plt.axvline(x=bdt09[2], color='m', linewidth=1.2, label="G2 - Mano Cerrada en Puño")
plt.axvline(x=bdt09[3], color='m', linewidth=1.2)
plt.axvline(x=bdt09[4], color='y', linewidth=1.2, label="G3 - Flexión de la Muñeca")
plt.axvline(x=bdt09[5], color='y', linewidth=1.2)
plt.axvline(x=bdt09[6], color='r', linewidth=1.2, label="G4 - extensión de la Muñeca")
plt.axvline(x=bdt09[7], color='r', linewidth=1.2)
plt.axvline(x=bdt09[8], color='g', linewidth=1.2, label="G5 - Desviación Radial")
plt.axvline(x=bdt09[9], color='g', linewidth=1.2)
plt.axvline(x=bdt09[10], color='k', linewidth=1.2, label="G6 - Desviación Cubital")
plt.axvline(x=bdt09[11], color='k', linewidth=1.2)
plt.axvline(x=bdt09[12], color='c', linewidth=1.2)
plt.axvline(x=bdt09[13], color='c', linewidth=1.2)
plt.axvline(x=bdt09[14], color='m', linewidth=1.2)
plt.axvline(x=bdt09[15], color='m', linewidth=1.2)
plt.axvline(x=bdt09[16], color='y', linewidth=1.2)
plt.axvline(x=bdt09[17], color='y', linewidth=1.2)
plt.axvline(x=bdt09[18], color='r', linewidth=1.2)
plt.axvline(x=bdt09[19], color='r', linewidth=1.2)
plt.axvline(x=bdt09[20], color='g', linewidth=1.2)
plt.axvline(x=bdt09[21], color='g', linewidth=1.2)
plt.axvline(x=bdt09[22], color='k', linewidth=1.2)
plt.axvline(x=bdt09[23], color='k', linewidth=1.2)
plt.legend(loc='lower left', fontsize='x-small', borderpad=None)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useOffset=True)
plt.savefig(file_name_2, dpi = 201)  
plt.show()

#Parametros Matemáticos Cambio de eje Tiempo
reduction_to= int(np.floor(len(abscwt) / 4)) #Intervalos de Ejes
time_axis = np.linspace(0.0, emg_funcional.shape[0] * np.array(dt), emg_funcional.shape[0]+1)
time = np.arange(0, time_axis.shape[0] + 1, int(reduction_to*2.5))
output_time_axis = time_axis[time]
outtime = np.round(output_time_axis, 0)

#Parametros Matemáticos Cambio de eje Frecuencia
scales = range(0,num-1)
step = 200 
outscales = scales[0:num:step]
outfreqs = np.round(frequencies[0:num:step], 2)

# Gráfica de CWT Normalizada
plt.figure(figsize=(16,10))
plt.text(int(bdt09[0]*1000), int(len(outscales)+1500), 'G1', color='w')
plt.text(int(bdt09[2]*1000), int(len(outscales)+1500), 'G2', color='w')
plt.text(int(bdt09[4]*1000), int(len(outscales)+1500), 'G3', color='w')
plt.text(int(bdt09[6]*1000), int(len(outscales)+1500), 'G4', color='w')
plt.text(int(bdt09[8]*1000), int(len(outscales)+1500), 'G5', color='w')
plt.text(int(bdt09[10]*1000), int(len(outscales)+1500), 'G6', color='w')
plt.text(int(bdt09[12]*1000), int(len(outscales)+1500), 'G1', color='w')
plt.text(int(bdt09[14]*1000), int(len(outscales)+1500), 'G2', color='w')
plt.text(int(bdt09[16]*1000), int(len(outscales)+1500), 'G3', color='w')
plt.text(int(bdt09[18]*1000), int(len(outscales)+1500), 'G4', color='w')
plt.text(int(bdt09[20]*1000), int(len(outscales)+1500), 'G5', color='w')
plt.text(int(bdt09[22]*1000), int(len(outscales)+1500), 'G6', color='w')
plt.imshow(normxgsts, cmap='jet', aspect ='auto')
plt.xticks(time, outtime)
plt.yticks(outscales,outfreqs)
plt.ylim(top=1100)
#plt.gca().invert_yaxis()
plt.colorbar()
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.title(title_cwt)
plt.savefig(file_name, dpi = 201)
plt.show()
