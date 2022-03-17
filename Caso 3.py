# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:31:09 2020

@author: amira
"""

import ross as rs
from bokeh.io import output_notebook, show
import numpy as np
from bokeh.layouts import row
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import pandas  as pd
import scipy
import sklearn
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from scipy.signal import hilbert
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
import time
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from scipy.fft import rfft, rfftfreq
sns.set(style="ticks")

scaler = StandardScaler()

def set_rotor(num_d,p):
    
    steel = rs.Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)
    steel.save_material()
    steel = rs.Material.load_material('Steel')
    
    "Definindo distancia entre mancais"
    d_entre =1442*10**-3
    "distancia total do do eixo"
    t_eixo = 1846*10**-3
    "Definindo o numero de divisões  dentro da distancia total do eixo"
    n= num_d + 1
    "Lista diametro dos impeledores? lista deve conter n-1 elementos "
    lista_diam =[420*10**-3]
    "Definindo parametros do eixo// tamanho do elementos de eixo entre os mancais"
    L = d_entre/n
    "largura do mancal"
    mancal_l =39.6*10**-3
    "lista com o tamanho entre os nós do eixo/ elementos do meio/ elementos antes do elemento que recebe o macal"
    l_list = [L]*n
    shaft_elements = [
        rs.ShaftElement(L=l,idl=0,odl=146*10**-3,material=steel)
        for l in l_list]
    "elementos que recebem o mancal"
    shaft_mancal0 =rs.ShaftElement(L=(mancal_l)/2,idl=0,odl=90*10**-3,material=steel)
    shaft_mancal1 = rs.ShaftElement(L=(mancal_l)/2,idl=0,odl=90*10**-3,material=steel)
    list_mancal0 = [shaft_mancal0,shaft_mancal0]
    list_mancal1 = [shaft_mancal1,shaft_mancal1]
    "acoplando os elementos que recebem os mancais no eixo"
    shaft_elements.insert(0,list_mancal0)
    shaft_elements.append(list_mancal1)
    "Definindo distancia entre o mancal e o acoplamento// 0<d<404// distaAncia pro lado livre = LL"
    d_mancal_acopl = 202*10**-3
    d_LL= t_eixo-d_entre-d_mancal_acopl
    "elementos que recebem o acoplamento"
    shaft_acoplamento0 =rs.ShaftElement(L=d_mancal_acopl,idl=0,odl=80*10**-3,material=steel)
    shaft_acoplamento1 = rs.ShaftElement(L=d_LL,idl=0,odl =80*10**-3,material=steel)
    "acoplando os elementos que recebem o acoplamento"
    shaft_elements.insert(0,shaft_acoplamento0)
    shaft_elements.append(shaft_acoplamento1)
    disk_elements = []
    '''for d in range (0,len(lista_diam)):
        disk_elements.insert(0,rs.DiskElement.from_geometry(n=(len(shaft_elements)+2)/2,
        material=steel,
        width=5.7*10**-3,
        i_d=146*10**-3,
        o_d=lista_diam[d]))'''
    "definindo disk rotor discos e adcionando a lista de shaft elements"
    n_list = list(range(4,4+num_d))
    m_list = [35.8]*num_d
    Id_list = [0.17808928]*num_d
    Ip_list = [0.32956362]*num_d
    N = len(n_list)
    disk_elements = [
        rs.DiskElement(
            n=n_list[i],
            m=m_list[i],
            Id=Id_list[i],
            Ip=Ip_list[i],
        )
        for i in range(N)
        ]
    "definindo os mancais/ lugar de adicionar os acoplamentos no Kxy"
    stfx=9e8
    stfy=9e8
    bearing0 = rs.BearingElement(n=2, kxx=np.random.normal(stfx,(p/100)*stfx), kyy=np.random.normal(stfy,(p/100)*stfy),cxx=1e4,cyy=1e4)
    bearing1 = rs.BearingElement(n=len(shaft_elements), kxx=np.random.normal(stfx,(p/100)*stfx), kyy=np.random.normal(stfy,(p/100)*stfy),cxx=1e4,cyy=1e4)
    bearing_elements = [bearing0, bearing1]
    "definindo o rotor"
    rotor1 = rs.Rotor(shaft_elements,disk_elements,bearing_elements)
    
    "plot rotor"
    #show(rotor1.plot_rotor())
    "plotando os resultados// tentar descobrir como plotar numa pagina só"
    
    '''static = rotor1.run_static()
    campbell = rotor1.run_campbell(np.linspace(0, 2000, 101))
    show(campbell.plot())
    #show(static.plot_free_body_diagram())
    #show(static.plot_deformation())
    show(rotor1.run_freq_response().plot(inp=20,out=16))
    modal = rotor1.run_modal(500)
    mode = 0
    modal.plot_mode3D(mode)'''

    return(rotor1)

"Funnção para criar os dados de medição do nó 2, de amplitude de movimento no eio x e y"
def train_gen (num_d,p, speed):
    n_mag = list(np.arange(0,15,15/100))
    size =1000
    x1 = []
    y1 =[]
    x2 = []
    y2 =[]
    class_list =[]
    class_list_name =[]
    for eq_mag in n_mag:
        rotor=set_rotor(num_d, p)
        eq_mag_c = ((eq_mag*1000)/speed)*1e-6
        magnitude = rotor.m * eq_mag_c *speed**2
        n_disk= [x.n for x in rotor.disk_elements]
        for node in n_disk:
            rotor=set_rotor(num_d, p)
            magnitude = rotor.m *1e-6* eq_mag *speed**2
            t =np.linspace(0, 10, size)
            F = np.zeros((size, rotor.ndof))
            F[:, 4 * node] = magnitude * np.cos(speed*t)
            F[:, 4 * node + 1] = magnitude * np.sin(speed*t)
            
            "Retirando a resposta de amplitude do nó 2 em x e y"
            response = rotor.run_time_response(speed, F, t)
            
            
            "Retirando o sinal de resposta no tempo para os mancais nas direções x e y"
            '''unbalance_aux1 = response.yout[:, 8]
            unbalance_aux2 = response.yout[:, 9]
            unbalance_aux3 = response.yout[:, (n+4)*4]
            unbalance_aux4 = response.yout[:, ((n+4)*4)+1]'''
            
            
            unbalance_aux1 = response.yout[:, rotor.bearing_elements[0].n*4]
            unbalance_aux2 = response.yout[:, (rotor.bearing_elements[0].n*4)+1]
            unbalance_aux3 = response.yout[:, (rotor.bearing_elements[1].n)*4]
            unbalance_aux4 = response.yout[:, ((rotor.bearing_elements[1].n)*4)+1]
            
        
            
            
            
            "Colocando os resultado do nó na lista geral de resultados"
            x1.append(unbalance_aux1)
            y1.append(unbalance_aux2)
            x2.append(unbalance_aux3)
            y2.append(unbalance_aux4)
            
            
# =============================================================================
#               
#             "Criando lista com labels para disoc de origem do desabalnceamento"
#             if node ==4 :
#                 dl="D1"
#             elif node ==5 :
#                 dl="D2"
#             elif  node ==6 :
#                 dl="D3"
# =============================================================================
            "Criando lista com labels para disoc de origem do desabalnceamento"
            dl = "D" + str(n_disk.index(node)+1)


            "Criando lista com labels para magnitude de desabalnceamento"
            if eq_mag<= 2.5 :
                dgn = 0
                dgc="DB"
            elif 2.5 < eq_mag <= 6.3:
                dgn = 1
                dgc="DM1"
            elif  6.3 < eq_mag<= 10: 
                dgn = 2
                dgc="DM2"
            elif  10 < eq_mag: 
                dgn= 3
                dgc="DG"

                
# =============================================================================
#             "Criando lista com labels Acoplando posição do disco e magnitude de desabalnceamento"
#             if eq_mag<= 2.5 :
#                 c = 0
#             elif 2.5 < eq_mag <= 3.6:
#                c = len(n_disk)*1
#             elif  3.6 < eq_mag<= 10: 
#                 c = len(n_disk)*2
#             
#             elif  10 < eq_mag: 
#                 c = len(n_disk)*3
# =============================================================================
         
            
                
            #class_list.append(0)
            class_list_name.append(dgc+dl)
            
            le = preprocessing.LabelEncoder()
            
# =============================================================================
#             le.fit(['DBD1', 'DBD2', 'DBD3', 'DM1D1', 'DM1D2', 'DM1D3', 'DM2D1',
#        'DM2D2', 'DM2D3', 'DGD1', 'DGD2', 'DGD3'])
# =============================================================================
            le.fit(list(set(class_list_name)))
            
            class_list= le.transform(class_list_name)
            
            
    return (x1,y1,x2,y2,class_list,class_list_name)

"Função que retorna o valor maximo absoluto da lista de pontos"
def max_value (signal):
    unbalance_aux1 = max(signal[100:])
                          
    return (unbalance_aux1)

"Função que retorna Root mean square da lista de pontos"
def rms (signal):
    aux = [abs(x)**2 for x in signal]
    aux = sum(aux)/len(aux)
    aux =np.sqrt(aux)
    
    return(aux)

"Função que retorna energia do sinal"
def energy (signal):
    aux = sum([abs(x)**2 for x in signal])
    return (aux)

"Função que retorna o momento de energia"
def energy_moment (signal):
    size= signal.size
    t =np.linspace(0, 10, size)
    lista =[]
    for x in range(len(t)):
        aux = t[x]*(abs(signal[x])**2)
        lista.append(aux)
    resultado = sum(lista)
    
    return(resultado) 

"Função que retorna a frequência da função de acordo com a transformação de hilbert "
def hht (signal):
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase)
    resultado = sum(instantaneous_frequency)/len(instantaneous_frequency)
#"duvida sobre a unidade da frequencia, fazer teste com função mais simples"
    
    return(resultado)

def feature_ex (num,p, speed):
    start_time = time.time()
    
    "Gerando os dados de medição"
    train_data = train_gen(num,p, speed)
    
    "eecutando feature extraction"
    max_value_x1 = [max_value(x) for x in train_data[0]]
    max_value_y1 = [max_value(x) for x in train_data[1]]
    max_value_x2 = [max_value(x) for x in train_data[2]]
    max_value_y2 = [max_value(x) for x in train_data[3]]
    
    rms_x1 = [rms(x)for x in train_data[0]]
    rms_y1 =[rms(x)for x in train_data[1]]
    rms_x2 = [rms(x)for x in train_data[2]]
    rms_y2 =[rms(x)for x in train_data[3]]
    
    energy_x1 = [energy(x)for x in train_data[0]]
    energy_y1 = [energy(x)for x in train_data[1]]
    energy_x2 = [energy(x)for x in train_data[2]]
    energy_y2 = [energy(x)for x in train_data[3]]
    
    energy_moment_x1 = [energy_moment(x)for x in train_data[0]]
    energy_moment_y1 = [energy_moment(x)for x in train_data[1]]
    energy_moment_x2 = [energy_moment(x)for x in train_data[2]]
    energy_moment_y2 = [energy_moment(x)for x in train_data[3]]
    
    hht_x1 =[hht(x)for x in train_data[0]]
    hht_y1 =[hht(x)for x in train_data[1]]
    hht_x2 =[hht(x)for x in train_data[2]]
    hht_y2 =[hht(x)for x in train_data[3]]
    
# =============================================================================
#     "Criando um data frame para exibição dos resultados utilizando informação apenas de um mancal"
#     
# 
#     df = pd.DataFrame(list(zip(max_value_x1, max_value_y1,
#                                rms_x1, rms_y1, 
#                                energy_x1,energy_y1,
#                                energy_moment_x1, energy_moment_y1,
#                                train_data[4], train_data[5])), 
#                columns =["max_value_x1", "max_value_y1",
#                          "rms_x1", "rms_y1",
#                          "energy_x1","energy_y1",
#                         "energy_moment_x1", "energy_moment_y1",
#                         "class", "class name"]) 
# =============================================================================
    
    

    "Criando um data frame para exibição dos resultados utilizando informações de ambos os mancais"
    

    df = pd.DataFrame(list(zip(max_value_x1, max_value_y1,max_value_x2, max_value_y2,
                                rms_x1, rms_y1,rms_x2, rms_y2, 
                                energy_x1,energy_y1,energy_x2,energy_y2,
                                energy_moment_x1, energy_moment_y1,energy_moment_x2, energy_moment_y2,
                                hht_x1, hht_y1,hht_x2, hht_y2,
                                train_data[4], train_data[5])), 
                columns =["max_value_x1", "max_value_y1","max_value_x2", "max_value_y2",
                          "rms_x1", "rms_y1","rms_x2", "rms_y2",
                          "energy_x1","energy_y1","energy_x2","energy_y2",
                        "energy_moment_x1", "energy_moment_y1","energy_moment_x2", "energy_moment_y2",
                        "hht_x1", "hht_y1","hht_x2", "hht_y2",
                        "class","class name"]) 

    
    print( (time.time() - start_time)/60)
    return (df)

def  Classificador (df,ft_type):
    
  
    
    
    "Preparando o tipo de feature utilizando ambos os mancais"
    ft_type_x1 = ft_type +"_x1"
    ft_type_y1 = ft_type +"_y1"
    ft_type_x2 = ft_type +"_x2"
    ft_type_y2 = ft_type +"_y2"
    
# =============================================================================
#     "Separando dados em entrada  e classes utilizando informação de um mancal apenas"
#     X= df[[ft_type_x1,ft_type_y1]]
# =============================================================================
    

    "Separando dados em entrada  e classes utilizando informação de ambos os mancais"
    X= df[[ft_type_x1,ft_type_y1,ft_type_x2,ft_type_y2]]
    

    
    Y = df ['class']
    
    "Padronizando os dados e separando em dados de treinamento e dados de teste"
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size= .2, random_state =17)
    
    "Criando  o classificadorKNN"
    clfknn = neighbors.KNeighborsClassifier()
    
    "Criando o classificadorSVC"
    clfsvc=make_pipeline(StandardScaler(), SVC(gamma='auto'))
    
    "Criando o classificador decision tree"
    clfdt = DecisionTreeClassifier(random_state=0)
    
    "Criando classificador random forest"
    clfrf = RandomForestClassifier(max_depth=4, random_state=0)
    
    "Criando classificador com rede neural"
    scaler.fit(x_train) 
    rnx_train = scaler.transform(x_train)
    rnx_test = scaler.transform(x_test)
    
    clfrn = MLPClassifier(random_state=1, max_iter=1000)
    
    "Treinando os classificadores"
    clfknn.fit(x_train,y_train)
    clfsvc.fit(x_train,y_train)
    clfdt.fit(x_train,y_train)
    clfrf.fit(x_train,y_train)
    clfrn.fit(rnx_train,y_train)
    
    
    #print(clfknn, clfsvc)
    
    "Classes reais e as classes de saída dos classificadores"
    y_expect = y_test
    y_pred_knn = clfknn.predict(x_test)
    y_pred_svc = clfsvc.predict(x_test)
    y_pred_dt = clfdt.predict(x_test)
    y_pred_rf = clfrf.predict(x_test)
    y_pred_rn = clfrn.predict(rnx_test)
    
    "pegando metricas de avaliação ds classificadores"
    
    "recall da classe 2"
    # recall_knn = recall_score(y_expect, y_pred_knn, average=None)[2]
    # recall_svc = recall_score(y_expect, y_pred_svc, average=None)[2]
    # recall_dt = recall_score(y_expect, y_pred_dt, average=None)[2]
    # recall_rf = recall_score(y_expect, y_pred_rf, average=None)[2]
    # recall_rn = recall_score(y_expect, y_pred_rn, average=None)[2]
    
    
    "accuracy"
    acc_knn = accuracy_score(y_expect, y_pred_knn)
    acc_svc = accuracy_score(y_expect, y_pred_svc)
    acc_dt = accuracy_score(y_expect, y_pred_dt)
    acc_rf = accuracy_score(y_expect, y_pred_rf)
    acc_rn = accuracy_score(y_expect, y_pred_rn)
    
    "Criando lista para criação do data frame"
    cla =["KNN","SVM","Decision tree","Radom forest", "Neural networks"]
    # recall =[recall_knn,recall_svc,recall_dt,recall_rf,recall_rn]
    acc = [acc_knn,acc_svc,acc_dt,acc_rf,acc_rn]
    
    
    
    "Dataframe com as comparações"
    #df = pd.DataFrame(list(zip(cla, acc,recall))
    df = pd.DataFrame(list(zip(cla, acc)), 
               # columns =["Classificadores","Accuracy", "Recall"])
               columns =["Classificadores","Accuracy"])

    
    
    
    '''print(metrics.classification_report(y_expect, y_pred_knn),
          metrics.classification_report(y_expect, y_pred_svc),
          metrics.classification_report(y_expect, y_pred_dt),
          metrics.classification_report(y_expect, y_pred_rf),
          metrics.classification_report(y_expect, y_pred_rn))'''
    
   
    return(df)
    
def Classificadores(df):
    start_time = time.time()
    
    list_ft =["max_value","rms", "energy","energy_moment"]
    df = [Classificador(df,x)for x in list_ft]
    print( (time.time() - start_time)/60)
    return (df)
    

def eval_dev_acc(n):
    dev =[0,10,20,30]
    resul = pd.DataFrame()
    for x in dev:
        df=feature_ex(3,x,800)
        r = Classificadores(df)
        aux= pd.DataFrame(r[n].T.head(2).tail(1))
        resul=resul.append(aux)
        
    return (resul)

def mchart (df):
    sns.pairplot(df, hue="class")
    
def feat_mchart(df,ft_type):
    
    
    ft_type_x1 = ft_type +"_x1"
    ft_type_y1 = ft_type +"_y1"
    ft_type_x2 = ft_type +"_x2"
    ft_type_y2 = ft_type +"_y2"
    
    df =df[[ft_type_x1,ft_type_y1,ft_type_x2,ft_type_y2,"class name"]]
    
    df.rename(columns = { ft_type_x1 : "RMS 1-x", ft_type_y1 : "RMS 1-y", ft_type_x2:"RMS 2-x",ft_type_y2:"RMS 2-y"}, inplace = True)
    
    g = sns.pairplot(df, hue="class name",diag_kind="hist")
    g.add_legend()
    
def featchart(df):
    
    
      
    ax1=sns.relplot(x="max_value_x1", y="max_value_y1", hue="class name", data=df);
    ax1.set(xlabel="max_value_x1", ylabel="max_value_y1")
           
    ax2 =sns.relplot(x="rms_x1", y="rms_y1", hue="class name", data=df);
    ax2.set(xlabel="rms_x1", ylabel="rms_y1")
         
    ax3 =sns.relplot(x="energy_x1", y="energy_y1", hue="class name", data=df);
    ax3.set(xlabel="energy_x1", ylabel="energy_y1")
           
    ax4 =sns.relplot(x="max_value_x1", y="max_value_y1", hue="class name", data=df);
    ax4.set(xlabel="energy_moment_x1", ylabel="energy_moment_y1")
    
def featchartx(df,ft_type):
    
    "Preparando o tipo de feature utilizando ambos os mancais"
    ft_type_x1 = ft_type +"_x1"
    ft_type_y1 = ft_type +"_y1"
    ft_type_x2 = ft_type +"_x2"
    ft_type_y2 = ft_type +"_y2"
    
    x = df[ft_type_x1]
    y = df[ft_type_y1]
    z = df[ft_type_x2]
    w = df[ft_type_y2]
    
    
    temp = df[[ft_type_x1, ft_type_y1, ft_type_x2,ft_type_y2,"class name" ]]
    
    sns.pairplot(temp, hue="class name", diag_kind="hist")
    
  
   
    
    # =============================================================================
    # "max_value_x1", "max_value_y1",
    #                          "rms_x1", "rms_y1",
    #                          "energy_x1","energy_y1",
    #                         "energy_moment_x1", "energy_moment_y1"
    # =============================================================================
    
def tfft (signal):
    SAMPLE_RATE =1000
    yf = rfft(signal)
    xf = rfftfreq(1000, 1 / SAMPLE_RATE)
    plt.plot(xf,np.abs(yf))
    plt.show()
    
     
