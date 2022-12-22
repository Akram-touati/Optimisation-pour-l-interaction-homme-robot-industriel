import sim 
import vrep
import time
import math
import numpy as np
import sys 
import Checkpoint1_Jaco
import csv
import pandas as pd

def myFunction(v):
    sim.simxFinish(-1)
    
    clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5)
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)


    """DATA transfer """
    cityList = []
    x = len(v)
    print(x)
    for i in range (12):
        if i<=9:
            
            for j in range (7):
                cityList.append(v[i][j])
        else:
            for k in range (10):
                cityList.append(v[i][k])
            
# for i in range(12):
#     for j in range(7):
#         l1.append(v[j])
#         l2.append(v[j+7])
#         l3.append(v[j+14])
#         l4.append(v[j+21])
#         l5.append(v[j+28])
#         l6.append(v[j+35])
#         l7.append(v[j+42])
#         l8.append(v[j+49])
#         l9.append(v[j+56])
#         l10.append(v[j+63])
#     for o in range(10):
#         l11.append(listt[o+70])
#         l12.append(listt[o+80])

           
     
        
        # if i ==8:
        #     X = cityList[8]
    # np.radians(cityList[:len(cityList)-1])
            
    
    time.sleep(1)
    if clientID!= -1:
        
        
        
        
        emptyBuff = bytearray()
        res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(
            clientID,'/LBR',vrep.sim_scripttype_childscript,'myFunction',[x],np.radians(cityList),[],emptyBuff,vrep.simx_opmode_blocking)
        time.sleep(0.7)
        
        #♥time.sleep(15)
        
        # res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(
        #     clientID,'/LBR',vrep.sim_scripttype_childscript,'VREP_PYTHON',[],[],['I am Here'],emptyBuff,vrep.simx_opmode_blocking)

        # Fitness= retFloats
            

        while True:

            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(
                clientID,'/LBR',vrep.sim_scripttype_childscript,'VREP_PYTHON',[],[],['I am Here'],emptyBuff,vrep.simx_opmode_blocking)
            time.sleep(0.5)
            # if retInts[0]  == 1:
            #     Fitness= retFloats
            #     retInts[0]  = 0
                
            #     break
            
            
            
            if retInts != []:
                print("I'm here")
                Fitness= retFloats
                retInts = []
                break
                
                
        
            
        
    else:
        print('Could not connect')   
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

    
    
    print ('Program ended')
    return Fitness





#Ecrire sur un fichier CSV:
# def CSVfunc(Time,Energy,Velocity,Acceleration):
#     with open('CSV_ETVA.csv', 'w', newline='') as f:
#         relation=csv.writer(f)
#         #relation.writerow(['Position des Tags selon x',Position des Tags selon y,'distance tag-robot,'Position du robot selon x', 'position du robot selon y', Oriantation (Theta) du robot])
#         relation.writerow(['Time','Energy',"Velocity","Aceleration"])
#         for i in range(0,len(Time)):
#             relation.writerow([Time[i],Energy[i],Velocity[i],Acceleration[i]])  #,x_Robot[i],y_Robot[i],oriantation_robot[i]])
            
    
#     data= pd.read_csv("CSV_ETVA.csv")
#     data
#     return



# normalisation des valeurs