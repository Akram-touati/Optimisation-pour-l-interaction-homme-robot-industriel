import sim 
import vrep
import time
import numpy as np


def myFunction(v):
    sim.simxFinish(-1)
    
    clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5)
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
    

    """DATA transfer """
    cityList = []
    x = len(v)
    print(x)
    for i in range (len(v)):
        if i<= len(v)-3:
            
            for j in range (7):
                cityList.append(v[i][j])
        else:
            for k in range (len(v)-2):
                cityList.append(v[i][k])
    
    time.sleep(1)
    if clientID!= -1:
        
        
        
        
        emptyBuff = bytearray()
        res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(
            clientID,'/LBR',vrep.sim_scripttype_childscript,'myFunction',[x],np.radians(cityList),[],emptyBuff,vrep.simx_opmode_blocking)
        
        time.sleep(0.7)
        

        while True:

            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(
                clientID,'/LBR',vrep.sim_scripttype_childscript,'VREP_PYTHON',[],[],['DATA Arrived'],emptyBuff,vrep.simx_opmode_blocking)
            time.sleep(0.5)            
            
            if retInts != []:
                Fitness= retFloats
                retInts = []
                break
                
                
        
            
        
    else:
        print('Could not connect')   
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

    
    
    print ('Program ended')
    return Fitness



