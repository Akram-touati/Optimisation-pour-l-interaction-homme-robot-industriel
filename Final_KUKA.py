import sim 
import vrep
import time
import Final_myFunction

class Kuka(object):
    def init(self):
        self.clientID= None
    
    def API_cnx(self):
        sim.simxFinish(-1)
        self.clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5)
    
        if self.clientID== -1:
            print('Could not connect')
            
        else:
            print("API Connexion Established ")
            
        return 
    
    def Start_Simulation(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)        
        
        
        return  
    
    def Stop_Simuation(self):
        
        time.sleep(0.1)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(0.1)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

def main(v):
    robot = Kuka()
    robot.API_cnx()
    fit = Final_myFunction.myFunction(v)
    robot.Stop_Simuation()
    
    return fit
