

""" Here are some terms that you might need in order to understand the algorithm:
*** Gene        : a city (represented as (x, y) coordinates)
*** Individual  : (aka “chromosome”): a single route satisfying the conditions above
*** Population  : a collection of possible routes (i.e., collection of individuals)
*** Parents     : two routes that are combined to create a new route
*** Mating pool : a collection of parents that are used to create our next population (thus creating the next 
generation of routes)
*** Fitness     : a function that tells us how good each route is (in our case, how short the distance is)
*** Mutation    : a way to introduce variation in our population by randomly swapping two cities in a route
*** Elitism     : a way to carry the best individuals into the next generation"""


# Final GA code :
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from  typing import Iterable
import  Final_KUKA
import Transformation
import time
start_time = time.time() 
"Initialisation :"
T = []
E = []
Velocity =[]
Acceleration = []
DATA=[]
cityList = []
class Fitness1:
    def __init__(self, route):
        self.route = route
        self.fit = 0.0
        self.fitness  = 0.0
        
        
        
    def CalculFit(self):
        Fit = 0.0
        P = self.route
        v = P
        print("L'algorithme a donné le résultat suivant: ", v ) # Si vous voulez voir l'individu généré a chaque fois
        
        Fit = Final_KUKA.main(v)
        print ("Fitness = ",Fit)
        
        Fitt = 0.8 * Transformation.TimeTransfo(Fit[0]) + 0.2 * Transformation.EnergyTransfo(Fit[1])
        Fitt = 1/Fitt
        Fitt = 1/Fitt
        k = [Fitt, Fit]
        DATA.append(k)
        # print("Data = ",DATA)
        self.fit = Fitt
        return self.fit
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.CalculFit())
        return self.fitness



def flatten(items):
    for x in items:
        if isinstance(x,Iterable) and not isinstance(x, (str,bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x    

"Choisir l'ensemble de points "
"""
"Test pour 20 Points aléatoire"

p1=[-59.58761069,  14.32394488, -25.78310078, -84.79775368,  6.30253575,  82.5059225 , -74.48451337]
p2=[-114.01860123,   54.43099054,   -5.15662016,  -52.71211715, 5.15662016,   73.91155557, -108.28902328]
p3=[ -95.11099399,   20.05352283,  -29.79380535,  -65.89014644,9.74028252,   96.82986738, -103.70536092]
p4=[-105.4242343 ,   52.71211715,   -5.72957795,  -41.25296125,5.15662016,   86.51662706,  -91.10028943]
p5=[-102.55944533,   43.54479243,  -11.4591559 ,  -83.65183809,10.31324031,   55.00394833,  -89.95437384]
p6=[-61.30648408,  10.88619811, -33.23155212, -79.64113352,6.30253575,  91.10028943, -69.90085101]
p7=[ 30.93972094,  36.66929889,  11.4591559 , -71.61972439,-7.44845134,  72.19268219,   9.74028252]
p8=[ 52.13915936,  62.45239967,   0.5729578 , -61.87944187,-6.30253575,  56.72282172,  32.08563653]
p9=[  0.        ,  41.25296125,   0.        , -74.48451337,-0.        ,  64.74423085,   0.5729578 ]
p10=[  0.34377468,  60.16056849,   1.14591559, -56.72282172,-0.5729578 ,  63.59831526,   0.5729578 ]
p11=[-29.79380535,  56.14986392,  -0.5729578 , -56.72282172, 0.5729578 ,  68.18197762, -26.92901637]
p12=[-13.17802929,  20.62648062,  -1.71887339, -77.34930234,0.5729578 ,  81.9329647 , -26.35605858]
p13=[-71.61972439,  22.91831181, -20.62648062, -95.68395179,9.74028252,  63.59831526, -81.9329647 ]
p14=[-119.74817918,   57.86873731,   -8.02140913,  -72.19268219, 9.74028252,   52.71211715,  -92.81916281]
p15=[ -85.37071147,   22.34535401,  -18.33464944,  -66.46310424,7.44845134,   92.24620502, -115.73747462]
p16=[-36.09634109,  17.76169165,  -9.16732472, -67.60901983,2.86478898,  95.11099399, -68.75493542]
p17=[ 52.13915936,  29.79380535,  16.61577606, -79.06817573,-8.59436693,  72.19268219,  45.26366582]
p18=[ 29.79380535,  57.29577951,   0.5729578 , -70.4738088 ,-1.14591559,  52.71211715,  27.50197417]
p19=[  15.46986047,   21.19943842,    2.86478898, -104.27831871, -1.71887339,   55.00394833,   27.50197417]
p20=[ -6.30253575,  60.73352628,  -0.5729578 , -64.74423085,0.5729578 ,  55.00394833,  -1.71887339]
p21=[ 26.92901637,  21.77239621,   9.16732472, -52.71211715,        -3.43774677, 105.9971921 ,  18.33464944]
p22=[ 48.70141259,  50.42028597,   8.02140913, -91.10028943,        -9.74028252,  40.10704566,  24.0642274 ]

cityList = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20]
"""

"  Initialisation Liste des vecteurs"

"Test pour 7 Points aléatoire"
p1=[-103.88 ,50     ,1.26   ,-92.32 ,-1.45  ,37.67  ,4.4    ]           
p2=[-7.27   ,36.23  ,1.26   ,-55.72 ,-0.66  ,88.05  ,-4.55  ]           
p3=[116.22  ,34.79  ,1.26   ,-87.50 ,-0.78  ,57.66  ,50.49  ]            
p4=[-40.10  ,53.34  ,1.26   ,-66.46 ,-1.09  ,60.27  ,37.13  ]        
p5=[-56.97  ,28.88  ,1.26   ,-40.06 ,-0.6   ,111.0  ,50.24  ]        
p6=[-144.61 ,42.07  ,1.26   ,-48.58 ,-0.78  ,89.42  ,-37.34 ]        
p7=[-21.31  ,34.79  ,1.26   ,-87.51 ,-0.79  ,57.66  ,50.49  ]           

cityList = [p1,p2,p3,p4,p5,p6,p7]


# Velocity vector
cityList2 = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]

# Acceleration vector
cityList3= [0.50,0.70,0.90,1.0,1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.0]


    
""" Création de population """

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))    # Vecteur  des points
    Route = random.choices(cityList2,k = len(route))  # Vecteur vitesse
    RRoute = random.choices(cityList3,k = len(route)) # Vecteur acceleration
    route.append(Route)
    route.append(RRoute)
    global lengthRoute
    lengthRoute= len(Route)
    
    # Envoyer le chemin à vrep pour le simuler
    return route

""" La fonction Create Route ne produit qu'un seul individu, nous avons donc créé la fonction InitialPopulation
pour avoir autant de routes que nous voulons pour notre population """

def initialPopulation(popSize, cityList):
    population = []
    
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


"""  Fitness fonction  """


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness1(population[i]).routeFitness()
    print("Classement des indiviuds = ",sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True))
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


""""  Selection the mating pool  """


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    print("Selection is here : ")
    print( df )
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults



def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


"""  Bredding function aka children creation  """


def PREBREED_Vitesse(parent1,parent2):  
    B = parent1[len(parent1)-2 :len(parent1)-1 ]
    B = list(flatten(B))
    
    A = parent2[len(parent2)-2 : len(parent2)-1]
    A = list(flatten(A))
    
    KID = []
    
    index = int(len(parent1)/2)
    
    for i in range(index):
        KID.append(B[i])
        
    
    for i in range (index,len(A)):
        KID.append(A[i])
        if len(KID) > len(A):
            break
        
    return KID

def PREBREED_Accel(parent1,parent2):      
    BB = parent1[len(parent1)-1 :len(parent1) ]
    BB = list(flatten(BB))
    
    AA = parent2[len(parent2)-1 : len(parent2)]
    AA = list(flatten(AA))
    
    kid = []
    
    index = int(len(BB)/2)

    for i in range(index):
        kid.append(BB[i])
        
    
    for i in range (index,len(AA)):
        kid.append(AA[i])
        if len(kid) > len(AA):
            break
    
    return kid

def breed(parent1, parent2):
    
    child = []
    childP1 = []
    childP2 = []
    P1 = parent1
    P2 = parent2
    
    parent1 = parent1[:len(parent1)-2]
    parent2 = parent2[:len(parent2)-2]
    
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]
    
    child = childP1 + childP2           # Permutation des points
    
    x = PREBREED_Vitesse(P1,P2)         # Permutation Vitesse
    y = PREBREED_Accel(P1,P2)           # Permutation Acceleration
    child.append(x)                     
    child.append(y)
    return child        
     

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children 

"""  Mutation """
def Amutate(vector,mutationRate):
    vector = list(flatten(vector))
    for i in range (lengthRoute):
 
        if random.random() < mutationRate:
            if random.random() < 0.5:
                vector[i] += 0.1
            else:
                vector[i] -= 0.1
        while True:
            if vector[i] > 2:
                
                vector[i] = 2
                break
            elif vector[i] < 0.5:
                vector [i] = 0.5
                break
            else:
                break
    
    
    return vector

def Vmutate(V,mutationRate):
    V = list(flatten(V))
    for i in range (lengthRoute):
 
        if random.random() < mutationRate:
            if random.random() < 0.5:
                V[i] += 0.10
            else:
                V[i] -= 0.10
        while True:
            if V[i] > 1:
                
                V[i] = 1
                break
            elif V[i] <= 0.10:
                V [i] = 0.10
                break
            else:
                break
     
    return V



def mutate(individual, mutationRate):

    Avector = individual[len(individual)-1:len(individual)]    # Acceleration
    VVector = individual[len(individual)-2:len(individual)-1]  # Velocity

    individual = individual[:len(individual)-2]
    
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    
    x = list(flatten(Vmutate(VVector,mutationRate)))        # Vecteur vutesse aprés mutation
    y = list(flatten(Amutate(Avector,mutationRate)))        # Vecteur accélération aprés mutation
    individual.append(x)
    individual.append(y)

    return individual                                       # Vecteur chemin aprés mutation

def mutatePopulation(population, mutationRate, eliteSize):
    mutatedPop = []
    
    

    for i in range(0,eliteSize):
        mutatedPop.append(population[i])
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


"""  Repeating the Process  """


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate, eliteSize)
    return nextGeneration

"""  Genetic Algorithm Function """
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1])) # distance is the inverse of the fitness
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]          
    bestRoute = pop[bestRouteIndex]                 # Best route using indexes 
    print("best route is ",bestRoute)
    return bestRoute



"""  Plotting The Results """
def TE(A,P):
    Time = []
    Energy = []
    S=[]
    for i in range(len(P)):

        for j in range (len(A)):
            
            if A[j][0] == P[i]:
                S = A[j][1]
                # print("S = ",S)
                
                Time.append(S[0])
                Energy.append(S[1])
                break
    plt.plot(P)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()
    
    plt.plot(Time)
    plt.ylabel('Time')
    plt.xlabel('Generation')
    plt.show()
    print('Time = ',Time)
    plt.plot(Energy)
    plt.ylabel('Energy')
    plt.xlabel('Generation')
    plt.show()
    print('Energy = ',Energy)
    return

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Fitnesse initiale : " + str(1 / rankRoutes(pop)[0][1])) 
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        print("progress [",i+1,"] = ", progress)
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    print("Data = ", DATA)
    print("Progresse finale =", progress)
    TE(DATA,progress)
    
    print("Fitness finale : " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]          
    bestRoute = pop[bestRouteIndex]                              # best route to take 
    print("Meilleure résultat est : ",bestRoute)
    print("Simulation time = ", time.time()-start_time)

        #######################################################################################

"""  Launching the Algorithm """

n = input('Do you want to plot the results ? 1 for YES Or 0 for NO : ')
g = input('Number Of Generations Desired : ')
if (int(n) == 1):
    geneticAlgorithmPlot(population=cityList, popSize=5, eliteSize=2, mutationRate=0.02, generations=int(g))
else:
    geneticAlgorithm(population=cityList, popSize=15, eliteSize=2, mutationRate=0.02,   generations=int(g))
        
    


                   
    
### Without Plotting :
#geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=79)
### Plotting the results :
#geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=1000)
