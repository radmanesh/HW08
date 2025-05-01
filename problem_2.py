#the intial framework for a particle swarm optimization for Schwefel minimization problem
#author: Charles Nicholson
#for ISE/DSA 5113


#need some python libraries
import copy
import math
from random import Random
import numpy as np


#to setup a random number generator, we will specify a "seed" value
seed = 12345
myPRNG = Random(seed)

#to get a random number between 0 and 1, write call this:             myPRNG.random()
#to get a random number between lwrBnd and upprBnd, write call this:  myPRNG.uniform(lwrBnd,upprBnd)
#to get a random integer between lwrBnd and upprBnd, write call this: myPRNG.randint(lwrBnd,upprBnd)

lowerBound = -500  #bounds for Schwefel Function search space
upperBound = 500   #bounds for Schwefel Function search space

#you may change anything below this line that you wish too -----------------------------------------------------

#note: for the more experienced Python programmers, you might want to consider taking a more object-oriented approach to the PSO implementation, i.e.: a particle class with methods to initialize itself, and update its own velocity and position; a swarm class with a method to iterates through all particles to call update functions, etc.

#number of dimensions of problem
dimensions = 200

#number of particles in swarm
swarmSize = 100

maxIterations = 10000  # maximum number of iterations
noImprovementMaxIterations = 50  # number of iterations without improvement before stopping

#Schwefel function to evaluate a real-valued solution x
# note: the feasible space is an n-dimensional hypercube centered at the origin with side length = 2 * 500

def evaluate(x):
    val = 0
    d = len(x)
    for i in range(d):
        val = val + x[i]*math.sin(math.sqrt(abs(x[i])))

    val = 418.9829*d - val

    return val

# gunction that returns the global best position of the swarm
def getGlobalBest(pBest, pBestVal):
    gBest = pBest[0][:]  #initialize gBest to the first particle's best position
    gBestVal = pBestVal[0]  #initialize gBestVal to the first particle's best value

    for i in range(1, len(pBest)):
        if pBestVal[i] < gBestVal:
            gBest = pBest[i][:]
            gBestVal = pBestVal[i]

    return gBest, gBestVal

# PSO parameters
w = 0.729  # inertia weight
c1 = 1.5  # cognitive coefficient
c2 = 1.5  # social coefficient
velocityMax = 0.1 * (upperBound - lowerBound)  # maximum velocity

# velocity update function
def updateVelocities(vel, pos, pBest, gBest, w=w, c1=c1, c2=c2):  # corrected parameter w and added c2
    for i in range(len(vel)):
        for j in range(len(vel[i])):
            r1 = myPRNG.random()
            r2 = myPRNG.random()
            vel[i][j] = (w * vel[i][j]) + (c1 * r1 * (pBest[i][j] - pos[i][j])) + (c2 * r2 * (gBest[j] - pos[i][j]))
            # check if the velocity is within bounds
            if vel[i][j] < -velocityMax:
                vel[i][j] = -velocityMax
            elif vel[i][j] > velocityMax:
                vel[i][j] = velocityMax
    return vel

# position update function
def updatePositions(pos, vel):  # corrected function name
    for i in range(len(pos)):
        for j in range(len(pos[i])):
                pos[i][j] += vel[i][j]
                # check if the position is within bounds
                if pos[i][j] < lowerBound:
                    pos[i][j] = lowerBound
                elif pos[i][j] > upperBound:
                    pos[i][j] = upperBound
    return pos

#function to generate a summary of the swarm's current state
def summarizeSwarm(pos, vel, pBest, pBestVal):
    vel_array = np.array(vel)
    p_array = np.array(pBestVal)

    mean_vel = np.mean(vel_array)
    std_vel = np.std(vel_array)
    mean_p = np.mean(p_array)
    std_p = np.std(p_array)

    return mean_vel, std_vel, mean_p, std_p  # updated return statement to include mean_p and std_p

#the swarm will be represented as a list of positions, velocities, values, pbest, and pbest values

pos = [[] for _ in range(swarmSize)]      #position of particles -- will be a list of lists; e.g., for a 2D problem with 3 particles: [[17,4],[-100,2],[87,-1.2]]
vel = [[] for _ in range(swarmSize)]      #velocity of particles -- will be a list of lists similar to the "pos" object

#note: pos[0] and vel[0] provides the position and velocity of particle 0; pos[1] and vel[1] provides the position and velocity of particle 1; and so on.

curValue = [] #evaluation value of current position  -- will be a list of real values; curValue[0] provides the evaluation of particle 0 in it's current position
pBest = []    #particles' best historical position -- will be a list of lists: pBest[0] provides the position of particle 0's best historical position
pBestVal = [] #value of pbest position  -- will be a list of real values: pBestVal[0] provides the value of particle 0's pbest location


#initialize the swarm randomly
for i in range(swarmSize):
    for j in range(dimensions):
        pos[i].append(myPRNG.uniform(lowerBound,upperBound))    #assign random value between lower and upper bounds
        vel[i].append(myPRNG.uniform(-1,1))                     #assign random value between -1 and 1   --- maybe these are good bounds?  maybe not...
        # vel[i].append(myPRNG.uniform(-velocityMax, velocityMax))  # assign random velocity within the max bounds
    curValue.append(evaluate(pos[i]))   #evaluate the current position

pBest = pos[:]          # initialize pbest to the starting position
pBestVal = curValue[:]  # initialize pbest to the starting position

gBest , gBestVal = getGlobalBest(pBest, pBestVal)  # get the global best position and value

print(f"Initial Global Best Value = {gBestVal}")
print(f"Initial Global Best Position = {gBest}")

iteration = 0  # iteration counter


# main loop to find the global best position
done = False
noImprovementIterations = 0  # counter for iterations without improvement

while not done:
    noImprovementIterations += 1  # increment the no improvement counter
    # evaluate the swarm
    for i in range(swarmSize):
        curValue[i] = evaluate(pos[i])  #evaluate the current position
        #update pbest if the current value is better than the historical best
        if curValue[i] < pBestVal[i]:
            pBest[i] = pos[i][:]  #copy the current position to the pbest position
            pBestVal[i] = curValue[i]  #copy the current value to the pbest value
            if curValue[i] < gBestVal:  #update the global best if the current value is better than the historical best:
                gBest = pos[i][:]
                gBestVal = curValue[i]
                noImprovementIterations = 0  # reset the no improvement counter

    # update the velocity and position of the particles
    #gBest, gBestVal = getGlobalBest(pBest, pBestVal)  #get the global best position and value
    vel = updateVelocities(vel, pos, pBest, gBest)
    pos = updatePositions(pos, vel)  #update the positions of the particles

    summary = summarizeSwarm(pos, vel, pBest, pBestVal)  # get the summary of the swarm
    mean_vel, std_vel, mean_p, std_p = summary  # unpack the summary
    print(f"Iteration {iteration}: Global Best Value = {gBestVal} , Mean Velocity = {mean_vel}, Std Velocity = {std_vel}, Mean pBestVal = {mean_p}")

    # check stopping criteria
    if iteration >= maxIterations or noImprovementIterations >= noImprovementMaxIterations:
        done = True  #stop the loop if the maximum number of iterations is reached

    iteration += 1  #increment the iteration counter

print(f"Final Global Best Value = {gBestVal}")


