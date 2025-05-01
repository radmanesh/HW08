#basic genetic algorithm Python code provided as base code for the DSA/ISE 5113 course
#author: Charles Nicholson
#date: 4/5/2019

#NOTE: You will need to change various parts of this code.  However, please keep the majority of the code intact (e.g., you may revise existing logic/functions and add new logic/functions, but don't completely rewrite the entire base code!)
#However, I would like all students to have the same problem instance, therefore please do not change anything relating to:
#   random number generation
#   number of items (should be 150)
#   random problem instance
#   weight limit of the knapsack

#------------------------------------------------------------------------------

#Student name: Arman Radmanesh, Ali Abdullah
#Date: 04-29-2025


#need some python libraries
import copy
import math
from random import Random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#to setup a random number generator, we will specify a "seed" value
#need this for the random number generation -- do not change
seed = 51132023
myPRNG = Random(seed)

#to get a random number between 0 and 1, use this:             myPRNG.random()
#to get a random number between lwrBnd and upprBnd, use this:  myPRNG.uniform(lwrBnd,upprBnd)
#to get a random integer between lwrBnd and upprBnd, use this: myPRNG.randint(lwrBnd,upprBnd)

#number of elements in a solution (dimensions of Shwefel function)
dimensions = 2

#upper and lower bounds of the problem
lowerBound = -500  #bounds for Schwefel Function search space
upperBound = 500   #bounds for Schwefel Function search space

#change anything you like below this line, but keep the gist of the program ------------------------------------

#monitor the number of solutions evaluated
solutionsChecked = 0

populationSize = 5000 #size of GA population
Generations = 100  #number of GA generations

crossOverRate = 0.85  #crossover rate
mutationRate = 0.1  #mutation rate
eliteSolutions = 10  #number of elite solutions to retain

tournamentSize = 3  #number of competitors in tournament selection

#create an continuous valued chromosome
def createChromosome(d):
    #this code as-is expects chromosomes to be stored as a list, e.g., x = []
    #write code to generate chromosomes, most likely want this to be randomly generated
    x = [myPRNG.uniform(lowerBound, upperBound) for _ in range(d)]
    return x


#create initial population by calling the "createChromosome" function many times and adding each to a list of chromosomes (a.k.a., the "population")
def initializePopulation(): #n is size of population; d is dimensions of chromosome

    population = []
    populationFitness = []

    for i in range(populationSize):
        population.append(createChromosome(dimensions))
        populationFitness.append(evaluate(population[i]))

    tempZip = zip(population, populationFitness)
    popVals = sorted(tempZip, key=lambda tempZip: tempZip[1])

    #the return object is a reversed sorted list of tuples:
    #the first element of the tuple is the chromosome; the second element is the fitness value
    #for example:  popVals[0] is represents the best individual in the population
    #popVals[0] for a 2D problem might be  ([-70.2, 426.1], 483.3)  -- chromosome is the list [-70.2, 426.1] and the fitness is 483.3

    return popVals

#implement a crossover
def crossover(x1,x2):
    # if the random number is greater than the crossover rate, then do not perform crossover
    if(myPRNG.random() > crossOverRate):
        return x1, x2

    #x1 and x2 are the two parents
    d = len(x1)  #get the number of dimensions in the chromosome
    crossOverPt = myPRNG.randint(0, d-1)  #randomly select a crossover point
    beta = myPRNG.random()  #create a new random number between 0 and 1, named "beta"
    new1 = list(np.array(x1) - beta*(np.array(x1) - np.array(x2)))  #create a new list with the first alpha elements from x1
    new2 = list(np.array(x2) + beta*(np.array(x1) - np.array(x2)))  #create a new list with the first alpha elements from x2

    if(crossOverPt<d/2):
        offspring1 = x1[0:crossOverPt] + new1[crossOverPt:d]  #create the first offspring
        offspring2 = x2[0:crossOverPt] + new2[crossOverPt:d]  #create the second offspring
    else:
        offspring1 = new1[0:crossOverPt] + x1[crossOverPt:d]  #create the first offspring
        offspring2 = new2[0:crossOverPt] + x2[crossOverPt:d]  #create the second offspring

    return offspring1, offspring2  #two offspring are returned


#Schwefel function to evaluate a real-valued solution x
# note: the feasible space is an n-dimensional hypercube centered at the origin with side length = 2 * 500

def evaluate(x):
    val = 0
    d = len(x)
    for i in range(d):
            val = val + x[i]*math.sin(math.sqrt(abs(x[i])))

    val = 418.9829*d - val

    return val          #returns the fitness value of the chromosome x


#performs tournament selection; k chromosomes are selected (with repeats allowed) and the best advances to the mating pool
#function returns the mating pool with size equal to the initial population
def tournamentSelection(pop,k):

    #randomly select k chromosomes; the best joins the mating pool
    matingPool = []

    while len(matingPool)<populationSize:

        ids = [myPRNG.randint(0,populationSize-1) for i in range(k)]
        competingIndividuals = [pop[i][1] for i in ids]
        bestID=ids[competingIndividuals.index(min(competingIndividuals))]
        matingPool.append(pop[bestID][0])

    return matingPool

#performs roulette wheel selection; k chromosomes are selected (with repeats allowed) and the best advances to the mating pool
#function returns the mating pool with size equal to the initial population
def rouletteWheel(pop):

    totalFitness = sum(1 / individual[1] for individual in pop)  # calculate total fitness
    selectionProbabilities = [1 / individual[1] / totalFitness for individual in pop]  # calculate selection probabilities
    cumulativeProbabilities = [sum(selectionProbabilities[:i+1]) for i in range(len(selectionProbabilities))]  # cumulative probabilities

    matingPool = []
    for _ in range(populationSize):
        rand = myPRNG.random()
        for i, cumProb in enumerate(cumulativeProbabilities):
            if rand < cumProb:
                matingPool.append(pop[i][0])  # add the selected chromosome to the mating pool
                break

    return matingPool

#function to mutate solutions
#this is a simple mutation function that adjusts the value of each gene in the chromosome by a small uniform random amount
def mutate(x, epsilon=0.01):

    #create some mutation logic  -- make sure to incorporate "mutationRate" somewhere and don't do TOO much mutation
    if myPRNG.random() < mutationRate:
        for i in range(len(x)):
            x[i] += x[i] * myPRNG.uniform(-epsilon, epsilon)  #add a small random number to the chromosome
            if(x[i] < lowerBound):
                x[i] = lowerBound
            if(x[i] > upperBound):
                x[i] = upperBound

    return x




#breeding -- uses the "mating pool" and calls "crossover" function
def breeding(matingPool):
    #the parents will be the first two individuals, then next two, then next two and so on

    children = []
    childrenFitness = []
    for i in range(0,populationSize-1,2):
        child1,child2=crossover(matingPool[i],matingPool[i+1])

        child1=mutate(child1)
        child2=mutate(child2)

        children.append(child1)
        children.append(child2)

        childrenFitness.append(evaluate(child1))
        childrenFitness.append(evaluate(child2))

    tempZip = zip(children, childrenFitness)
    popVals = sorted(tempZip, key=lambda tempZip: tempZip[1])

    #the return object is a sorted list of tuples:
    #the first element of the tuple is the chromosome; the second element is the fitness value
    #for example:  popVals[0] is represents the best individual in the population
    #popVals[0] for a 2D problem might be  ([-70.2, 426.1], 483.3)  -- chromosome is the list [-70.2, 426.1] and the fitness is 483.3

    return popVals

#insertion step
def insert(pop,kids):

    #this is not a good solution here... essentially this is replacing the previous generation with the offspring and not implementing any type of elitism
    #at the VERY LEAST evaluate the best solution from "pop" to make sure you are not losing a very good chromosome from last generation
    #maybe want to keep the top 5? 10? solutions from pop -- it's up to you.

    # Check the k best solutions from the previous generation and keep them in the new generation if they are better than the new ones
    for i in range(eliteSolutions):
        if pop[i][1] < kids[i][1]:
            kids[i] = pop[i]

    return kids



#perform a simple summary on the population: returns the best chromosome fitness, the average population fitness, and the variance of the population fitness
def summaryFitness(pop):
    a=np.array(list(zip(*pop))[1])
    return np.min(a), np.mean(a), np.var(a)
#the best solution should always be the first element...
def bestSolutionInPopulation(pop):
    print ("Best solution: ", pop[0][0])
    print ("Value: ", pop[0][1])


# Function to visualize the population in the search space
def graphPopulation(pop):
    # Generated by Copilot
    # Extract chromosomes and fitness values
    chromosomes = [p[0] for p in pop]
    fitness = [p[1] for p in pop]

    # If 2D problem, create a scatter plot
    if dimensions == 2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot population points
        x = [c[0] for c in chromosomes]
        y = [c[1] for c in chromosomes]

        # Create scatter plot of population
        ax.scatter(x, y, fitness, c='blue', marker='o', label='Population')

        # Highlight the best solution
        ax.scatter(chromosomes[0][0], chromosomes[0][1], fitness[0],
                  c='red', marker='*', s=200, label='Best Solution')

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Fitness Value')
        ax.set_title('Population in Schwefel Function Search Space')
        ax.legend()

        plt.show(block=False)
        plt.pause(10)
    elif dimensions > 2:
        print("Population visualization not available for dimensions > 2")
    print ("Best solution: ", pop[0][0])
    print ("Value: ", pop[0][1])



def main():
    #GA main code
    Population = initializePopulation()

    # graphPopulation(Population)  #<--this is a function that will graph the population -- you can comment this out if you don't want to see the graph
    #optional: you can output results to a file -- i've commented out all of the file out put for now
    #f = open('out.txt', 'w')  #---uncomment this line to create a file for saving output


    for j in range(Generations):

        mates=tournamentSelection(Population,tournamentSize)  #<--need to replace this with roulette wheel selection, e.g.:  mates=rouletteWheel(Population)
        # mates = rouletteWheel(Population)
        Offspring = breeding(mates)
        Population = insert(Population, Offspring)

        #end of GA main code

        minVal, meanVal, varVal=summaryFitness(Population)          #check out the population at each generation
        print("Iteration: ", j, " min:", minVal, " mean:", meanVal, " var:", varVal)
        # graphPopulation(Population)

        #f.write(str(minVal) + " " + str(meanVal) + " " + str(varVal) + "\n")  #---uncomment this line to write to  file

    #f.close()   #---uncomment this line to close the file for saving output

    print (summaryFitness(Population))
    bestSolutionInPopulation(Population)


if __name__ == "__main__":
    main()



