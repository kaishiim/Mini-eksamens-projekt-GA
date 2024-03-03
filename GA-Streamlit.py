"""
Genetic Algorithms are a family of evolutionary algorithms which can be implemented in any language (including python)
they solve problems which have no clear solution by generating random solutions and picking the best results 
then applying a crossover and a mutation to the best solutions before starting the process again. 
"""

import csv
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from random import randint, shuffle
import streamlit as st
import requests 


### adjustable parameters
num_pop = 50000
num_gen = 150
hist_len = 700
x3_min, x3_max = 200, 650


def getfloat(x):
    return float(x.replace(",", '.'))


def createtour(numcities):
    tour = [i for i in range(1, numcities)]
    shuffle(tour)
    tour = [0] + tour
    return tour
    

def tourlen(tour, dist):
    numcities = len(tour)
    sum = 0
    for i in range(numcities):
        fra, til = tour[i], tour[(i+1)%numcities]
        sum += dist[fra] [til]
    return(sum)


def creategenome(dist):
    numcities = len(dist)
    tour = createtour(numcities)
    tlength = tourlen(tour, dist) # Afstand/Fitness score
    return (tour, tlength)


def overcross(tour1, tour2):
    numcities = len(tour1)
    r = randint(1, numcities-1) # random integer [0:36]
    #print("kryds =", tour1[0:r], tour2[r:]+tour2[0:r]) 
    tour3 = tour1[0:r]  #splejs tour
    for i in range(numcities):
        x = tour2[(r + i) % numcities] # element med index r + i og modulo
        if x not in tour3:  # Betingelse som sikrer at en by ikke forekommer 2 gange 
            tour3.append(x)  
    return tour3


#### Main function ####


def TSP_GA(num_pop, num_gen):

    dummy = """
    # Data structures

    'coordinates': list of tuple(lattitude, longtitude)

    'dist': array of array of distance from city to city

    'tour': list of citynumber(0...37)

    'pop': list of tuple(tour, length)

    """

    url = "https://raw.githubusercontent.com/kaishiim/TSP_EU-GA/main/capitals.csv"
    response = requests.get(url)
    lines = list(line.decode('utf-8') for line in response.iter_lines())
    reader = csv.reader(lines, delimiter=';')
    header = next(reader)
    rows2 = []
    for row in reader:
        name, lat, lng, pop = row[0], getfloat(row[1]), getfloat(row[2]), int(row[3])
        rows2.append([name, lat, lng, pop])
    #print(rows2)
        


    # adjacency-distance matrix
    citynames = []
    coordinates = [] 
    for row in rows2:
        name, lat, lng, pop = row
        #print(name, lat, lng, pop) 
        citynames += [name] # assign citynamems elements
        coordinates.append((lat, lng)) # assign coordinates in same order as citynames elements
    #print("Citynames =", citynames)
    #print("City coordinates =", coordinates)


    #plot capitals routine
    if False:
        xx = [x[1] for x in coordinates]
        yy = [y[0] for y in coordinates]
        plt.title('Europe capitals')
        plt.xlabel('Longtitude')
        plt.ylabel('Lattitude')
        plt.scatter(xx, yy)    
        plt.show()
        #plt.draw()
        #plt.pause(5)
        #plt.close()


    # Initialize list of city names and matrix of adjacent distances 
    numcities = len(rows2)
    dist = np.zeros((numcities, numcities))  # Matrix creation
    for i in range(numcities):  # For all origin cities do
        c_ix, c_iy = coordinates[i][0], coordinates[i][1] # assign x,y coordinates for origin city 
        for j in range(numcities):  # For alle destionation cities do
            c_jx, c_jy = coordinates[j][0], coordinates[j][1]  # assign x,y coordinates for destination city
            dist[i][j] = sqrt((c_ix - c_jx)*(c_ix - c_jx) + (c_iy - c_jy)*(c_iy - c_jy))  # calculate distance between origin and destination city
    #print("dist =",dist)


    ## create population
    pop = [(creategenome(dist)) for i in range(num_pop)] # pop: list of tuples of tour and length
    pop.sort(key = lambda x:x[1])  # sorting to find minimum distance
    best_tour = pop[0]
    best_length_list = [best_tour[1]]

    # Plotting of iterative generations
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, width_ratios=(4, 3, 3), figsize=(15, 4))

    fig1, ax1 = plt.subplots()
    ax1.set_title('Europe TSP')
    ax1.set_xlabel('Longtitude')
    ax1.set_ylabel('Lattitude')

    fig2, ax2 = plt.subplots()
    ax2.set_title("Tour length over generations")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Tour length")

    fig3, ax3 = plt.subplots()
    ax3.set_title("Cumulative distribution over generations")
    ax3.set_xlabel("Tour length")
    ax3.set_ylabel("Count(%)")
    ax3.set_xlim(200,hist_len)


    for generation in range(1, num_gen+1):
        new_pop = []
        for k in range(num_pop):
            i, j = randint(1, num_pop-1), randint(1, num_pop-1)
            tour = overcross(pop[i][0], pop[j][0])
            tlength = tourlen(tour, dist)
            new_pop.append((tour, tlength))
        pop += new_pop  # Doubling of population (variation)
        pop.sort(key = lambda x:x[1])  # sorting to find minimum distance (selection)
        pop = pop[0:num_pop]  # selection
        hist = [0]*hist_len 
        for p in pop:
            inx = int(p[1]+0.5)
            if inx < hist_len:
                    hist[inx] += 1
        #print("hist =", hist)
        sum = 0
        cum_sum = [0]*len(hist)
        for i in range(len(hist)):
            sum += hist[i]
            cum_sum[i] = sum
        cum_sum = [100*c/num_pop for c in cum_sum]  # list comprehension (proncent omregning af cum_sum)
        #print("cum_sum=", cum_sum)
        best_tour = pop[0]
        best_length_list.append(best_tour[1])
        #print("gen, best_tour =", generation, best_tour)
        colours = ["r", "g", "b"]
        xx = [coordinates[city][1] for city in best_tour[0]]
        yy = [coordinates[city][0] for city in best_tour[0]] 
    
        # figure 1
        ax1.clear()
        ax1.set_title('Europe TSP %d: %f' % (generation, best_tour[1]))
        for i in range(numcities+1):
            i0 = i%numcities
            i1 = (i+1)%numcities
            ax1.plot([xx[i0], xx[i1]], [yy[i0], yy[i1]], marker="o", c=colours[i%len(colours)])
        ax1.set_xlabel('Longtitude')
        ax1.set_ylabel('Lattitude')
        ax1.plot(xx[0], yy[0], marker="o", c="black")    
        #plt.draw()
        
        # figure 2
        ax2.plot(best_length_list)
            
        # figure 3
        plt.axes(ax3)
        plt.xlim(x3_min, x3_max)
        xx = [i for i in range(hist_len)]
        plt.plot(xx, cum_sum)
        x_ticks = [t for t in range(x3_min, x3_max+1,50)]
        x_labels = [str(t) for t in x_ticks]
        plt.xticks(x_ticks, labels=x_labels,)
        #plt.pause(0.01)

    st.write('')
    st.write("""Figur 1: Visuallisering af sidste tur m. kortest totalafstand """
             """rundt om Europas hovedestader;  aka. den bedste tur ^_^""")
    st.pyplot(fig1)
    st.write('')
    st.write('')
    st.write('Figur 2: Plot der viser de bedste ture som funktion af antallet af generationer kørt')
    st.pyplot(fig2)
    st.write('')
    st.write('')
    st.write("""Figur 3: Kumulativfordeling af populationen idenfor en generationen, som funktion af turlængden..
                Kurver angiver procentdelen af populationen for en bestemt generation, som funnktion af tur længde. """ 
             """(Kurven helt til højre er allerførste generation)""")
    st.pyplot(fig3)

    plt.show()



##### Main section #####
#TSP_GA(100, 100)


st.title("Kør en gentisk algoritme for TSP på Europas hovedestader!")   
st.write("Du kan justere på 2 parametere på algoritmen ved at bruge nedenstående sliders:") 
num_pop = st.slider("Antal kromosomer/populations størelse (brug pile-taster til fin-justering)", 100, 50000, 10000) #10000
num_gen = st.slider("Antal generationer/iterationer", 1, 10000, 50) #50
st.write("En større population vil give algoritmen flere muligigheder at vælge imellem og dermed finde frem til en kortere totalafstand") 
st.write('')
st.write('')
if st.button('Start algoritmen'):
    TSP_GA(num_pop, num_gen)
