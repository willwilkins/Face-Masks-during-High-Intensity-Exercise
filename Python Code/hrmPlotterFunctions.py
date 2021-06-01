# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:29:29 2021

Functions for the test which sees what the model does over a very long time

Functions to optimise lamda to fit recovery data.

@author: willi
"""

#import libraries
import numpy as np
#import scipy as sp
import csv

#constants found in Zak's 2015 paper
a, a1, a2, a3, a4, a5, a6, a7, a8 = 0.08, 10, 10, 4, 0.003, 4, 0.5, 420, 2.3E-6

#L0 = 6.5 # according to the literature
Lbasal = 1. # mM, unsure if I need to convert units (E-3 M)?

def HRmin(lamda, sex):
    """binary-sex heart rate minimum estimate using lamda"""
    
    if sex == 'male':
        if lamda == 0:
            return 10000000 # a really big number
        else:
            return 35/lamda
    elif sex == 'female':
        if lamda == 0:
            return 10000000 # a really big number
        else:
            return 35/lamda + 5
    else:
        raise Exception('ArgumentError: Please enter either Male or Female into HRmin.') # idk if this will work


def HRmax(age):
    """Finds the maximum heart rate of a participant based on their age"""
    return 220 - age


def fmax(HR, age):
    """Repeller term from HRmax"""
    return np.exp(-((HR - HRmax(age))/a1)**2) - 1


def fmin(HR, lamda, sex):
    """Repeller term from HRmin"""
    return 1 - np.exp(-((HR-HRmin(lamda, sex))/a2)**2)


def fD(HR, HR0, HRrest, lamda, L0, age, sex, t):
    """Attractor term toward heart-rate demand"""
    
    d = a*lamda
    
    Dss = HRrest
    DLa = a3*(Lbasal + (L0-Lbasal) * np.exp(- a8*lamda/(L0-Lbasal) * t**2))
    
    Dhat = Dss + DLa

    o = a4*lamda*((HRmax(age)-HRmin(lamda, sex))/(HR0-HRmin(lamda, sex)))**a5
    
    D = Dhat + (HR0 - Dhat) * np.exp(-o*t**2)
    
    return - d*(HR-D)


def HRdot(HR, HR0, HRrest, lamda, L0, age, sex, t):
    """Maria Zakynthinaki's instantaneous non-linear model of heart rate kinetics"""
    
    f1 = fmin(HR, lamda, sex)
    #print('fmin = ', f1)
    f2 = fmax(HR, age)
    #print('fmax = ', f2)
    f3 = fD(HR, HR0, HRrest, lamda, L0, age, sex, t)
    #print('fD = ', f3)
    
    dHRdt = -f1*f2*f3
    
    return dHRdt

#def deparametriser(f, HR0, HRrest, lamda, age, sex):
    """converts a function with many parameters into one with just two
    required to make HRdot compatible with RK45 solver"""
      
    #y(HR, t) = f(HR, HR0, HRrest, lamda, age, sex, t)
    
    #return y

def HRmodel(lamda, L0, hr0, model_time_vec, HRrest, age, sex, dt=0.1): # should it do it at specific lamdas?
    """Solves the non-linear ODE model of heart-rate kinetics, returning a vector"""
    
    # original method - Euler's method of numerical integration
    # the model should provide values for all times where the HR has data
    """# set recovery start time to when the heart rate peaks
    #t_recovery_begins = timesforHRData[argmax(HRdata)]"""
    model = np.zeros(len(model_time_vec)) # multiplier is the number of HRm samples for each HR data sample

    model[0] = hr0

    """ Original way I tried to fill out the model:
    model[1:len(model)] = np.array([model[i-1] 
                            + HRdot(model[i-1], model[0], lamda, age, sex, dt*i) * dt
                                for i in range(1, len(model))])"""
    
    for i in range(1, len(model)):
        #print("Working in HRmodel to understand why I'm getting 'ValueError: setting an array element with a sequence.'")
        #print(HRdot(model[i-1], model[0], HRrest, lamda, L0, age, sex, dt*i))
            
        model[i] = model[i-1] + HRdot(model[i-1], model[0], HRrest, lamda, L0, age, sex, dt*i) * dt # is it a subtraction?
        

    return model

# DATA IMPORTING AND CLEANING
def obtain(filepath):
    """Extracts the raw HR, SpO2 and Motion data from the CSV file specified, along with the specific time from the start"""
    
    with open(filepath, newline='') as csvfile: # may need to use string formatting with r's?
        reader = csv.DictReader(csvfile)
    
        # create a raw data array [Pulse Rate, SpO2, Motion]
        rawData = np.array([(count*4, int(row['Pulse Rate']), int(row['SpO2']), int(row['Motion'])) for count, row in enumerate(reader)])
                            #time
    return rawData # [Time, Pulse Rate, SpO2, Motion]


def restCleaner(rawData):
    """ Cleans original O2Ring CSV rest data into numpy array. Does not start count at peak HR unlike recovery cleaner."""
    # eliminate raw data which is impossible or happened before the rest period began
    # for item in rawData, eliminate if:
        # motion (item[3])<100, and will be for the next ten readings after?
        # 50 < HR (item[1] < 200
        # (70 <?) SpO2 (item[2]) < 100
    
    # create cleaned data array
    cleanedData = np.array([row for row in rawData if all((row[1]>=42, row[1]<200, row[2]<=100, row[3]<80))])
    
    # subtract the first time value from all the time values so time[0] = 0
    cleanedData[:,0] -= cleanedData[0,0]
    
    # if there's missing data the final time value in the data chunk won't be 4 x the final index
    if cleanedData[-1, 0] != 4*(len(cleanedData)-1):
        #raise Warning('Heart rate/SpO2 data during recording has been lost.')
        print('Warning: Heart rate/SpO2 data during recording has been lost.')
        print('Raw Data:', rawData)
        print("cleanedData:", cleanedData)
    
    return cleanedData # [Time, Pulse Rate, SpO2, Motion]


def recoveryCleaner(rawData):
    """ Cleans original O2Ring CSV recovery data into numpy array. Starts data at peak HR."""
    # eliminate raw data which is impossible or happened before the rest period began
    # for item in rawData, eliminate if:
        # motion (item[3])<100, and will be for the next ten readings after?
        # 50 < HR (item[1] < 200
        # (70 <?) SpO2 (item[2]) < 100
    
    
    """ Not sure how important this is, but:
    # if the first time the motion valid the others are not, raise an error
    for count, item in enumerate(rawData): # I wonder if there's a quicker way to do this?
        if item[3]<100 and any((item[1]<42, item[1]>200, item[2]>100)): # need to add the condition that it must be before the other rows
            print('DataError: Heart rate and SpO2 data are invalid when participant is stationary.')
            print("Raw Data, row ", count, ":", item)
            break # don't bother looping through all the values"""
        
    # create cleaned data array
    # orginal way, but it included data from the rest interval from hiit... 
    # wait i shouldn't need to worry about that if I use impulse test data!
    cleanedData = np.array([row for row in rawData if all((row[1]>=42, row[1]<200, row[2]<=100, row[3]<80))])
    
    # eliminate impossible heart rate (row[1]) and SpO2 (row[2]) readings
    #cleanedData = np.array([row for row in rawData if all((row[1]>=42, row[1]<200, row[2]<=100)])

    # eliminate the inclusion of HIIT rest interval data
    # change cleanedData so that it starts from the point where the motion is below 70 for all points following it
    #cleanedData = np.array
    
    # set it so the peak heart-rate is the moment when t = 0, the first reading
    j = np.argmax(cleanedData[:,1]) # j = point at first HRpeak
    
    cleanedData = cleanedData[j:-1] # remove any data from before the peak heart rate was reached
    
    # subtract the first time value from all the time values so time[0] = 0
    cleanedData[:,0] -= cleanedData[0,0]
    
    # if there's missing data the final time value in the data chunk won't be 4 x the final index
    if cleanedData[-1, 0] != 4*(len(cleanedData)-1):
        #raise Warning('Heart rate/SpO2 data during recording has been lost.')
        print('Warning: Heart rate/SpO2 data during recording has been lost.')
        print('Raw Data:', rawData)
        #print("cleanedData:", cleanedData)
    
    return cleanedData # [Time, Pulse Rate, SpO2, Motion]



def lamdaVector(startLamda, endLamda, n):
    
    if (startLamda or endLamda) > 1 or (startLamda or endLamda) < 0:
        raise Exception('ArgumentError: Enter lamda within range 0 to 1') #raise error that lamdas are outside the range
        
    return np.linspace(startLamda, endLamda, n)



def localMinFinder(y):
    """Finds the indices of the local minima of a 1D (numpy) array, returning 
    a list. Cannot find minima for edges"""
    
    indices = []

    for i in range(1,(len(y)-1)):
    
        dy1 = y[i] - y[i-1]
        dy2 = y[i+1] - y[i]
        
        # dl is always positive so doesn't need including in calculation
        
        # check if there's a sign change of gradient either side, if so it's a 
        # turning point. It could also be one if (..?)
        if np.sign(dy1) != np.sign(dy2) or (dy1 == 0 and dy2 == 0):
        
            # check if it's a local minimum. d^2(f)/d(l)^2
            if dy2 - dy1 > 0:
                indices.append(i)
            
    return indices


"""import matplotlib.pyplot as plt
from scipy import ndimage

def my_legend(axis = None):
    "" Plots legend/text next to the corresponding lines on a figure.
    Credit: Jan Kuiken, 8th June 2013, Stack Overflow""
    if axis == None:
        axis = plt.gca()

    N = 32
    Nlines = len(axis.lines)
    #print Nlines

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # the 'point of presence' matrix
    pop = np.zeros((Nlines, N, N), dtype=np.float)    

    for l in range(Nlines):
        # get xy data and scale it to the NxN squares
        xy = axis.lines[l].get_xydata()
        xy = (xy - [xmin,ymin]) / ([xmax-xmin, ymax-ymin]) * N
        xy = xy.astype(np.int32)
        # mask stuff outside plot        
        mask = (xy[:,0] >= 0) & (xy[:,0] < N) & (xy[:,1] >= 0) & (xy[:,1] < N)
        xy = xy[mask]
        # add to pop
        for p in xy:
            pop[l][tuple(p)] = 1.0

    # find whitespace, nice place for labels
    ws = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0 
    # don't use the borders
    ws[:,0]   = 0
    ws[:,N-1] = 0
    ws[0,:]   = 0  
    ws[N-1,:] = 0  

    # blur the pop's
    for l in range(Nlines):
        pop[l] = ndimage.gaussian_filter(pop[l], sigma=N/5)

    for l in range(Nlines):
        # positive weights for current line, negative weight for others....
        w = -0.3 * np.ones(Nlines, dtype=np.float)
        w[l] = 0.5

        # calculate a field         
        p = ws + np.sum(w[:, np.newaxis, np.newaxis] * pop, axis=0)
        plt.figure()
        plt.imshow(p, interpolation='nearest')
        plt.title(axis.lines[l].get_label())

        pos = np.argmax(p)  # note, argmax flattens the array first 
        best_x, best_y =  (pos / N, pos % N) 
        x = xmin + (xmax-xmin) * best_x / N       
        y = ymin + (ymax-ymin) * best_y / N       
        
        # Correct for the fact this is plotting in a way that needs to be 
        # rotated 90degrees anticlockwise
        z = y
        y = x
        x = z
        
        axis.text(x, y, axis.lines[l].get_label(), 
                  horizontalalignment='center',
                  verticalalignment='center')
    #plt.close('all')
"""
