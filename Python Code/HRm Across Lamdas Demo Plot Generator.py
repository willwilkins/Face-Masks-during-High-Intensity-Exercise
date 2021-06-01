# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:55:29 2021

Simulates HR recovery from exercise to demonstrate how HRm varies with lamda for P3.


@author: willi
"""

import numpy as np
import hrmPlotterFunctions as ff
import time
import matplotlib.pyplot as plt
from label_line import *


stop_time = 600 # time length of test in s
n = 201
L0 = 11.

startTime = time.time()

# set plotting style
plt.style.use(['science','no-latex','ieee'])
#plt.style.use(['science', 'ieee'])
#plt.style.use('default')
plt.rcParams.update({
    "font.family": "sans-serif",   # specify font family here
    "font.serif": ["Helvetica"],  # specify font here
    "font.size":8})          # specify font size here


# participant data
participants = ({'age':55, 'sex':'male'}, # participant 1, Dad
                {'age':54, 'sex':'female'}, # participant 2, Mum
                {'age':22, 'sex':'male'}) # participant 3, Will

# participant 1
ptcpt_no = 3
age = participants[ptcpt_no-1]['age']
sex = participants[ptcpt_no-1]['sex']

# data taken from the evening of the no mask test
HRrest = 54
HR0 = 157

# generate list of n lamdas which costs will be evaluated at.
lamda1 = 0.
lamda2 = .4
lamdas = ff.lamdaVector(lamda1, lamda2, n)


# make calculation last for 10,000s
dt = .1 # timesteps of 0.1 s
timeVectorforModel = np.arange(0, stop_time, dt)


# HR TIME SERIES SIMULATION PLOTS
# plot HR time graph for optimal lamdas to verify them
# obtain model's HRm values and create a time vector for it

startTime2 = time.time()

#plot the models tested on separate plots

# for each lamda
for count, lamda in enumerate(lamdas):
    
    # generate figure and axes
    fig, ax = plt.subplots()
    
    # generate the HR model
    HRm = ff.HRmodel(lamda, L0, HR0, timeVectorforModel, HRrest, age, sex, dt)
    
    #darkness = 1-((lamda-lamda1)/(lamda2-lamda1)*.6+.2)
 
    ax.plot(timeVectorforModel, HRm, label=r'$\lambda = {:0.3f}$'.format(lamda))
    
    # plot labels inline
    labelLines(plt.gca().get_lines(),zorder=2.5)

    ax.set_xlabel(r'$t$ / s')
    ax.set_ylabel(r'$HR$ / bpm')
    
    # set y lims so they don't change across lamdas
    ax.set_ylim(HRrest, 200)
    
    ax.set_title('HR Model Simulation for Participant 3 as $\lambda$ Varies'
                 +'\n'
                 +'$HR(0) = {} $bpm, $L(0) = {}$ mM,'.format(HR0, L0)
                 +'\n'
                 +'$HR_{} = {} $bpm, $Age = {}$, $Sex = {}$'.format('{rest}', HRrest, age, sex))
    plt.ioff()
    plt.close()

    # save the figure
    fig.savefig('..\HRm Plots lamda 0-0.4\Fig_{} HRm(lamda={}).png'.format(count, lamda))

    print('Model {}/{} plotted at:'.format(count+1, len(lamdas)), 
                                                  time.time()-startTime2, "s")
    

endTime2 = time.time()
print("Models Finished plotting at:", endTime2-startTime2, "s")


#endTime3 = time.time()
#print("Optimal Model Plotting time Elapsed", endTime3-startTime3)


# Plot HR time graph for the other optimal lamdas (out of interest)
#plt.plot(timeVectorforModel, ff.HRmodel(optimalLamdas[0], HRrecovery, 
#                                        timesRecovery, HRrest, age, sex, dt), 
#                                            label='HR Model at lamda = {:0.2f}'.format(optimalLamdas[0]))

#ax2.legend(bbox_to_anchor = (1.28, .78)) # plots legend off the axes

endTime = time.time()

print("Total Time Elapsed:", endTime - startTime)