# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:55:29 2021

Simulates HR recovery from exercise to demonstrate phenomena found analytically.

@author: willi
"""

import numpy as np
import longTimeModelFunctions as ff
import time
import matplotlib.pyplot as plt
#from label_line import *


stop_time = 8000 # time length of test in s
n = 5
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


# wide lamda sweep
lamda1 = 0.01
lamda2 = 0.99

HRrest = 54


# IMPULSE TEST DATA
# import raw test data
rawRecData = ff.obtain(recovery_file)
#rawRecData = ff.obtain(r'C:\Users\willi\Google Drive\IIB Project Experimental Data\Will\BM001\Evening\p108 will Evening hiit with bm001')
cleanedRecData = ff.recoveryCleaner(rawRecData) # clean the raw data for the recovery period
#print('cleanedRecData:', cleanedRecData)

# data cannot be read during exercise for reason Wellue stated
timesRecovery, HRrecovery = cleanedRecData[:, 0], cleanedRecData[:, 1]


dt = 0.1 # timesteps of 0.1 s
# make calculation last for 10,000s
timeVectorforModel = np.arange(0, stop_time, dt)

# generate list of n lamdas which costs will be evaluated at.
lamdas = ff.lamdaVector(lamda1, lamda2, n) 

lamda_opt = 0.198348348 # lamda which gave a global minima for this test at L0 = 11.0


# HR TIME SERIES PLOTS
# plot HR time graph for optimal lamdas to verify them
# obtain model's HRm values and create a time vector for it

time_series, ax2 = plt.subplots(figsize=(6,4))

# plot the empirical data
hr_data_plot, = ax2.plot(timesRecovery, HRrecovery, c='#ac0000', label='$HR_{Data}$') 


startTime2 = time.time()

#plot the range of models tested
m = 5 # plot 5 of the models
w = n/(m-1) # sample a model every w models
sampledLamdas = [lamdas[int(w*i)] for i in range(m-1)]
sampledLamdas.append(lamda2)

for count, lamda in enumerate(sampledLamdas):
    
    HRm = ff.HRmodel(lamda, L0, HRrecovery, timeVectorforModel, HRrest, age, sex, dt)
    
    darkness = 1-((lamda-lamda1)/(lamda2-lamda1)*.6+.2)
    
 
    ax2.plot(timeVectorforModel, HRm, 
                color='{}'.format(darkness),
                label=r'$HR_m(\lambda = {:0.2f}$)'.format(lamda))
    
    print('Model {} printed at:'.format(count), time.time()-startTime2, "s")

endTime2 = time.time()
print("Models Finished plotting at:", endTime2-startTime2, "s")

# plot the optimal Lamda, assuming there's only one global minimum
optimalHRmodel = ff.HRmodel(lamda_opt, L0, HRrecovery, timeVectorforModel,
                                HRrest, age, sex, dt)

hrmodel_plot, = ax2.plot(timeVectorforModel, optimalHRmodel, color='orange', 
                label=r'$HR_m(\hat \lambda = {:0.3f})$'.format(lamda_opt))


#endTime3 = time.time()
#print("Optimal Model Plotting time Elapsed", endTime3-startTime3)


# Plot HR time graph for the other optimal lamdas (out of interest)
#plt.plot(timeVectorforModel, ff.HRmodel(optimalLamdas[0], HRrecovery, 
#                                        timesRecovery, HRrest, age, sex, dt), 
#                                            label='HR Model at lamda = {:0.2f}'.format(optimalLamdas[0]))

#ax2.legend(bbox_to_anchor = (1.28, .78)) # plots legend off the axes
#labelLines(plt.gca().get_lines(),zorder=2.5)


ax2.set_xlabel(r'$t$ / s')
ax2.set_ylabel(r'$HR$ / bpm')

plt.text(5900, 72, '$HR_{} = {}$ bpm'.format('{rest}',HRrest)
                     +'\n'
                     +'$\delta t = {}$ s'.format(dt))

plt.legend()
                                     
plt.title('HR Model over a Long Time: \n P{} {} {} Recovery [{} {}]'.format(ptcpt_no, time_of_day, mask_config, test_date, test_time))

#ax2.text(1, 1,'{} {}'.format())
plt.show()
time_series.savefig('..\Time Series Plots\{}.png'.format(lamda))

endTime = time.time()

print("Total Time Elapsed:", endTime - startTime)