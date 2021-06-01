# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:55:29 2021

Simulates HR recovery from exercise for a long duration of time.

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

#test info
test_date = '12-03-21'
time_of_day = 'Evening'
test_time = '19-58'
mask_config = 'No Mask'

unmasked_rest_file = r'C:/Users/Willi/OneDrive - University of Cambridge/Engineering/IIB/IIB Project/Experimental Data/Participant 3/No Mask 12-03/Evening/12-03 1940 P3 No Mask Rest.xls'
recovery_file = r'C:/Users/Willi/OneDrive - University of Cambridge/Engineering/IIB/IIB Project/Experimental Data/Participant 3/No Mask 12-03/Evening/12-03 1958 P3 No Mask Recovery.xls'

# wide lamda sweep
dtMin = 0.01
dtMax = 10
"""
# NO MASK REST PERIOD
# import rest data for no mask
rawRestData = ff.obtain(unmasked_rest_file)

# clean rest data
cleanedRestData = ff.restCleaner(rawRestData)
restHRData = cleanedRestData[:, 1]

HRrest = min(restHRData) # lowest observed heart rate during the rest period beforehand
print('HRrest:', HRrest)"""

HRrest = 54

# IMPULSE TEST DATA
# import raw test data
rawRecData = ff.obtain(recovery_file)
#rawRecData = ff.obtain(r'C:\Users\willi\Google Drive\IIB Project Experimental Data\Will\BM001\Evening\p108 will Evening hiit with bm001')
cleanedRecData = ff.recoveryCleaner(rawRecData) # clean the raw data for the recovery period
#print('cleanedRecData:', cleanedRecData)

# data cannot be read during exercise for reason Wellue stated
timesRecovery, HRrecovery = cleanedRecData[:, 0], cleanedRecData[:, 1]
print("timesRecovery:", timesRecovery)

print("HRrecovery:", HRrecovery)

stop_time = timesRecovery[-1]

lamda = 35/54 # timesteps of 0.1 s


# generate list of n lamdas which costs will be evaluated at.
timeSteps = np.linspace(dtMin, dtMax, n) 

lamda = 0.198348348 # lamda which gave a global minima for this test at L0 = 11.0


# HR TIME SERIES PLOTS
# plot HR time graph for optimal lamdas to verify them
# obtain model's HRm values and create a time vector for it

fig, ax = plt.subplots(figsize=(6,4))

# plot the empirical data
hr_data_plot, = ax.plot(timesRecovery, HRrecovery, c='#ac0000', label='$HR_{Data}$') 


startTime2 = time.time()

#plot the range of models tested
m = n # plot 5 of the models
w = n/(m-1) # sample a model every w models
sampled_timeSteps = [timeSteps[int(w*i)] for i in range(m-1)]
sampled_timeSteps.append(dtMax)

for count, dt in enumerate(sampled_timeSteps):
    
    # create new time vector for the model with the correct timestep
    timeVectorforModel = np.arange(0, stop_time, dt)
    print('timeVectorforModel{}:'.format(count), timeVectorforModel)
    HRm = ff.HRmodel(lamda, L0, HRrecovery, timeVectorforModel, HRrest, age, sex, dt)
    print('HRm{}:'.format(count), HRm)
    darkness = 1-((dt-dtMin)/(dtMax-dtMin)*.6+.2)
    
 
    ax.plot(timeVectorforModel, HRm, 
                color='{}'.format(darkness),
                label=r'$\delta  = {}$)'.format(dt))
    
    print('Model {} printed at:'.format(count), time.time()-startTime2, "s")

#ax.set_ylim(-10, 220)

endTime2 = time.time()
print("Models Finished plotting at:", endTime2-startTime2, "s")

#endTime3 = time.time()
#print("Optimal Model Plotting time Elapsed", endTime3-startTime3)


# Plot HR time graph for the other optimal lamdas (out of interest)
#plt.plot(timeVectorforModel, ff.HRmodel(optimalLamdas[0], HRrecovery, 
#                                        timesRecovery, HRrest, age, sex, dt), 
#                                            label='HR Model at lamda = {:0.2f}'.format(optimalLamdas[0]))

#ax.legend(bbox_to_anchor = (1.28, .78)) # plots legend off the axes
#labelLines(plt.gca().get_lines(),zorder=2.5)


ax.set_xlabel(r'$t$ / s')
ax.set_ylabel(r'$HR$ / bpm')

plt.text(5900, HRrest+14, '$HR_{} = {}$ bpm'.format('{rest}', HRrest)
                     +'\n'
                     +'$\lambda = {}$ s'.format(lamda))

legend = plt.legend()
frame = legend.get_frame()
frame.set_facecolor('white')
                                     
plt.title('HR Models as Timestep Size Varies: \n (P{} {} {} Recovery [{} {}])'.format(ptcpt_no, time_of_day, mask_config, test_date, test_time))

#ax.text(1, 1,'{} {}'.format())
plt.show()
fig.savefig('..\Time Series Plots\Over a Long Time\Effect of Timestep Size on Model, Short Timescale, dt = {}-{}, HRrest = {}.png'.format(str(dtMin), str(dtMax), HRrest))

endTime = time.time()

print("Total Time Elapsed:", endTime - startTime)