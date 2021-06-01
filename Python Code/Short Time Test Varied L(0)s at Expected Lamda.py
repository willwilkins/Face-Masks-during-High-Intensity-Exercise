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


n = 7
HRrest = 54

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
L0Min = 2
L0Max = 14.
"""
# NO MASK REST PERIOD
# import rest data for no mask
rawRestData = ff.obtain(unmasked_rest_file)

# clean rest data
cleanedRestData = ff.restCleaner(rawRestData)
restHRData = cleanedRestData[:, 1]

HRrest = min(restHRData) # lowest observed heart rate during the rest period beforehand
print('HRrest:', HRrest)"""


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

# generate list of n lamdas which costs will be evaluated at.
L0s = np.linspace(L0Min, L0Max, n) 

lamda = 35/54 # lamda which gave a global minima for this test at L0 = 11.0
# try this again at lamda = 35/54

# create new time vector for the model with the correct timestep
dt = 1
stop_time = timesRecovery[-1] + 350 # time length of model in s, +260 to give space for legend
timeVectorforModel = np.arange(0, stop_time, dt)
"""
# FIND THE OPTIMAL L0 - trivial for lamda = 0.198
time1 = time.time()
costs = ff.costVector(lamdas, HRrecovery, timesRecovery, HRrest, L0, age, sex)
# would this work with the f function from before?
#costs = ff.costMatrix(lamdas, L0s, HRrecovery, timesRecovery, HRrest, age, sex)
time2 = time.time()
costCalcTime = time2-time1
print('Cost calculation time:', costCalcTime, 's')

# find the optimal lamda, using a function made myself
optimalIndices = ff.localMinFinder(costs)
print(costs.min())

# array of indices which should give the minima of the costs array
m = np.array(np.where(costs == costs.min()))[:,0]

if len(m)>1:
    minima_costs = [costs[i] for i in m]
    
    # if there really are multiple global minima
    for i, value in enumerate(minima_costs):
        if value == any(minima_costs.pop(i)):
            raise OverflowError("There exist multiple global minima.")
        
    else: # for whatever reason it's showing work out which is the real minimum
        globalMinimumIndex = m[np.argmin(minima_costs)]
else:
    globalMinimumIndex = np.array(np.where(costs == costs.min()))[0]
# how can I generalise this if there are multiple global minima?
# difficult since atm it outputs something which isn't a minimum as its second item


optimalLamdas = np.array([lamdas[index] for index in optimalIndices])

lamda_opt = lamdas[globalMinimumIndex] # the lamda's which give global minima

#print ('Optimal Lamdas:', optimalLamdas)
print('Observed resting heart rate:', HRrest, 'bpm')

# COST PLOT
# plot cost function against lamda
# where L0 is fixed

costGraph, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(lamdas, costs)
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel(r'f($\lambda$)')
ax1.set_title('{} {} Recovery [{} {}]'.format(time_of_day, mask_config, 
                                              test_date, test_time))

ax1.scatter(lamda_opt, costs.min(), s=80, c='r', marker='o', 
            label='Global Minimum at $\hat \lambda = {:0.3f} ± 0.001$'.format(lamda_opt))
ax1.legend()
#ax1.annotate(text, (0.75, .75), xycoords='figure fraction')
plt.show()
# save the figure as an svg and a png
#costGraph.savefig('..\Cost Functions\P{} {} {} Lamda Fitting Cost Function.svg'.format(ptcpt_no, time_of_day, mask_config))
costGraph.savefig('..\Cost Functions\P{} {} {} {} {} Impulse Recovery Lamda Fitting Cost Function.png'.format(ptcpt_no, test_date, test_time, time_of_day, mask_config))

"""



# HR TIME SERIES PLOTS
# plot HR time graph for optimal lamdas to verify them
# obtain model's HRm values and create a time vector for it

fig, ax = plt.subplots(figsize=(6,4))

# plot the empirical data
hr_data_plot, = ax.plot(timesRecovery, HRrecovery, c='#ac0000',
                        label='$HR_{}$ from P{} \n{} {} \nRecovery Test \n[{} {}]'
                        .format('{Data}', ptcpt_no, time_of_day, mask_config, 
                        test_date, test_time))

startTime2 = time.time()

#plot the range of models tested
m = n # plot 5 of the models
w = n/(m-1) # sample a model every w models
sampled_L0s = [L0s[int(w*i)] for i in range(m-1)]
sampled_L0s.append(L0Max)

for count, L0 in enumerate(sampled_L0s):
    

    HRm = ff.HRmodel(lamda, L0, HRrecovery, timeVectorforModel, HRrest, age, sex, dt)
    
    darkness = 1-((L0-L0Min)/(L0Max-L0Min)*.6+.2)
    
    if count%2 == 0: # if count is even, plot label
        ax.plot(timeVectorforModel, HRm, 
                    color='{}'.format(darkness),
                    label=r'$HR_{}(L(0) = {})$'.format('{m}', L0))
    else: # don't plot label
        ax.plot(timeVectorforModel, HRm, 
                        color='{}'.format(darkness))
                        
    """# only include labels for first and final lamdas
    if L0 == L0Min or L0 == L0Max:
        ax.plot(timeVectorforModel, HRm, 
                color='{}'.format(darkness),
                label=r'$HR_{}(L(0) = {})$'.format('{m}', L0))
    else:
        ax.plot(timeVectorforModel, HRm, 
                color='{}'.format(darkness))"""
    
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

"""txt = plt.text(5900, HRrest+20, '$HR_{} = {}$ bpm'.format('{rest}', HRrest)
                     +'\n'
                     +'$\lambda = \frac{:0.3f}$'.format(lamda)
                     +'\n'
                     +'$\delta t = {}$ s'.format(dt))"""
                     
ax.set_ylim(ymax=170)

#textBox = txt.get_frame()
#textBox.set_color('white')

legend = plt.legend(facecolor='white')
#frame = legend.get_frame()
#frame.set_facecolor('white')
                                     
plt.title('Effect of $L(0)$ on HR Model Over a Short Timescale \n'
          +'$HR_{} = {}$ bpm, '.format('{rest}', HRrest)
          +'$\lambda = \frac{}{}$, '.format('{35}','{}'.format(lamda))
          +'$\delta t = {}$ s'.format(dt))

#ax.text(1, 1,'{} {}'.format())
plt.show()
#fig.savefig('..\Time Series Plots\Over a Long Time\Effect of L(0) on HR Model, dt = {}-{}, HRrest = {}.png'.format(str(L0Min), str(L0Max), HRrest))

endTime = time.time()

print("Total Time Elapsed:", endTime - startTime)