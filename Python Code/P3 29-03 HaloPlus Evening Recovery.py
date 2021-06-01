# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:55:29 2021

Plot model vs time series Data for an Impulse test.

@author: willi
"""

import numpy as np
import lamdaFittingFunctions as ff
import statistics as s
import time
import matplotlib.pyplot as plt
#from label_line import *

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
test_date = '29-03-21'
time_of_day = 'Evening'
test_time = '19-57'
mask_config = 'HaloPlus'

unmasked_rest_file = r'C:/Users/Willi/OneDrive - University of Cambridge/Engineering/IIB/IIB Project/Experimental Data/Participant 3/HaloPlus 03-29/Evening/O2Ring_20210329193159.csv'
#masked_rest_file = r'C:/Users/Willi/OneDrive - University of Cambridge/Engineering/IIB/IIB Project/Experimental Data/Participant 3/BM001/Evening/O2Ring_20210318205654.csv'
recovery_file = r'C:/Users/Willi/OneDrive - University of Cambridge/Engineering/IIB/IIB Project/Experimental Data/Participant 3/HaloPlus 03-29/Evening/O2Ring_20210329195708.csv'

# wide lamda sweep
lamda1 = 0.01
lamda2 = 0.99
n = 1000

L0 = 11.


# NO MASK REST PERIOD
# import rest data for no mask
rawRestData = ff.obtain(unmasked_rest_file)

# clean rest data
cleanedRestData = ff.restCleaner(rawRestData)
restHRData, restSpO2Data = cleanedRestData[:, 1] , cleanedRestData[:, 2]
#print('Rest HR Data', restHRData)

HRrest = min(restHRData) # lowest observed heart rate during the rest period beforehand
print('HRrest:', HRrest)
#SpO2min = min(restSpO2Data) # necessary?
#print('Mean SpO2 during unmasked rest period:', s.mean(restSpO2Data),'%')


#HRu = HRrest + 2 # upper bound for the resting heart rate with the uncertainty of the O2Ring

"""
if time_of_day == 'Evening' and mask_config != 'No Mask':
    # MASKED REST PERIOD
    # import rest data whilst wearing the mask
    rawMaskedRestData = ff.obtain(masked_rest_file)
    
    # clean rest data
    cleanedMaskedRestData = ff.restCleaner(rawMaskedRestData)
    maskedRestHRData, maskedRestSpO2Data = cleanedMaskedRestData[:, 1], cleanedMaskedRestData[:, 2]
    HRrest_masked = min(maskedRestHRData) # lowest observed heart rate during the rest period beforehand
    
    # data analysis between rest periods - compare SpO2 and HR populations with t-tests
    # MISSING
    #state n
    meanRestSpO2 = s.mean(maskedRestSpO2Data)
    print('Mean SpO2 during masked rest period:', meanRestSpO2,'%')
"""

# IMPULSE TEST DATA
# import raw test data
rawRecData = ff.obtain(recovery_file)
#rawRecData = ff.obtain(r'C:\Users\willi\Google Drive\IIB Project Experimental Data\Will\BM001\Evening\p108 will Evening hiit with bm001')
cleanedRecData = ff.recoveryCleaner(rawRecData) # clean the raw data for the recovery period
#print('cleanedRecData:', cleanedRecData)

# data cannot be read during exercise for reason Wellue stated
timesRecovery, HRrecovery, SpO2recovery = cleanedRecData[:, 0], cleanedRecData[:, 1], cleanedRecData[:, 2]


#analyse the SpO2 data
print('number of samples:', len(SpO2recovery))

meanRecoverySpO2 = np.mean(SpO2recovery)
print('Mean SpO2 during recovery:', meanRecoverySpO2,'%')

sdRecSpO2 = np.std(SpO2recovery)
print('$s_{SpO_2}$ during recovery:', sdRecSpO2)

SpO2min = min(SpO2recovery)
print('Minimum SpO2 observed during recovery:', SpO2min,'%')

#record the number of times where SpO2 goes below 90%
if SpO2min < 90:
    hypoxemiaVector = np.array([[timesRecovery[i], SpO2recovery[i]] for i in range(len(SpO2recovery)) if SpO2recovery[i] < 90]) # is SpO2 in % or not here?

    print(hypoxemiaVector)
    # if Hypoxia occurs plot a graph?


#should I create an SpO2 graph of all tests with masks, on all people, superposed to show that SpO2 never goes below a certain value?

"""
# define the domain of lamdas being considered
if sex == 'male':
    lamda1 = 35/HRu # lower bound for lamda, if male
    lamda2 = 35/40 # upper bound for male lamda

elif sex == 'female':
    lamda1 = 35/(HRu-5) # lower bound for lamda, if female
    lamda2 = 35/45 # upper bound for female lamda
else:
    raise Exception('ArgumentError: Model only works for male or female participants.')
"""


#print('HRu',HRu)
print('lamda1', lamda1)
print('lamda2',lamda2)

# generate list of n lamdas which costs will be evaluated at.
lamdas = ff.lamdaVector(lamda1, lamda2, n) 
#L0s = np.linspace(L01,L02, n) # generate a list of L(0)'s which will be optimised across

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
    print('np.array(np.where(costs == costs.min())):', np.array(np.where(costs == costs.min())))
    globalMinimumIndex = np.array(np.where(costs == costs.min()))[0][0]
# how can I generalise this if there are multiple global minima?
# difficult since atm it outputs something which isn't a minimum as its second item


optimalLamdas = np.array([lamdas[index] for index in optimalIndices])

lamda_opt = lamdas[globalMinimumIndex] # the lamda's which give global minima

#print ('Optimal Lamdas:', optimalLamdas)
print('Observed resting heart rate:', HRrest, 'bpm')
"""predictedHRrests = [ff.HRmin(optimalLamdas[i], sex) for i in range(len(optimalLamdas))]
print('Predicted Resting Heart Rates:', predictedHRrests, 'bpm')"""
"""
# choose the lamda which is closest to the one predicted by the observed resting heart rate to ensure robustness.
j = np.argmin(np.array([abs(item - HRrest) for item in predictedHRrests])) # if there are two equivalently far predictions from HRrest this goes for the lower of the two

optimalLamda = optimalLamdas[j]
print('Optimal Lamda Selected:', optimalLamda)
"""
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

# HR TIME SERIES PLOTS
# plot HR time graph for optimal lamdas to verify them
# obtain model's HRm values and create a time vector for it

time_series, ax2 = plt.subplots(figsize=(6,4))

dt = 0.1
timeVectorforModel = np.arange(0, timesRecovery[-1]+dt, dt)

#startTime2 = time.time()

#plot the range of models tested
"""m = 5 # plot 5 of the models, print one
w = n/(m-1) # sample a model every w models
sampledLamdas = [lamdas[int(w*i)] for i in range(m-1)]
sampledLamdas.append(lamda2)

for lamda in sampledLamdas:
    
    HRm = ff.HRmodel(lamda, L0, HRrecovery, timesRecovery, HRrest, age, sex, dt)
    
    darkness = 1-((lamda-lamda1)/(lamda2-lamda1)*.6+.2)
    
    # only include labels for first and final lamdas
    if lamda == lamda1:
        ax2.plot(timeVectorforModel, HRm, 
                color='{}'.format(darkness),
                label=r'$HR_m(\lambda_{} = {:0.2f}$)'.format('\min', lamda))
    if lamda == sampledLamdas[-1]:
        ax2.plot(timeVectorforModel, HRm, 
            color='{}'.format(darkness),
            label=r'$HR_m(\lambda_{} = {:0.2f}$)'.format('\max', lamda))
    else:
        ax2.plot(timeVectorforModel, HRm, 
                color='{}'.format(darkness))"""

#endTime2 = time.time()
#print("Model Plotting time Elapsed", endTime2-startTime2)

#startTime3 = time.time()
 # plot the model at the optimal lamdas (local minima from the cost function)

"""for optimalLamda in optimalLamdas:
    
    optimalHRmodel = ff.HRmodel(optimalLamda, L0, HRrecovery, timesRecovery, 
                                HRrest, age, sex, dt)
    #ax2.plot(timeVectorforModel, optimalHRmodel, 
    #label='Lamda = {:0.2f}'.format(optimalLamda), color='black', 
    #linestyle='-')
    
    # plot the global minimum lamda differently
    if optimalLamda == lamda_opt:
        ax2.plot(timeVectorforModel, optimalHRmodel, color='orange', 
                label=r'$HR_m(\hat \lambda = {:0.2f})$'.format(optimalLamda), 
                lw=3)
    else:
        ax2.plot(timeVectorforModel, optimalHRmodel, 
                label=r'$HR_m(\lambda = {:0.2f})$'.format(optimalLamda))
"""

# plot SpO2 during recovery
# create a secondary axis
axSpO2 = ax2.twinx()
# add SpO2's error bars
axSpO2.errorbar(timesRecovery, SpO2recovery, yerr=0.02*SpO2recovery, 
                c='b', elinewidth=0.5, capsize=0.5, marker=None)

#plot SpO2
SpO2_plot, = axSpO2.plot(timesRecovery, SpO2recovery, color='blue', 
                         label=r'$\overline{} = {:0.1f}, s_{} = {:0.3f}$'.format('{SpO_2}', meanRecoverySpO2, '{SpO_2}', sdRecSpO2))


axSpO2.set_ylim(70,100)
axSpO2.set_ylabel(r'$SpO_2$ / %')

# make SpO2 axis the same colour as its plot?
#axSpO2.yaxis.set_color(SpO2_plot.get_color())
#axSpO2.set_color(SpO2_plot.get_color())

# plot the optimal Lamda, assuming there's only one global minimum
optimalHRmodel = ff.HRmodel(lamda_opt, L0, HRrecovery, timesRecovery, 
                                HRrest, age, sex, dt)

hrmodel_plot, = ax2.plot(timeVectorforModel, optimalHRmodel, color='orange', 
                label=r'$HR_m(\hat \lambda = {:0.3f})$'.format(lamda_opt))


# plot the empirical data
hr_data_plot, = ax2.plot(timesRecovery, HRrecovery, 'k', label='$HR_{Data}$') 
# html for scarlet: color='#aa0000'

# plot the error bars of HR data
ax2.errorbar(timesRecovery, HRrecovery, yerr=0.02*HRrecovery, elinewidth=0.5, 
             c='k', capsize=0.5, marker=None)

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

p = [hr_data_plot, hrmodel_plot, SpO2_plot]
plt.legend(p, [p_.get_label() for p_ in p],
           loc='center right')
#plt.legend()
"""# In case I want to plot the participant name on it
plt.title('Participant {} - {} {} Recovery \n[{} {}]'.format(ptcpt_no, 
                                                              time_of_day, 
                                                              mask_config, 
                                                              test_date, 
                                                              test_time))"""
    
plt.title('{} {} Recovery [{} {}]'.format(time_of_day, mask_config, test_date, test_time))

#ax2.text(1, 1,'{} {}'.format())
plt.show()
time_series.savefig('..\Time Series Plots\P{} {} {} {} {} Impulse Recovery Time Series.png'.format(ptcpt_no, test_date, test_time, time_of_day, mask_config))

# SEE WHAT THE LAMDA IS IF YOU CLIP THE HEART RATE DATA TO A LIMITED DURATION
"""#L0 = 6.5

clippedHRdata = plt.figure()
# clip HR recovery data to only 0 to 400s
clipTo = 400
HRrecovery = HRrecovery[:int(clipTo/4)]
timesRecovery = timesRecovery[:int(clipTo/4)]

lamdas = ff.lamdaVector(lamda1, lamda2, n) # generate list of n lamdas which costs will be evaluated at.
costs = ff.costVector(lamdas, HRrecovery, timesRecovery, HRrest, L0, age, sex)

# plot cost function against lamda
# where L0 is fixed

costGraph = plt.figure()
plt.plot(lamdas, costs)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'f($\lambda$)')
plt.show()

# find the optimal lamda, using a function made myself
optimalIndices = ff.localMinFinder(costs)
print(optimalIndices)

optimalLamdas = np.array([lamdas[index] for index in optimalIndices])

print ('Optimal Lamdas:', optimalLamdas)
print('Observed resting heart rate:', HRrest, 'bpm')
predictedHRrests = [ff.HRmin(optimalLamdas[i], sex) for i in range(len(optimalLamdas))]
print('Predicted Resting Heart Rates:', predictedHRrests, 'bpm')

# choose the lamda which is closest to the one predicted by the observed resting heart rate to ensure robustness.
j = np.argmin(np.array([abs(item - HRrest) for item in predictedHRrests])) # if there are two equivalently far predictions from HRrest this goes for the lower of the two

optimalLamda = optimalLamdas[j]
print('Optimal Lamda Selected:', optimalLamda)

# plot HR time graph for optimal lamdas to verify them
# obtain model's HRm values and create a time vector for it
dt = 0.1
optimalHRmodel = ff.HRmodel(optimalLamda, L0, HRrecovery, timesRecovery, HRrest, age, sex, dt)


lbyEye = .08
HRmodelbyEye = ff.HRmodel(lbyEye, L0, HRrecovery, timesRecovery, HRrest, age, sex, dt)

timeVectorforModel = np.arange(0, len(optimalHRmodel)*dt, dt)

print('Length of optimalHRmodel:', len(optimalHRmodel))
print('Length of timeVectorforModel:', len(timeVectorforModel))
print('Length of HRrecovery:', len(HRrecovery))
print('Length of timesRecovery:', len(timesRecovery))

#round lamdas, not needed
#l = np.round(np.array(optimalLamdas), 3)

HRgraph = plt.figure()
# plot the empirical data
plt.plot(timesRecovery, HRrecovery, label='$HR Data$')
# plot the model at the optimal lamda
plt.plot(timeVectorforModel, optimalHRmodel, label='HR Model at lamda = {:0.2f}'.format(optimalLamda))
plt.plot(timeVectorforModel, HRmodelbyEye, label='HR Model at lamda = {:0.2f}'.format(lbyEye))

plt.xlabel('$t$ / s')
plt.ylabel('$HR$ / bpm')
plt.title('HR Time Series: Model vs Empirical Data')
plt.legend()
plt.show()"""

print("Optimal Lamda:", lamda_opt)

endTime = time.time()

print("Total Time Elapsed:", endTime - startTime)