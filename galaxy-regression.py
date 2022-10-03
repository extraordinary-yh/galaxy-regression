#!/usr/bin/env python
# coding: utf-8

#Import packages
import pandas as pd #pandas is a very helpful package for data analysis that uses a DataFrame structure
pd.set_option('max_rows', 10) #this keeps pandas from displaying ALL the data
import matplotlib.pyplot as plt #importing matplotlib package
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from scipy import stats
from scipy.stats import linregress
import statsmodels.api as statsmodels
import seaborn as sns

#import data from simplified .csv file (no header)
filename = 'NED.csv'
data = pd.read_csv(filename)


'''This code cell is for data processing'''
distance = data['D'] #extract distance data
velocity = data['Vgsr (RC3)'] #extract velocity data

data #print the dataset


'''define a function to print out descriptivestats
This function will take two parameters:
"list" is the data to analyze, 
"name" is the name of this dataset
This function will print out the dataset's name, size, mean, median, mode, and standard deviation
'''
def CalculateandPrintDescriptiveStats(list, name):
    print('Name:', name)
    print('Count:',len(list))
    print('Mean:',np.mean(list))
    print('Median:', np.median(list))
    print('Mode:', stats.mode(list))
    print('Standard Deviation:',np.std(list,ddof=1),"\n")

CalculateandPrintDescriptiveStats(distance, "Distance")
CalculateandPrintDescriptiveStats(velocity, "Velocity")



#create a scatter plot of distacne and velocity
plt.scatter(distance, velocity , c = 'darkorange', marker = '*')

# calculates slope, intercept, r value, p value and standard error 
b1, b0, r, p, stdErr = linregress(distance, velocity)

#draw line of best fit
plt.plot(distance, b1*distance+b0)

#draw equation and r value on the plot
lineOBF = 'y = ' + str(round(b1,3)) + 'x' + ' + ' + str(round(b0,3))
rValue = 'r = ' + str(round(r,3))
plt.text(14, 1, lineOBF,fontsize=15)
plt.text(23, 250, rValue,fontsize=15)

#define the graph's title, x-axis label and y-axis label
plt.title("Scatter Plot of Galaxie's Recessional Velocity and Distance")
plt.xlabel("Distance in megaparsecs from Earth(Mpc)")
plt.ylabel("Recessional Velocity(km/s)")

#draw the graph
plt.show()



''' this function will construct a single linear regression model 
    with two predictor variables by using built in library functions. 
    It will output two plots to assess the validity of the model.'''
def regression(x,y):

    # define predictors X and response Y:
    X = data[x]
    X = statsmodels.add_constant(X)
    Y = data[y]
    
    # construct model:
    global regressionmodel 
    regressionmodel = statsmodels.OLS(Y,X).fit()

    # residual plot:
    plt.figure()
    resPlot = sns.residplot(x=regressionmodel.predict(), y=regressionmodel.resid, color='mediumseagreen')
    resPlot.set(xlabel='Fitted values for '+ y, ylabel='Residuals')
    resPlot.set_title('Residuals vs Fitted values',fontweight='bold')
    
    # QQ plot:
    qqPlot = statsmodels.qqplot(regressionmodel.resid,fit=True,line='45')
    qqPlot.suptitle("Normal Probability (\"QQ\") Plot for Residuals",fontweight='bold')


regression(['D'],'Vgsr (RC3)')
regressionmodel.summary()

