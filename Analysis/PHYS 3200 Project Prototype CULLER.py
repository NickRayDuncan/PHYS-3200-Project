
# coding: utf-8

# In[80]:


# A E S T H E T I C diminishing, dampened sine wave method ATTEMPT #2

# DATA CULLER 1000
print('start')

from matplotlib import pyplot
import numpy as np
from scipy import optimize
graph = np.genfromtxt(r'C:\Users\nickr\Phys3200\biorsch2.csv')

columnchecknumber = int(input('Enter column you would like to check '))
x = graph[:,0]
y = graph[:,columnchecknumber]

def best_fit(x,a,b,c,d,e,f): # Function which is the best expected best fit equation NEEDS CUSTOM PARAMATERS
    return (a*np.e**(-b*(x-c)))*np.cos(d*(x-c))+e*(x-c)+f

# This system cuts the desired graph into 5ths in order to minimize the effects an outlier can have on the entire function


#pyplot.plot(x,best_fit(x,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
#pyplot.plot(x,y,'g*')


# In[81]:


# This system cuts the desired graph into 5ths in order to minimize the effects an outlier can have on the entire function

#1st 5th

x_firstfifth = graph[0:int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0),0]
y_firstfifth = graph[0:int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0),columnchecknumber]

params, params_covariance = optimize.curve_fit(best_fit,x_firstfifth,y_firstfifth,p0=[1500,0.015,graph[int(len(graph[:,0])-1),0],0.2,100,7000],maxfev=1000000)
pyplot.plot(x_firstfifth,best_fit(x_firstfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_firstfifth,y_firstfifth, 'g*')

pyplot.title('Column '+str(columnchecknumber)+' as a Function of Time')
pyplot.xlabel('Time (Hours)')
pyplot.ylabel("Luminence (Topcount Units)")
pyplot.show()


# In[82]:


# This system cuts the desired graph into 5ths in order to minimize the effects an outlier can have on the entire function

#2nd 5th

x_secondfifth = graph[int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0)-2:(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*2,0]
y_secondfifth = graph[int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0)-2:(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*2,columnchecknumber]

params, params_covariance = optimize.curve_fit(best_fit,x_secondfifth,y_secondfifth,p0=[1500,0.015,graph[int(len(graph[:,0])-1),0],0.2,100,7000],maxfev=1000000)
pyplot.plot(x_secondfifth,best_fit(x_secondfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_secondfifth,y_secondfifth, 'g*')

pyplot.title('Column '+str(columnchecknumber)+' as a Function of Time')
pyplot.xlabel('Time (Hours)')
pyplot.ylabel("Luminence (Topcount Units)")
pyplot.show()


# In[83]:


# This system cuts the desired graph into 5ths in order to minimize the effects an outlier can have on the entire function

#3rd 5th

x_thirdfifth = graph[(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*2-2:(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*3,0]
y_thirdfifth = graph[(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*2-2:(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*3,columnchecknumber]

params, params_covariance = optimize.curve_fit(best_fit,x_thirdfifth,y_thirdfifth,p0=[1500,0.015,graph[int(len(graph[:,0])-1),0],0.2,100,7000],maxfev=1000000)
pyplot.plot(x_thirdfifth,best_fit(x_thirdfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_thirdfifth,y_thirdfifth, 'g*')

pyplot.title('Column '+str(columnchecknumber)+' as a Function of Time')
pyplot.xlabel('Time (Hours)')
pyplot.ylabel("Luminence (Topcount Units)")
pyplot.show()


# In[84]:


# This system cuts the desired graph into 5ths in order to minimize the effects an outlier can have on the entire function

#4th 5th

x_fourthfifth = graph[(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*3-2:(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*4,0]
y_fourthfifth = graph[(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*3-2:(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*4,columnchecknumber]

params, params_covariance = optimize.curve_fit(best_fit,x_fourthfifth,y_fourthfifth,p0=[1500,0.015,graph[int(len(graph[:,0])-1),0],0.2,100,7000],maxfev=1000000)
pyplot.plot(x_fourthfifth,best_fit(x_fourthfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_fourthfifth,y_fourthfifth, 'g*')

pyplot.title('Column '+str(columnchecknumber)+' as a Function of Time')
pyplot.xlabel('Time (Hours)')
pyplot.ylabel("Luminence (Topcount Units)")
pyplot.show()


# In[85]:


# This system cuts the desired graph into 5ths in order to minimize the effects an outlier can have on the entire function

#5th 5th

x_fifthfifth = graph[(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*4-2:len(graph[:,0]),0]
y_fifthfifth = graph[(int(len(graph[:,0])/5)+((len(graph[:,0])%5)>0))*4-2:len(graph[:,0]),columnchecknumber]

params, params_covariance = optimize.curve_fit(best_fit,x_fifthfifth,y_fifthfifth,p0=[1500,0.015,graph[int(len(graph[:,0])-1),0],0.2,100,7000],maxfev=1000000)
pyplot.plot(x_fifthfifth,best_fit(x_fifthfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_fifthfifth,y_fifthfifth, 'g*')

pyplot.title('Column '+str(columnchecknumber)+' as a Function of Time')
pyplot.xlabel('Time (Hours)')
pyplot.ylabel("Luminence (Topcount Units)")
pyplot.show()


# In[79]:


pyplot.plot(x_firstfifth,best_fit(x_firstfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_firstfifth,y_firstfifth, 'g*')

pyplot.plot(x_secondfifth,best_fit(x_secondfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_secondfifth,y_secondfifth, 'g*')

pyplot.plot(x_thirdfifth,best_fit(x_thirdfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_thirdfifth,y_thirdfifth, 'g*')

pyplot.plot(x_fourthfifth,best_fit(x_fourthfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_fourthfifth,y_fourthfifth, 'g*')

pyplot.plot(x_fifthfifth,best_fit(x_fifthfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_fifthfifth,y_fifthfifth, 'g*')

pyplot.show()

