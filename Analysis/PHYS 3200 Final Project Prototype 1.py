
# coding: utf-8

# In[25]:


# Smoothed Data method
from matplotlib import pyplot
import numpy as np
from scipy.signal import savgol_filter
graph = np.genfromtxt(r'C:\Users\nickr\Phys3200\biorsch2.csv')

columnchecknumber = int(input('Enter column you would like to check '))
x = graph[:,0]
y = graph[:,columnchecknumber]

y_new = savgol_filter(y,57,20)

pyplot.title('Column '+str(columnchecknumber)+' as a Function of Time')
pyplot.xlabel('Time (Hours)')
pyplot.ylabel("Luminence (Topcount Units)")

pyplot.plot(x,y_new,'r--')
pyplot.plot(x,y,'g+')
pyplot.show()


# In[2]:


#Taylor Series Method
from matplotlib import pyplot
import numpy as np
from scipy import optimize
graph = np.genfromtxt(r'C:\Users\nickr\Phys3200\biorsch2.csv')

columnchecknumber = int(input('Enter column you would like to check: '))
x = graph[:,0]
y = graph[:,columnchecknumber]

def best_fit(x,a,b,l,m,n,o,p,q,r,s,t,u,v):
    return b*x**12+a*x**11+l*x**10+m*x**9+n*x**8+o*x**7+p*x**6+q*x**5+r*x**4+s*x**3+t*x**2+u*x**1+v


params, params_covariance = optimize.curve_fit(best_fit,x,y)

pyplot.title('Column '+str(columnchecknumber)+' as a Function of Time')
pyplot.xlabel('Time (Hours)')
pyplot.ylabel("Luminence (Topcount Units)")

pyplot.plot(x,best_fit(x,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[10],params[11],params[12]),'r--')
pyplot.plot(x,y,'g-+')
pyplot.show()


# In[88]:


# A E S T H E T I C sine wave method ATTEMPT #1
from matplotlib import pyplot
import numpy as np
from scipy import optimize
graph = np.genfromtxt(r'C:\Users\nickr\Phys3200\biorsch2.csv')

columnchecknumber = int(input('Enter column you would like to check '))
x = graph[:,0]
y = graph[:,columnchecknumber]

def best_fit(x,a,b,c,d,e,f,g,h,i,j):
    return (h**((i*x)))*a/(x+0.00000000000000000001)**(e)*np.sin((j*x**g)/b)+c*(x**f)+d

params, params_covariance = optimize.curve_fit(best_fit,x,y,p0=[10000,3.81971863421,0,0,0,1.5,1,0,1,1])

pyplot.title('Column '+str(columnchecknumber)+' as a Function of Time')
pyplot.xlabel('Time (Hours)')
pyplot.ylabel("Luminence (Topcount Units)")

print(params)
pyplot.plot(x,best_fit(x,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]),'r--',)
pyplot.plot(x,y,'g-+')
pyplot.show()


# In[1]:


#Interpolation Method 
from matplotlib import pyplot 
import numpy as np 
from scipy import interpolate 

graph = np.genfromtxt(r'C:\Users\nickr\Phys3200\biorsch2.csv')
columnchecknumber = int(input('Enter column you would like to check ')) 
x = graph[:,0] 
y = graph[:,columnchecknumber] 
length_of_x = len(x)-1

f1 = interpolate.interp1d(x,y, kind = 'cubic') 
x1new = np.arange(x[0],x[length_of_x],float((x[length_of_x]-x[0])/10000))
y1new = f1(x1new)

pyplot.title('Column '+str(columnchecknumber)+' as a Function of Time') 
pyplot.xlabel('Time (Hours)') 
pyplot.ylabel("Luminence (Arbitrary Topcount Units)") 
pyplot.plot(x1new,y1new,'r--') 
pyplot.plot(x,y,'g+') 
pyplot.show()


# FINDS LOCAL MIN/MAX
length_y1new = len(y1new)-2 # this is -2 because minima/maxima cant be calculated at the first and last data point
y1new_period_place_counter = 1 
y1new_period_counter = 0 # for the while loop
y1new_maxima = np.array([])
y1new_minima = np.array([])
y1new_maxima_place_counter = np.array([])
y1new_minima_place_counter = np.array([])
y1new_maxima_counter = 0
y1new_minima_counter = 0

while y1new_period_counter < length_y1new:
    if (y1new[y1new_period_place_counter])>(y1new[y1new_period_place_counter-1]) and (y1new[y1new_period_place_counter])>(y1new[y1new_period_place_counter+1]):
        y1new_maxima = np.insert(y1new_maxima,y1new_maxima_counter,y1new[y1new_period_place_counter])
        y1new_maxima_place_counter = np.insert(y1new_maxima_place_counter,y1new_maxima_counter,y1new_period_counter+1)
        y1new_maxima_counter = y1new_maxima_counter + 1
    if (y1new[y1new_period_place_counter])<(y1new[y1new_period_place_counter-1]) and (y1new[y1new_period_place_counter])<(y1new[y1new_period_place_counter+1]):
        y1new_minima = np.insert(y1new_minima,y1new_minima_counter,y1new[y1new_period_place_counter]+1)
        y1new_minima_place_counter = np.insert(y1new_minima_place_counter,y1new_minima_counter,y1new_period_counter)
        y1new_minima_counter = y1new_minima_counter + 1
    y1new_period_counter = y1new_period_counter + 1
    y1new_period_place_counter = y1new_period_place_counter + 1

    
#FINDS PERIOD OF SAMPLE - uses code from FIND LOCAL MIN/MAX

#       MAXIMA
y1new_maxima_x1new_values = np.array([]) # values of x1new where y1new is at local maxima
period_while_loop_counter = 0 # for the while loop
while period_while_loop_counter < (len(y1new_maxima_place_counter)): # this while loop outputs an array of corrosponding y1new maxima's x1new value
    y1new_maxima_x1new_values = np.insert(y1new_maxima_x1new_values,period_while_loop_counter,x1new[int(y1new_maxima_place_counter[period_while_loop_counter])])
    period_while_loop_counter = period_while_loop_counter + 1

y1new_maxima_x1new_values_differences = np.array([]) # values of differences of x1new values corrosponding to y1new maxima
period_while_loop_counter = 0 # for while loop
while period_while_loop_counter < (len(y1new_maxima_x1new_values)-1): # builds array that has all distances between x1new values corrosponding to y1new maxima
    y1new_maxima_x1new_values_differences = np.insert(y1new_maxima_x1new_values_differences,period_while_loop_counter,(y1new_maxima_x1new_values[period_while_loop_counter+1]-y1new_maxima_x1new_values[period_while_loop_counter]))
    period_while_loop_counter = period_while_loop_counter + 1

#       MINIMA
y1new_minima_x1new_values = np.array([]) # values of x1new where y1new is at local maxima
period_while_loop_counter = 0 # for the while loop
while period_while_loop_counter < (len(y1new_minima_place_counter)): # this while loop outputs an array of corrosponding y1new maxima's x1new value
    y1new_minima_x1new_values = np.insert(y1new_minima_x1new_values,period_while_loop_counter,x1new[int(y1new_minima_place_counter[period_while_loop_counter])])
    period_while_loop_counter = period_while_loop_counter + 1

y1new_minima_x1new_values_differences = np.array([]) # values of differences of x1new values corrosponding to y1new maxima
period_while_loop_counter = 0 # for while loop
while period_while_loop_counter < (len(y1new_minima_x1new_values)-1): # builds array that has all distances between x1new values corrosponding to y1new maxima
    y1new_minima_x1new_values_differences = np.insert(y1new_minima_x1new_values_differences,period_while_loop_counter,(y1new_minima_x1new_values[period_while_loop_counter+1]-y1new_minima_x1new_values[period_while_loop_counter]))
    period_while_loop_counter = period_while_loop_counter + 1

#       Periodicity Average Calculations
y1new_maxima_x1new_period_average = np.average(y1new_maxima_x1new_values_differences)
y1new_minima_x1new_period_average = np.average(y1new_minima_x1new_values_differences)
y1new_minima_maxima_periodicity = np.average(np.insert(y1new_maxima_x1new_values_differences,len(y1new_maxima_x1new_values_differences)-1,y1new_minima_x1new_values_differences))
print(str('Period: ') + str(y1new_minima_maxima_periodicity))


#print(y1new_maxima_x1new_values_differences)


#maxima_period_tracker_time = np.array([0]) # array which contains element number of minima/maxima
#maxima_period_tracker_counter = 0 # for the while loop
#while maxima_period_tracker_counter < len(y1new_maxima_place_counter):
#    maxima_period_tracker_time = np.insert(maxima_period_tracker_time,maxima_period_tracker_counter,y1new_maxima_place_counter)
#    maxima_period_tracker_counter = maxima_period_tracker_counter + 1
#print(maxima_period_tracker_time)

#x1new[int(y1new_maxima_place_counter[0])]
#x1new[int(y1new_minima_place_counter[0])]


# In[1]:


# A E S T H E T I C diminishing, dampened sine wave method ATTEMPT #2

# DATA CULLER 1000

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

#1st 5th
x_firstfifth = graph[0:len(graph[0:int(len(graph[:,0])/5 + ((len(graph[:,0])%5)>0)),0]),0] #first part is the number the modulus part returns a 1 or 0 based on whether it is true or not so it rounds up essentially 
y_firstfifth = graph[0:len(graph[0:int(len(graph[:,columnchecknumber])/5 + ((len(graph[:,0])%5)>0)),0]),columnchecknumber]
params, params_covariance = optimize.curve_fit(best_fit,x_firstfifth,y_firstfifth,p0=[1500,0.015,graph[int(len(graph[:,0])-1),0],0.2,100,7000],maxfev=1000000)
pyplot.plot(x_firstfifth,best_fit(x_firstfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_firstfifth,y_firstfifth, 'g*')

#2nd 5th
x_secondfifth = graph[0:len(graph[len(x_firstfifth):int(len(graph[:,0])/5 + ((len(graph[:,0])%5)>0)),0]),0] #first part is the number the modulus part returns a 1 or 0 based on whether it is true or not so it rounds up essentially 
y_secondfifth = graph[0:len(graph[len(y_firstfifth):int(len(graph[:,columnchecknumber])/5 + ((len(graph[:,0])%5)>0)),0]),columnchecknumber]
params, params_covariance = optimize.curve_fit(best_fit,x_secondfifth,y_secondfifth,p0=[1500,0.015,graph[int(len(graph[:,0])-1),0],0.2,100,7000],maxfev=1000000)
pyplot.plot(x_secondfifth,best_fit(x_secondfifth,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
pyplot.plot(x_secondfifth,y_secondfifth, 'g*')


#params, params_covariance = optimize.curve_fit(best_fit,x,y,p0=[1500,0.015,graph[int(len(graph[:,0])-1),0],0.2,100,7000],maxfev=1000000)

#print(params)

pyplot.title('Column '+str(columnchecknumber)+' as a Function of Time')
pyplot.xlabel('Time (Hours)')
pyplot.ylabel("Luminence (Topcount Units)")
pyplot.show()

#pyplot.plot(x,best_fit(x,params[0],params[1],params[2],params[3],params[4],params[5]),'r--',)
#pyplot.plot(x,y,'g*')

