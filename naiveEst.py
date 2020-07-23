# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:16:15 2020

@author: Lenovo
"""
data=[]
with open("data.txt", "r") as file:
    first_line = file.readline()
    n= first_line[0]
    m= first_line[2]
    for i in file:
        x = i.split()
        data.append(x)
m=int(m)        
n=int(n) 
bin_width=2 
for i in range(n):
    for j in range(m):
        data[i][j]=float(data[i][j])
        
prob_den=[]

for i in range(n):
    ref=data[i]
    counter=0
    for j in range(n):
        flag=0
        for k in range(m):
            if abs(data[j][k]-ref[k])/bin_width>=0.5:
                flag=1        
        if flag==0:
            counter+=1
    prob_den.append(counter/(n*(bin_width**m)))

file = open("output.txt","w") 
for i in prob_den:
    a=str(i)
    file.write(a)
    file.write('\n')
file.close() 


    