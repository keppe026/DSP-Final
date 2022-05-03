# -*- coding: utf-8 -*-
"""
@author: pkeppel
Final Project - Data Analysis of Experimental Pressure Data
"""
import os
import csv
import scipy.signal as signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

"""
Set Working Directory and Define File Paths
"""
SetDirectory = r"G:\My Drive\URI\Data##\Final"
os.chdir(SetDirectory)
filename_1 = "Scope-Trial1.csv"

"""
Initialize list for desired data
"""
time=list()
p1=list()
p2=list()

"""
Import and Trim Data
""" 
with open(filename_1,'r') as r:
    for i in range(18):
        next(r)
        csvreader = csv.reader(r)
    for row in csvreader:
        row0=float(row[0])
        row1=float(row[1])
        row2=float(row[2])
        if float('0') < row0 < float('0.035'):
            time.append(row0)
            p1.append(row1)
            p2.append(row2)
        elif row0 > float('0.05'):
            break
       

resolution_sensor1=1 # Units of mv/psi
resolution_sensor2=5 # Units of mv/psi

time=np.array(time)*1000 # Convert to ms
p1=np.array(p1)*(1000/resolution_sensor1)
p2=np.array(p2)*(1000/resolution_sensor2)
"""
Perform Interval-dependent denoising of sensor data
"""
coeffs_p1 = pywt.wavedec(p1, "db4", level=8)
approx_p1 = coeffs_p1[0]
details_p1 = coeffs_p1[1:]

coeffs_p2 = pywt.wavedec(p2, "db4", level=8)
approx_p2 = coeffs_p2[0]
details_p2 = coeffs_p2[1:]


def neighbor_b(details, n, sigma):
    res = []
    L0 = int(np.log2(n) // 2)
    L1 = max(1, L0 // 2)
    L = L0 + 2 * L1
    def nb_beta(sigma, L, detail):
        S2 = np.sum(detail ** 2)
        lmbd = 4.50524 # solution of lmbd - log(lmbd) = 3
        beta = (1 - lmbd * L * sigma**2 / S2)
        return max(0, beta)
    for d in details:
        d2 = d.copy()
        for start_b in range(0, len(d2), L0):
            end_b = min(len(d2), start_b + L0)
            start_B = start_b - L1
            end_B = start_B + L
            if start_B < 0:
                end_B -= start_B
                start_B = 0
            elif end_B > len(d2):
                start_B -= end_B - len(d2)
                end_B = len(d2)
            assert end_B - start_B == L
            d2[start_b:end_b] *= nb_beta(sigma, L, d2[start_B:end_B])
        res.append(d2)
    return res

details_p1_n = neighbor_b(details_p1, len(p1), 0.8)
details_p2_n = neighbor_b(details_p2, len(p2), 0.8)

p1_dn = pywt.waverec([approx_p1] + details_p1_n, "db4") # Denoised signal
p2_dn = pywt.waverec([approx_p2] + details_p2_n, "db4") # Denoised signal

"""
Find Incident and Reflected Pressures and corresponding times
Find time where sensor 1 returns to Pressure equal to zero
"""
p1_reflected,_= signal.find_peaks(p1_dn,height=10,threshold=None,distance=None,prominence=15)
p1_reflected_time=time[p1_reflected]
p1_reflected=p1[p1_reflected]
p2_reflected,_= signal.find_peaks(p2_dn,height=10,threshold=None,distance=None,prominence=15)
p2_reflected_time=time[p2_reflected]
p2_reflected=p2[p2_reflected]

sliceend1=750
sliceend2=500
sliceend3=int(15/0.0004) # Desired max time for sensor 1 to return to zero is 15ms, sampling rate is 0.0004 ms

p1_sliced=p1[0:sliceend1]
p1_incident=max(p1_sliced)
p1_incident_index=np.argmax(p1_sliced)
p1_incident_time=time[p1_incident_index]
p2_sliced=p2[0:sliceend2]
p2_incident=max(p2_sliced)
p2_incident_index=np.argmax(p2_sliced)
p2_incident_time=time[p2_incident_index]

p1_slicedd=p1[sliceend1:sliceend3]
for row in p1_slicedd:
    if row < float('0'):
        p1_min=row
        break
    else:
        p1_min = None

p1_min_index=np.where(p1_slicedd == p1_min)
p1_min_index=p1_min_index[0]
p1_min_index=p1_min_index[0]
p1_min_time=time[p1_min_index]+time[sliceend1]

"""
Calculate Incident and Reflected Velocities Using Arrival times 
of pressure pulse and distance between sensors
"""
dist=0.161925 # distance between sensors in meters
v_incident=dist/((p1_incident_time-p2_incident_time)/1000)
v_reflected=dist/((p2_reflected_time-p1_reflected_time)/1000)

"""
Calculate Impulse - Convert units from psi to kg/m^2 and time from ms to s
"""

Ao=0.01824147 # Initial area in m^2
p1_dn_slice=p1_dn[0:(sliceend1+p1_min_index)]
p1_dn_slice=np.array(p1_dn_slice)*(1/2.20462)*(1550/1)
time_slice=time[0:(sliceend1+p1_min_index)]
time_slice=np.array(time_slice)/1000
impulse=np.trapz(p1_dn_slice,x=time_slice)*0.01824147

"""
Plot First 35 ms of raw and denoised data
"""
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400

fig1, (ax1, ax2) = plt.subplots(2, sharey=True)
fig1.suptitle('Shock Tube Pressure Data')
ax1.plot(time,p1_dn,label='P1 - Denoised Data',zorder=10)
ax1.plot(time,p1,label='P1 - Raw Data',zorder=9)
ax1.set_xlim(0,35)
ax1.set(xlabel='Time [ms]', ylabel='Pressure [psig]')
ax1.legend()
ax1.minorticks_on()
ax1.grid(which='both')

ax2.plot(time,p2_dn, label='P2 - Denoised Data',zorder=10)
ax2.plot(time,p2,label='P2 - Raw Data',zorder=9)
ax2.set_xlim(0,35)
ax2.set(xlabel='Time [ms]', ylabel='Pressure [psig]')
ax2.legend()
ax2.minorticks_on()
ax2.grid(which='both')

"""
Plot First 15 ms of denoised data and relavent pressure maximas
"""

fig2, (ax1, ax2) = plt.subplots(2, sharey=True)
fig2.suptitle('Shock Tube Pressure Data')
ax1.plot(time,p1_dn,label='P1 - Denoised Data',zorder=10)
ax1.plot(p1_incident_time,p1_incident,'*',label='P1 Incident')
ax1.plot(p1_reflected_time,p1_reflected,'*',label='P1 Reflected')
ax1.plot(p1_min_time,p1_min,'*',label='Impulse $t_{end}$')
ax1.set_xlim(0,15)
ax1.set(xlabel='Time [ms]', ylabel='Pressure [psig]')
ax1.legend()
ax1.minorticks_on()
ax1.grid(which='both')

ax2.plot(time,p2_dn, label='P2 - Denoised Data',zorder=10)
ax2.plot(p2_incident_time,p2_incident,'*',label='P2 Incident')
ax2.plot(p2_reflected_time,p2_reflected,'*',label='P2 Reflected')
ax2.set_xlim(0,15)
ax2.set(xlabel='Time [ms]', ylabel='Pressure [psig]')
ax2.legend()
ax2.minorticks_on()
ax2.grid(which='both')

"""
Print Summary Table
"""

data = [
        ['Incident Pressure [psi]','Reflected Pressure [psi]','Incident Velocity [m/s]','Reflected Velocity [m/s]','Impulse [J]'],
        [p1_incident,float(p1_reflected),float(v_incident),float(v_reflected),float(impulse)],

        ]

column_headers = data.pop(0)
cell_text = []
for row in data:
    cell_text.append(['%.2f'% x for x in row])

plt.figure(linewidth=2,edgecolor='black',tight_layout={'pad':1},)
the_table = plt.table(cellText=cell_text,cellLoc='center',rowLabels=None,rowLoc='right',colLabels=column_headers,loc='center')
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.box(on=None)












