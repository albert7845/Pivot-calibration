# -*- coding: utf-8 -*-

import numpy as np# (2 points)
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


filePath = 'PivotCalib3.tsv'
op = np.genfromtxt(filePath,skip_header=1, delimiter='\t') 

Q0 = op[:,5] 
Qx = op[:,6] 
Qy = op[:,7] 
Qz = op[:,8] 

Ts = op[:,9:12] 

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(Ts[:,0],Ts[:,1],Ts[:,2],c=Ts[:,2])
plt.show()

refTs = np.array(Ts.reshape(-1,1)) 

refRs = []

for measurementOP in range(len(Ts)):

    r = R.from_quat([Qx[measurementOP], Qy[measurementOP], Qz[measurementOP], Q0[measurementOP]])
    refRs.extend(np.concatenate((r.as_matrix(), -np.identity(3)),axis=1))

refRs = np.stack(refRs) 

optimizedT = np.linalg.lstsq(refRs,np.negative(refTs)) 

sTp = optimizedT[0][0:3] 

estimatedPositions = -refRs@optimizedT[0] 
error = refTs - estimatedPositions 
error_reshaped = np.array(error.reshape(len(Ts),3))

error_Euclidean = np.linalg.norm(error_reshaped,axis=1) 
mean_error = np.mean(error_Euclidean)  

print('Calibration error is: %.2f mm' %mean_error)
