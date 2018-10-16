import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument("-file",default="./core.hdf5")
parser.add_argument("-start",type=float,default=2.0,help='')
parser.add_argument("-to",type=float,default=2.7,help='')
args = parser.parse_args()

start = int(args.start*10)-10
end = int(args.to*10)-10

T = np.arange(args.start,args.to,0.1)

with h5py.File(args.file,'r') as f:
    LOSS = np.array(f["loss"])
    STD = np.array(f['std'])

loss1 = LOSS[0]
loss2 = LOSS[1]
loss3 = LOSS[2]
loss4 = LOSS[3]
loss5 = LOSS[4]

error1 = STD[0]
error2 = STD[1]
error3 = STD[2]
error4 = STD[3]
error5 = STD[4]

exact = np.array([2049.049789690087,1863.262057528506,1708.764233110076,1578.4785309636084,1467.3754747251378,1371.7903617635538,1288.9971104105036,1216.9341973396154,1154.024456199465,1099.055787488181,1051.104987614517,1009.4996784717939,973.8408186679096,944.3823241807323,920.8489585246118,901.1617063065412,884.3618952526879,869.8418508112793,857.1705412103629,846.0253831110782,836.1570646028667,827.3685472309867,819.5014489849076,812.4267256294704,806.0380467741903,800.2469515985453])

fix = np.array([2954.9706600952036,2722.2957088348962,2524.861133432476,2354.828220112043,2206.5553784379786,2075.8743282435726,1959.6355634994688,1855.4128511930287,1761.305306230543,1675.801337703902,1597.682970064889,1525.9571863911228,1459.8057676137014,1398.5480483845513,1341.612857520735,1288.5170967249728,1238.8491888829687,1192.2561471669637,1148.4333700194834,1107.1165118352076,1068.0749509785837,1031.1064990650011,996.0330835834103,962.697200232009,930.958978748977,900.6937413401133])

exact = exact[start:end+1]
fix = fix[start:end+1]

res1 = np.abs(-loss1-exact-fix)/(exact+fix)
res2 = np.abs(-loss2-exact-fix)/(exact+fix)
res3 = np.abs(-loss3-exact-fix)/(exact+fix)
res4 = np.abs(-loss4-exact-fix)/(exact+fix)
res5 = np.abs(-loss5-exact-fix)/(exact+fix)

error5 /= (512)**0.5*(exact+fix)
error4 /= (512)**0.5*(exact+fix)
error3 /= (512)**0.5*(exact+fix)
error2 /= (512)**0.5*(exact+fix)
error1 /= (512)**0.5*(exact+fix)

plt.errorbar(T,res1,label="depth = 5",yerr=error1)
plt.errorbar(T,res2,label="depth = 4",yerr=error2)
plt.errorbar(T,res3,label="depth = 3",yerr=error3)
plt.errorbar(T,res4,label="depth = 2",yerr=error4)
plt.errorbar(T,res5,label="depth = 1",yerr=error5)

plt.legend()
plt.xlabel("Temperature")
plt.ylabel("$|loss-lnZ|$")

plt.show()


import pdb
pdb.set_trace()