import numpy as np
import json
import matplotlib.pyplot as plt

platforms = ['i7', 'TX2', 'ODROID', '']
labels = ['i7', 'TX2', 'ODROID', 'yours']
sytles = ['.r-.', '.g:', '.b--', '.k-']
fig = plt.figure()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
for i, platform in enumerate(platforms):
    
    try:
        with open("../data/avgTrajGen" + platform + ".json") as f:
            data = json.load(f)
    except Exception:
        print(platform + ' file not found.')
        continue
    
    compTimeMs = np.array(data["CompTime"], dtype=np.double) * 1e3
    avgTrajGen = np.array(data["AvgTrajGen"], dtype=np.double)
    
    plt.plot(compTimeMs, avgTrajGen, sytles[i], label=labels[i])

plt.yscale("log")
plt.ylim([10, 200000])
plt.legend(markerscale=0)
plt.xlabel('Allocated Computation Time [ms]')
plt.ylabel('# of Trajectories Evaluated')

plt.show()
