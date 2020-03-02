import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

with open("../data/conservativenessTest.json") as f:
    data = json.load(f)

fig, ax = plt.subplots()

maxNumPyramids = np.array(data["conservative_test_options"]["max_num_pyramids"], dtype=np.int)
avgConservativeness = []
for pyramidLimit in maxNumPyramids:
    conservativeness = np.array(data["MaxNumPyramids"]["Conservativeness" + str(pyramidLimit)], dtype=np.double)
    avgConservativeness.append(np.mean(conservativeness))
plt.plot(maxNumPyramids, avgConservativeness)

plt.ylim(0.04, 0.08)
plt.xlim(0, 15)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
plt.xlabel('Maximum Allowed Pyramids')
plt.ylabel('Convervativeness')
 
plt.show()
