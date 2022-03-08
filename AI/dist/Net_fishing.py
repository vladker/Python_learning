import pybrain3
import pickle
import matplotlib.pylab as plt
from numpy import ravel
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from  pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.xml.networkreader import NetworkReader
from pybrain3.tools.xml.networkwriter import NetworkWriter

# Create a dataset for training

ds = SupervisedDataSet(4, 1)
ds.addSample([2, 3, 80, 1], [5])
ds.addSample([5, 5, 50, 2], [4])
ds.addSample([10, 7, 40, 3], [3])
ds.addSample([15, 9, 20, 4], [2])
ds.addSample([20, 11, 10, 5], [1])

# Create structure of network
net = buildNetwork(4, 3, 1, bias=True)

# Network training with visualization
trainer = BackpropTrainer(net, dataset=ds, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)
trnerr, valerr = trainer.trainUntilConvergence()
plt.plot(trnerr, 'b', valerr, 'r')
plt.show()

# Save trained network to file
fileObject = open('MyNer_Fish.txt', 'wb')
pickle.dump(net, fileObject)
fileObject.close()