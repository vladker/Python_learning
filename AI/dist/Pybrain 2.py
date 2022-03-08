import pybrain3
import pickle
import matplotlib.pylab as plt
from pybrain3.datasets import SupervisedDataSet
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.tools.xml.networkreader import NetworkReader

net = buildNetwork(2, 3, 1)
y = net.activate([2, 1])

ds = SupervisedDataSet(2, 1)

xorModel = [
    [(0, 0), (0,)],
    [(0, 1), (1,)],
    [(1, 0), (1,)],
    [(1, 1), (0,)],
]
for input, target in xorModel:
    ds.addSample(input, target)

print(ds)

trainer = BackpropTrainer(net)
trnerr, valerr = trainer.trainUntilConvergence(dataset=ds, maxEpochs=100)
plt.plot(trnerr, 'b', valerr, 'r')
plt.show()
y = net.activate([1, 1])
print('Y=', y)

fileObject = open('MyNet.txt', 'wb')
pickle.dump(net, fileObject)
fileObject.close()
fileObject = open('MyNet.txt', 'rb')
net2= pickle.load(fileObject)
y = net2.activate([1,1])
print('Y2=', y)