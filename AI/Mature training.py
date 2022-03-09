import pybrain3
import pickle
import csv
import matplotlib.pylab as plt
from numpy import ravel
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from  pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.xml.networkreader import NetworkReader
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.structure import SoftmaxLayer
from pybrain3.structure import TanhLayer

# Create a dataset for training
ds = SupervisedDataSet(15, 1)
with open('training mature.csv', newline='') as csvfile:
    data_csv = csv.reader(csvfile, delimiter=';')
    for row in data_csv:
        input=[]
        for elements in row[:-1]:
            elements=elements.replace(",",".")
            input=float(elements)
        target = float(row[-1].replace(",","."))
        ds.addSample(input,target)

# Create structure of network
net = buildNetwork(15, 3, 1, hiddenclass=TanhLayer, outclass=SoftmaxLayer)


# Network training with visualization
trainer = BackpropTrainer(net, dataset=ds, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)
trnerr, valerr = trainer.trainUntilConvergence()
plt.plot(trnerr, 'b', valerr, 'r')
plt.show()

# Save trained network to file
ask = input('Save model? y/n ')
if input() == "y":
    fileObject = open('Mature dump.txt', 'wb')
    pickle.dump(net, fileObject)
    fileObject.close()
else:
    pass
