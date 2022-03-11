import pickle

fileObject = open('Mature dump 4 full dataset.txt', 'rb')
net2 = pickle.load(fileObject)
fileObject.close()
print(net2)

# Good weather conditions
y = net2.activate([0,0,0,0.5])
print('Forecast value', y)
#
# # Average conditions
# y = net2.activate([1, 1, 0])
# print('Average conditions', y)
#
# # Bad conditions
# y = net2.activate([1, 1, 1])
# print('Bad conditions', y)
