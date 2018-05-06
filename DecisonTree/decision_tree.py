from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

allElectronicsData = open('AllElectronics.csv','r')
reader = csv.reader(allElectronicsData)
headers = reader.next()

print(reader)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

vec = DictVectorizer()
dumpyX = vec.fit_transform(featureList).toarray()

print("dumpyX: " + str(dumpyX))
#print(vec.get_feature_names())
#print("labelList: " + str(labelList))

lb = preprocessing.LabelBinarizer()
dumpyY = lb.fit_transform(labelList)
print("dumpyY: " + str(dumpyY))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dumpyX, dumpyY)
print("clf: " + str(clf))

with open("allElectronicInformationGainori.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dumpyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX:" + str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY: " + str(predictedY))
