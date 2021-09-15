import sys 
import numpy

if __name__ == '__main__':
	trainInput = sys.argv[1]
	testInput = sys.argv[2]
	splitIndex = int(sys.argv[3])
	trainOut = sys.argv[4]
	testOut = sys.argv[5]
	metricsOut = sys.argv[6]

def readFile(path):
	# from 15-112 website
	with open(path, "rt") as f:
		return f.read()

def writeFile(path, contents):
	# from 15-112 website
	with open(path, "wt") as f:
		f.write(contents)

trainData = readFile(trainInput)
testData = readFile(testInput)

def stringToList(s):
	res = s.split("\n")
	res2 = list()
	for r in res:
		res2.append(r.split())
	res2 = res2[1:-1]
	return res2

def find_response_vars(D):
	lasts = [d[-1] for d in D]
	variables = ["",""]
	for item in lasts:
		if variables[0] == "" and item != "" :
			variables[0] = item
		elif variables[0] != item:
			variables[1] = item
	return variables

def majority(D):
	vars  = find_response_vars(D)
	zero_count = 0
	one_count = 0
	for d in D:
		if d[-1] == vars[0]:
			zero_count += 1
		else: 
			one_count += 1
	if zero_count > one_count:
		return vars[0]
	return vars[1]

def find_expl_vars(D):
	variables = ["",""]
	for i in range(len(D)):
		for j in range(len(D[i])-1):
			if variables[0] == "" and D[i][j] != "" :
				variables[0] = D[i][j]
			elif variables[0] != D[i][j]:
				variables[1] = D[i][j]
	return variables

def train(D,D_0,D_1):	
	D_list = stringToList(D)
	vars = find_expl_vars(D_list)	
	for d in D_list:
		if d[splitIndex]==vars[0]:
			D_0.append(d)
		else:
			D_1.append(d)
	return [majority(D_0),majority(D_1),vars[0],vars[1]]

[D0m_test,D1m_test,v0_test,v1_test] = train(testData,list(),list()) 
[D0m_train,D1m_train,v0_train,v1_train]  = train(trainData,list(),list())

def h(X,D0m,D1m,v0,v1,path):
	string = ""
	for x in X:
		if x[splitIndex] == v0:
			string = string + str(D0m) + "\n"
		elif x[splitIndex] == v1:
			string = string + str(D1m) + "\n"
	writeFile(path,string[:-1])

h(stringToList(testData),D0m_test,D1m_test,v0_test,v1_test,testOut)
h(stringToList(trainData),D0m_train,D1m_train,v0_train,v1_train,trainOut)

yhat_test = (readFile(testOut)).split("\n")
yhat_train = (readFile(trainOut)).split("\n")

y_test = [s[-1] for s in stringToList(testData)]
y_train = [s[-1] for s in stringToList(trainData)]

def error(y,yhat):
	error = 0
	for i in range(len(y)):
		if y[i]!=yhat[i]:
			error += 1
	return error/(len(y)) 

writeFile(metricsOut,"error(train): "+str(error(y_train,yhat_train))+"\n"+"error(test): "\
	+str(error(y_test,yhat_test))+"\n")
