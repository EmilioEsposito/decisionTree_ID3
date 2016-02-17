
import sys
import math

# Decision Tree Class
class DecTree:
    def __init__(self, t, ts, cn):
        self.__train = t
        self.__tst = ts
        self.__colnames = cn
        self.__predTrain = []
        self.__truthTrain = []
        self.__predTest = []
        self.__truthTest = []
        self.__outLst = []
        self.__allytrain = transposeLL(self.__train)[len(transposeLL(self.__train))-1]
        self.__splits = self.__splitTree(self.__train, self.__tst, self.__colnames, 0)
    def getSplits(self):
        return self.__splits
    def getOutLst(self):
        for i in sorted(self.__outLst):
            print i[2:]
    def getpredTrain(self):
        return self.__predTrain
    def gettruthTrain(self):
        return self.__truthTrain
    def getpredTest(self):
        return self.__predTest
    def gettruthTest(self):
        return self.__truthTest
    def __splitTree(self, trnR, tstR, colnames, level, parent = "root"):
        trnC = transposeLL(trnR)
        tstC = transposeLL(tstR)
        xtrnC = trnC[:len(trnC)-1]
        xtstC = tstC[:len(tstC)-1]
        ytrain = trnC[len(trnC)-1]
        ytest = tstC[len(tstC)-1]
        # find which x has highest IG with output
        igs = [info_gain(x,ytrain) for x in xtrnC]
        # print zip(colnames, igs)
        # get x with highest IG
        in_maxIG = igs.index(max(igs))

        # print "highest ig: ",colnames[in_maxIG]
        xtr = xtrnC[in_maxIG]
        xts = xtstC[in_maxIG]

        # Grow the tree
        if max(igs)>.1 and level<2:
            trnRSplit,grps = tapply(trnR, xtr, list, returnGrp=True)
            tstRSplit,grps2 = tapply(tstR, xts, list, returnGrp=True)

            if level == 0:
                print countPosNeg(ytrain, self.__allytrain)

            # store tree node info in strings
            counter = 0
            for branch,gr in zip(trnRSplit,grps):
                if level > 0:
                    outScr = "| "
                else:
                    outScr= ""

                outScr += colnames[in_maxIG]+" = "+gr+": "
                vals = transposeLL(branch)
                vals = vals[len(vals)-1]
                outScr += countPosNeg(vals, self.__allytrain)

                if parent=="root":
                    if counter==0:
                        self.__outLst.append(str(in_maxIG)+gr+outScr)
                    else:
                        self.__outLst.append(str(4)+gr+outScr)
                else:
                    self.__outLst.append(parent+gr+outScr)
                counter += 1


            # RECURSION
            trnRSplit  = [self.__splitTree(s, sts, colnames, level+1, str(gr)) for s,sts,gr in zip(trnRSplit,tstRSplit,[3,5])]

        else:
            trnRSplit = trnR
            self.__predTrain += getPredMCV(ytrain)
            self.__predTest += [mcv(ytrain)]*len(tstR)
            self.__truthTrain += ytrain
            self.__truthTest += ytest
        return trnRSplit

# get transposed df: dfT
def transposeLL(df):
    dfT = []
    for c in range(0, len(df[0])):
        dfT.append([])
        for r in range(0, len(df)):
            dfT[c].append(df[r][c])
    return dfT

# calc entropy for a vector
def entropy(x):
    p = prob(x)
    hx = sum([list_el * math.log(1/list_el,2) for list_el in p])
    return hx

# calc prob of unique values in a vector
def prob(x):
    unique = set(x)
    p = []
    for i, val in enumerate(unique):
        p.append(0)
        for item in x:
            if val == item:
                p[i] += 1
    p = [list_el/float(len(x)) for list_el in p]
    return p

# return the most common value of a vector
def mcv(x):
    freq = [x.count(i) for i in x]
    mcv = x[freq.index(max(freq))]
    return mcv

# calculate error rate when given a vector of true labels and a vector of pred labels
def errorrate(truth, pred):
    right = 0
    wrong = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
            right +=1
        else:
            wrong +=1
    er = wrong/float(len(truth))
    return er

# use majority vote (MCV) for all predictions
# used for tree leaves
def getPredMCV(x):
    pred = [mcv(x) for i in x]
    return pred

# write my own tapply similar to R. It runs a function on a list grouped by any other list
def tapply(tgLst, groupLst, fun, returnGrp = False):
    result = []
    for group in set(groupLst):
        tg_groupi =[]
        for k, groupk in enumerate(groupLst):
            if group == groupk:
                tg_groupi.append(tgLst[k])
        result.append(fun(tg_groupi))

    #optionally return the groups
    if returnGrp:
        return result, set(groupLst)
    else:
        return result

# calculate information gain aka mutual information
def info_gain(x,y):
    hy = entropy(y)
    pxi = prob(x)
    hy_xi = tapply(y,x,entropy, returnGrp = False)
    hy_x = 0
    for p,h in zip(pxi,hy_xi):
        if p!=0:
            hy_x += (p*h)
        if p==1:
            hy_x = hy
    # calculate Mutual Info or Info Gain
    ig = hy - hy_x
    return ig

# format counts of binary classes into [#+/#-]
def countPosNeg(vals, iToCount):
    cntStr = "["
    for v in sorted(list(set(iToCount))):
        cntStr += str(vals.count(v))
        if v=="democrat" or v =="A" or v=="Ayes":
            cntStr += "+/"
        else:
            cntStr += "-]"
    return cntStr


def main():
    # TRAIN
    f = open(sys.argv[1], 'r')
    train = [line.strip().split(',') for line in f.readlines()]
    train.remove(train[0])

    for r,line in enumerate(train):
        for c,i in enumerate(line):
            if train[r][c] == "yes":
                train[r][c] = "A"

    f = open(sys.argv[2],"r")
    test = [line.strip().split(',') for line in f.readlines()]

    for r,line in enumerate(test):
        for c,i in enumerate(line):
            if test[r][c] == "yes":
                test[r][c] = "A"

    colnames = test[0]
    test.remove(test[0])

    # grow the tree
    print("Show the decision tree splits on training data:")
    tree1 = DecTree(train, test, colnames)

    # show training nodes splits
    tree1.getOutLst()

    # show train/test error rates
    print "error(train):",errorrate(tree1.getpredTrain(),tree1.gettruthTrain())
    print "error(test):",errorrate(tree1.getpredTest(),tree1.gettruthTest())
main()
