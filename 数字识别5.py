
import cv2
import numpy as np
testSize=13
trainNum=165
k=7

g=np.zeros((testSize,1024),np.float32)
g1=np.zeros((trainNum,1024),np.float32)
for i in range (0,trainNum):
    str1="data"+"/"+str(i)+".jpg"
    img=cv2.imread(str1,0)
    img1=img.flatten()
    g1[i,:]=img1/255
def loadDataSettrainData(path):
    file = open(path, "r")
    List_row = file.readlines()
    list_source = []
    j=0
    for list_line in List_row:
        list_line = list(list_line.strip().split(','))
        s = []
        j = j + 1
        for i in list_line:

            s.append(int(i))
        list_source.append(s)
    print(j)
    return list_source
#read txt method one
path="data/trainDataLabel1.txt"
traindatalabel=loadDataSettrainData(path)
print(traindatalabel)

#trainLabelarray=[[0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0] ,[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0]]
trainLabelarray=traindatalabel
trainLabel=np.array(trainLabelarray,np.float32)
trainData=g1
for i in range (0,testSize):
    str1="testdata"+"/"+str(i)+".jpg"
    img=cv2.imread(str1,0)
    img1=img.flatten()
    g[i,:]=img1/255#将每个图的1024个像素值赋值在0矩阵的每一行，有几行代表有几个测试图
#testLabelarray=[[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0]]
path1="testdata/testdatalabel.txt"
testLabelarray=loadDataSettrainData(path1)
testLabel1=np.array(testLabelarray,dtype=np.float32)
print(testLabel1.shape)
print(testLabel1)
print(g.shape)
print(g)
testData=g
testLabel=testLabel1

#testLabel=mnist.test.labels[testIndex]
juzheng=np.expand_dims(g,axis=1)
juzheng1=np.tile(juzheng,[trainNum,1])
print("testdata增加维度",juzheng.shape)
print("testdata增加维度放大trainNum倍",juzheng1.shape)
juzheng2=np.expand_dims(g1,axis=0)
juzhengg2=np.tile(g1,[testSize,1,1])
print(juzheng2.shape)
print(juzhengg2.shape)
distance=(juzheng1-juzhengg2)**2
dsihe=np.sum(distance,axis=2)
sqrds=dsihe**0.5
print("第一张图对n张训练图的欧式距离",sqrds[0])
sortgg=np.argsort(sqrds)#从小到大排序
print("排序他们的索引号",sortgg)
#取前k个符合的
suitable=sortgg[:,0:k]
print(suitable)
labelarray=np.expand_dims(suitable,axis=2)
print(labelarray.shape)
labelarray2=np.tile(labelarray,[1,1,10])
print(labelarray2.shape)
for i in range(0,testSize):
    for j in range(0,k):
        labelarray2[i,j,:]=traindatalabel[int(suitable[i:i+1,j])]#将获取的某个矩阵中的值转换为一个数值
#print(int(suitable[0:0+1,0]))
#print(traindatalabel[0])
print(labelarray2)#每张图的k个特征标签
labelarray3=np.sum(labelarray2,axis=1)#将行数值累加
print(labelarray3)

p=[]#写出机器识别的数字
for i in range(0,testSize):
    max = -1
    for j in range(0,10):
        if(labelarray3[i][j]>=max):#max 相等时输出不同数字
            max=labelarray3[i][j]
            value=j
    flag=0
    for m in range(0,10):
        if(flag<1):
            if(labelarray3[i][m]==max):
                flag=flag+1
                p.append(m)
print("机器识别出的数字",p)
p1=[]#写出测试例子的数字
for j in range (0,testSize):
    max=-1
    for i in range(0,10):
        if(testLabelarray[j][i]==1):
            max=i
            p1.append(max)
print("测试的数字",p1)
#写出正确率
j=0
for i in range(0,testSize):
    if(p[i]==p1[i]):
        j=j+1
print("正确率",j/testSize*100,"%")
