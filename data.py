import numpy as np

def load():
    x_train = []
    with open("sample_images.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            blocks = line.strip().split(";")
            x_train.append([blocks[0].split(":"),blocks[1].split(":")])
    x_train = [[[(np.asarray(x_train[i][j][k].split(",")).astype(np.float32)/255) for k in range(len(x_train[i][j]))] for j in range(len(x_train[i]))] for i in range(len(x_train))]
    return x_train

x_train = load()

##Example of how it works
#for i in range(len(x_train)):
#    for j in range(len(x_train[i])):
#        for k in range(len(x_train[i][j])):
#            print(x_train[i][j][k]," author: ",i," block: ",j," entry: ",k)