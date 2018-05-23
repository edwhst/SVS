import numpy as np
from keras.regularizers import l2
from keras import backend as K
from keras import callbacks
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout
from keras.layers import concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses_val = []
        self.losses_train = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))
        self.losses_train.append(logs.get('loss'))


class EarlyStopping_byvalue(callbacks.Callback):
     def __init__(self,monitor='val_mean_absolute_error',value=0.1,verbose=0):
        super(callbacks.Callback,self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

     def on_epoch_end(self,epoch, logs={}):
        current = logs.get(self.monitor)
        if current is not None:
            if current < self.value:
                if self.verbose > 0:
                    print("Early stopping at epoch %02d" % epoch)
                    self.model.stop_training = True


"""Siamese network pairs signatures classification"""

class batches():
    def __init__(self,files):
        self.files = files
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def sets(self):
        for i,f in enumerate(self.files):
            if i==0:
                self.x_train, self.y_train = self.load_data(f)
            else:
                self.x_test, self.y_test = self.load_data(f)

    def load_data(self,file):
        X = []
        with open(file,"r") as f:
            lines = f.readlines()
            for line in lines:
                blocks = line.strip(",:\n").split(",;")
                X.append([blocks[0].strip().split(":"),blocks[1].strip().split(":")])
        X = [[[(np.asarray(X[i][j][k].strip(",").split(",")).astype(np.float32)/255)\
                                    for k in range(len(X[i][j])) if len(X[i][j][k])>0]\
                                    for j in range(len(X[i]))]\
                                    for i in range(len(X))] # i:writer, j:genuines or forgeries block, k:id signature in block
        return self.create_pairs(X)

    def create_pairs(self,X):
        x_pairs = []
        y_pairs = []
        for writer in range(len(X)):    # number of authors
            right_pairs = []    # genuine vs genuine
            fake_pairs = []     # genuine vs forgerie
            y_right = []    # ones (label)
            y_fake = []     # zeros (label)
            for i in range(len(X[writer][0])):
                for j in range(len(X[writer][0])):
                    if(i!=j):
                        right_pairs.append([X[writer][0][i],X[writer][0][j]]) # genuines pairs
                        y_right.append(1)
                for k in range(len(X[writer][1])):
                    fake_pairs.append([X[writer][0][i],X[writer][1][k]]) # forgeries pairs
                    y_fake.append(0)
            x_pairs += [np.concatenate((np.asarray(right_pairs),np.asarray(fake_pairs)),axis=0)]
            y_pairs += [np.concatenate((y_right,y_fake),axis=0)]
        return np.asarray(x_pairs),np.asarray(y_pairs)


class add_genuines():
    def __init__(self):
        self.x_pairs = []
        self.y_pairs = []

    def setxy(self):
        self.x_pairs, self.y_pairs = self.load_testset_genuines()

    def load_trainset(self):
        X = []
        with open("traingset.txt","r") as f:
            lines = f.readlines()
            for line in lines:
                blocks = line.strip(",:\n").split(",;")
                X.append([blocks[0].strip().split(":"),blocks[1].strip().split(":")])
        X = [(np.asarray(X[i][j][k].strip(",").split(",")).astype(np.float32)/255)\
                                    for i in range(len(X))\
                                    for j in range(len(X[i]))\
                                    for k in range(len(X[i][j])) if len(X[i][j][k])>0]
        return np.asarray(X)

    def load_testset_genuines(self):
        with open("testset_ref.txt","r") as f:
            X = []
            lines = f.readlines()
            for line in lines:
                blocks = line.strip(",:\n").split(",:")
                X.append(blocks)
            X = [[(np.asarray(X[i][j].strip(",").split(",")).astype(np.float32)/255) for j in range(len(X[i]))]\
                                                                                    for i in range(len(X))]
        return self.best10(X)

    def best10(self,X):
        pairs = self.load_trainset()
        best10 = []
        for writer in X:
            best_writer = []
            for genuine in writer:
                best_pair = []  # closest to writer's genuine signature
                for pair in pairs:
                    best_pair.append((np.sum(np.abs(genuine-pair))/pair.size))
                best_pair = np.asarray(best_pair)
                if len(best_writer)==0: best_writer.append(np.where(best_pair==best_pair[best_pair.argsort()][-1])[0][0])
                else:
                    aux2 = -1
                    while(len(best_writer) < 3):
                        aux = np.where(best_pair==best_pair[best_pair.argsort()][aux2])[0][0]
                        for bw in best_writer:
                            if aux!= bw:
                                best_writer.append(aux)
                        aux2-=1
            best10.append(pairs[best_writer])
        return self.y_best10(best10,X)
    
    def y_best10(self,best10,genuines):
        x_pairs = []
        y_pairs = []
        for wi,writer in enumerate(genuines):
            for i,gi in enumerate(writer):              
                right_pairs = []
                fake_pairs = []
                y_right = []
                y_fake = []
                for i2,gi2 in enumerate(writer):
                    if(i != i2):
                        right_pairs.append([gi,gi2])
                        y_right.append(1)
                for b10 in best10[wi]:
                    fake_pairs.append([gi,b10])
                    y_fake.append(0)
            x_pairs += [np.concatenate((np.asarray(right_pairs),np.asarray(fake_pairs)),axis=0)]
            y_pairs += [np.concatenate((y_right,y_fake),axis=0)]
        return np.asarray(x_pairs),np.asarray(y_pairs)


class siamese_convnet():
    def __init__(self,sh):
        self.sh = sh

    def create(self):
        self.model = self.convnet_features(self.sh)

    def convnet_features(self,sh):
        img_in = Input(shape = sh, name = 'FeatureNet_ImageInput')
        n_layer = img_in
        for i in range(2):
            n_layer = Conv2D(8*2**i, kernel_size = (1,1), activation = 'linear')(n_layer)
            n_layer = BatchNormalization()(n_layer)
            n_layer = Activation('relu')(n_layer)
            n_layer = Conv2D(16*2**i, kernel_size = (1,1), activation = 'linear')(n_layer)
            n_layer = BatchNormalization()(n_layer)
            n_layer = Activation('relu')(n_layer)
            n_layer = MaxPool2D((2,2))(n_layer)
        n_layer = Flatten()(n_layer)
        n_layer = Dense(32, activation = 'linear')(n_layer)
        #n_layer = Dropout(0.5)(n_layer)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        feature_model = Model(inputs = [img_in], outputs = [n_layer], name = 'FeatureGenerationModel')
        #feature_model.summary()
        return self.similarity(self.sh,feature_model)

    def similarity(self,sh,feature_model):
        img_a_in = Input(shape = sh, name = 'ImageA_Input')
        img_b_in = Input(shape = sh, name = 'ImageB_Input')
        img_a_feat = feature_model(img_a_in)
        img_b_feat = feature_model(img_b_in)
        combined_features = concatenate([img_a_feat, img_b_feat], name = 'merge_features')
        combined_features = Dense(16, activation = 'linear')(combined_features)
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(4, activation = 'linear')(combined_features)
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(1, activation = 'sigmoid')(combined_features)
        similarity_model = Model(inputs = [img_a_in, img_b_in], outputs = [combined_features], name = 'Similarity_Model')
        #similarity_model.summary()
        return similarity_model


datasets = ["traingset.txt","test_questioned.txt"]
dp = batches(datasets)
dp.sets()

x_sets = [dp.x_train,dp.x_test]
y_sets = [dp.y_train,dp.y_test]
pairs_sets = []
labels_sets = []

for x,y in zip(x_sets,y_sets):
    i = 0
    for xi,yi in zip(x,y):
        if i == 0:
            pairs = xi.reshape(-1,2,136,80,1)
            labels = yi
            i+=1
        else:
            pairs = np.append(pairs,xi.reshape(-1,2,136,80,1),axis=0)
            labels = np.append(labels, yi)
    pairs_sets.append(pairs)
    labels_sets.append(labels)

##Image verification
#print(ptr[0,0].shape)
#plt.figure()
#plt.imshow(np.transpose(ptr[0,0]),cmap='gray')
#plt.show()
#exit()

"""Siamese+convnet 1st approach: Mix datasets and randomely split for traning, validation and testing"""
full_set_pairs = np.append(pairs_sets[0],pairs_sets[1],axis=0)
full_set_labels = np.append(labels_sets[0],labels_sets[1],axis=0)
print(full_set_pairs.shape,full_set_labels.shape)
idx = np.arange(full_set_labels.shape[0])
np.random.shuffle(idx)
trn1 = idx[:((int)(full_set_pairs.shape[0]*0.6))] #60% training
val1 = idx[((int)(full_set_pairs.shape[0]*0.6)):((int)(full_set_pairs.shape[0]*0.8))] #20% validation
tst1 = idx[((int)(full_set_pairs.shape[0]*0.8)):] #20% testing

model1 = siamese_convnet(full_set_pairs[0,0,0:].shape)
model1.create()
model1.model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics = ['mae','acc'])

history1 = LossHistory()

loss = model1.model.fit([full_set_pairs[trn1,0],full_set_pairs[trn1,1]],full_set_labels[trn1],
                    callbacks=[EarlyStopping_byvalue(monitor='val_mean_absolute_error', value=0.05, verbose=1),history1],
                    epochs = 50,
                    batch_size = 100,
                    validation_data = ([full_set_pairs[val1,0],full_set_pairs[val1,1]],full_set_labels[val1]),
                    verbose = False)

print("Siamese+convnet | Accuracy with a mix of sets and further splitting: {0:2.4f}".format(accuracy_score(full_set_labels[tst1],np.where(model1.model.predict([full_set_pairs[tst1,0],full_set_pairs[tst1,1]])>0.9,1,0))))

"""Siamese+convnet 2nd approach: Testing with test dataset"""
idx2 = np.arange(pairs_sets[0].shape[0])
np.random.shuffle(idx2)
trn2 = idx2[:((int)(pairs_sets[0].shape[0]*0.7))]   # 70% training
val2 = idx2[((int)(pairs_sets[0].shape[0]*0.7)):]   # 30% validation

model2 = siamese_convnet(pairs_sets[0][0,0,0:].shape)
model2.create()
model2.model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics = ['mae','acc'])

history2 = LossHistory()

loss = model2.model.fit([pairs_sets[0][trn2,0],pairs_sets[0][trn2,1]],labels_sets[0][trn2],
                    callbacks=[EarlyStopping_byvalue(monitor='val_mean_absolute_error', value=0.05, verbose=1),history2],
                    epochs = 50,
                    batch_size = 100,
                    validation_data = ([pairs_sets[0][val2,0],pairs_sets[0][val2,1]],labels_sets[0][val2]),
                    verbose = False)

print("Siamese+convnet | Accuracy with independent test set: {0:2.4f}".format(accuracy_score(labels_sets[1],np.where(model2.model.predict([pairs_sets[1][:,0],pairs_sets[1][:,1]])>0.9,1,0))))


"""Siamese+convnet 3rd approach: Add genuine signatures from test set and make their forgeries pairs with the closest signateres from the traning set"""
gn = add_genuines()
gn.setxy()
i = 0

for x,y in zip(gn.x_pairs,gn.y_pairs):
    if i==0:
        x_train_addition = x
        y_train_addition = y
        i+=1
    else:
        x_train_addition = np.append(x_train_addition,x,axis=0)
        y_train_addition = np.append(y_train_addition,y,axis=0)

x_train = np.append(pairs_sets[0],x_train_addition.reshape(x_train_addition.shape[0],2,136,80,1),axis=0)
y_train = np.append(labels_sets[0],y_train_addition,axis=0)

idx3 = np.arange(x_train.shape[0])
np.random.shuffle(idx3)
trn3 = idx2[:((int)(x_train.shape[0]*0.7))]   # 70% training
val3 = idx2[((int)(x_train.shape[0]*0.7)):]   # 30% validation

model3 = siamese_convnet(x_train[0,0,0:].shape)
model3.create()
model3.model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics = ['mae','acc'])

history3 = LossHistory()

loss = model3.model.fit([x_train[trn3,0],x_train[trn3,1]],y_train[trn3],
                    callbacks=[EarlyStopping_byvalue(monitor='val_mean_absolute_error', value=0.05, verbose=1),history3],
                    epochs = 50,
                    batch_size = 100,
                    validation_data = ([x_train[val3,0],x_train[val3,1]],y_train[val3]),
                    verbose = False)

print("Siamese+convnet | Accuracy with independent test set: {0:2.4f}".format(accuracy_score(labels_sets[1],np.where(model3.model.predict([pairs_sets[1][:,0],pairs_sets[1][:,1]])>0.9,1,0))))


"""MLP individual signatures classification"""

class batches_indiv():
    def __init__(self,files):
        self.files = files
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def sets(self):
        for i,f in enumerate(self.files):
            if i==0:
                self.x_train, self.y_train = self.load_data(f)
            else:
                self.x_test, self.y_test = self.load_data(f)

    def load_data(self,file):
        X = []
        with open(file,"r") as f:
            lines = f.readlines()
            for line in lines:
                blocks = line.strip(",:\n").split(",;")
                X.append([blocks[0].strip().split(":"),blocks[1].strip().split(":")])
        X = [[[(np.asarray(X[i][j][k].strip(",").split(",")).astype(np.float32)/255)\
                                    for k in range(len(X[i][j])) if len(X[i][j][k])>0]\
                                    for j in range(len(X[i]))]\
                                    for i in range(len(X))] # i:writer, j:genuines or forgeries block, k:id signature in block
        return self.createxy(X)

    def createxy(self,X):
        for i,writer in enumerate(X):    # number of authors
            if i==0:
                x = writer[0]
                y = np.ones(len(writer[0]))
                x = np.append(x,writer[1])
                y = np.append(y,np.zeros(len(writer[1])))
            else:
                x = np.append(x,writer[0])
                y = np.append(y,np.ones(len(writer[0])))
                x = np.append(x,writer[1])
                y = np.append(y,np.zeros(len(writer[1])))
        return x,y

class mlp_convnet():
    def __init__(self,sh):
        self.sh = sh

    def create(self):
        self.model = self.convnet_features(self.sh)

    def convnet_features(self,sh):
        img_in = Input(shape = sh, name = 'FeatureNet_ImageInput')
        n_layer = img_in
        for i in range(2):
            n_layer = Conv2D(8*2**i, kernel_size = (1,1), activation = 'linear')(n_layer)
            n_layer = BatchNormalization()(n_layer)
            n_layer = Activation('relu')(n_layer)
            n_layer = Conv2D(16*2**i, kernel_size = (1,1), activation = 'linear')(n_layer)
            n_layer = BatchNormalization()(n_layer)
            n_layer = Activation('relu')(n_layer)
            n_layer = MaxPool2D((2,2))(n_layer)
        n_layer = Flatten()(n_layer)
        n_layer = Dense(32, activation = 'linear')(n_layer)
        #n_layer = Dropout(0.5)(n_layer)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        feature_model = Model(inputs = [img_in], outputs = [n_layer], name = 'FeatureGenerationModel')
        #feature_model.summary()
        return self.classifier(self.sh,feature_model)

    def classifier(self,sh,feature_model):
        img_in = Input(shape = sh, name = 'Image_Input')
        img_feat = feature_model(img_in)
        features = img_feat
        features = Dense(16, activation = 'linear')(features)
        features = BatchNormalization()(features)
        features = Activation('relu')(features)
        features = Dense(4, activation = 'linear')(features)
        features = BatchNormalization()(features)
        features = Activation('relu')(features)
        features = Dense(1, activation = 'sigmoid')(features)
        mlp_model = Model(inputs = img_in, outputs = features, name = 'MLP_model')
        #mlp_model.summary()
        return mlp_model

datasets = ["traingset.txt","test_questioned.txt"]
dp2 = batches_indiv(datasets)
dp2.sets()

dp2.x_train = dp2.x_train.reshape(-1,136,80,1)
dp2.x_test = dp2.x_test.reshape(-1,136,80,1)


"""MLP+convnet 1st approach: Mix datasets and randomely split for traning, validation and testing"""
x_all = np.append(dp2.x_train,dp2.x_test,axis=0)
y_all = np.append(dp2.y_train,dp2.y_test,axis=0)

idx4 = np.arange(x_all.shape[0])
np.random.shuffle(idx4)
trn4 = idx4[:((int)(x_all.shape[0]*0.6))] #60% training
val4 = idx4[((int)(x_all.shape[0]*0.6)):((int)(x_all.shape[0]*0.8))] #20% validation
tst4 = idx4[((int)(x_all.shape[0]*0.8)):] #20% testing

model4 = mlp_convnet(x_all.reshape(-1,136,80,1)[0].shape)
model4.create()
model4.model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics = ['mae','acc'])

history4 = LossHistory()

loss = model4.model.fit(x_all[trn4],y_all[trn4],
                    callbacks=[EarlyStopping_byvalue(monitor='val_mean_absolute_error', value=0.2, verbose=1),history4],
                    epochs = 50,
                    batch_size = 100,
                    validation_data = (x_all[val4],y_all[val4]),
                    verbose = False)

print("MLP+convnet | Accuracy with a mix of sets and further splitting: {0:2.4f}".format(accuracy_score(y_all[tst4],np.where(model4.model.predict(x_all[tst4])>0.9,1,0))))


"""MLP+convnet 2nd approach: Testing with test dataset"""
idx5 = np.arange(dp2.x_train.shape[0])
np.random.shuffle(idx5)
trn5 = idx5[:((int)(dp2.x_train.shape[0] * 0.7))]  # 70% training
val5 = idx5[((int)(dp2.x_train.shape[0] * 0.7)):]  # 30% validation

model5 = mlp_convnet(dp2.x_train.reshape(-1,136,80,1)[0].shape)
model5.create()
model5.model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics = ['mae','acc'])

history5 = LossHistory()

loss = model5.model.fit(dp2.x_train[trn5],dp2.y_train[trn5],
                    callbacks=[EarlyStopping_byvalue(monitor='val_mean_absolute_error', value=0.2, verbose=1),history5],
                    epochs = 50,
                    batch_size = 100,
                    validation_data = (dp2.x_train[val5],dp2.y_train[val5]),
                    verbose = False)

print("MLP+convnet | Accuracy with independent testing set: {0:2.4f}".format(accuracy_score(dp2.y_test,np.where(model5.model.predict(dp2.x_test)>0.9,1,0))))


"""MLP-convnet 3rd approach: Adding the testing set genuines to training set"""
def load_testset_genuines():
    with open("testset_ref.txt","r") as f:
        X = []
        lines = f.readlines()
        for line in lines:
            blocks = line.strip(",:\n").split(",:")
            X.append(blocks)
        X = [(np.asarray(X[i][j].strip(",").split(",")).astype(np.float32)/255) for i in range(len(X))\
                                                                                for j in range(len(X[i]))]
    return X

aux = load_testset_genuines()
dp2.x_train = np.append(dp2.x_train,np.asarray(aux).reshape(-1,136,80,1),axis=0)
dp2.y_train = np.append(dp2.y_train,np.ones(np.asarray(aux).shape[0]),axis=0)

idx6 = np.arange(dp2.x_train.shape[0])
np.random.shuffle(idx6)
trn6 = idx6[:((int)(dp2.x_train.shape[0] * 0.7))]  # 70% training
val6 = idx6[((int)(dp2.x_train.shape[0] * 0.7)):]  # 30% validation

model6 = mlp_convnet(dp2.x_train.reshape(-1,136,80,1)[0].shape)
model6.create()
model6.model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics = ['mae','acc'])

history6 = LossHistory()

loss = model6.model.fit(dp2.x_train[trn6],dp2.y_train[trn6],
                    callbacks=[EarlyStopping_byvalue(monitor='val_mean_absolute_error', value=0.2, verbose=1),history6],
                    epochs = 50,
                    batch_size = 100,
                    validation_data = (dp2.x_train[val6],dp2.y_train[val6]),
                    verbose = False)

print("MLP+convnet | Accuracy with independent test set: {0:2.4f}".format(accuracy_score(dp2.y_test,np.where(model6.model.predict(dp2.x_test)>0.9,1,0))))


"""SVM Individual signatures classification """

class svm_convnet():
    def __init__(self,sh):
        self.sh = sh

    def create(self):
        self.model = self.convnet_features(self.sh)

    def convnet_features(self,sh):
        img_in = Input(shape = sh, name = 'FeatureNet_ImageInput')
        n_layer = img_in
        for i in range(2):
            n_layer = Conv2D(8*2**i, kernel_size = (1,1), activation = 'linear')(n_layer)
            n_layer = BatchNormalization()(n_layer)
            n_layer = Activation('relu')(n_layer)
            n_layer = Conv2D(16*2**i, kernel_size = (1,1), activation = 'linear')(n_layer)
            n_layer = BatchNormalization()(n_layer)
            n_layer = Activation('relu')(n_layer)
            n_layer = MaxPool2D((2,2))(n_layer)
        n_layer = Flatten()(n_layer)
        n_layer = Dense(32, activation = 'linear')(n_layer)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        n_layer = Dense(1, activation = 'linear')(n_layer)
        feature_model = Model(inputs = [img_in], outputs = [n_layer], name = 'FeatureGenerationModel')
        #feature_model.summary()
        return feature_model


"""SVM+convnet 1st approach: Mix datasets and randomely split for traning, validation and testing"""

idx7 = np.arange(x_all.shape[0])
np.random.shuffle(idx7)
trn7 = idx7[:((int)(x_all.shape[0]*0.6))] #60% training
val7 = idx7[((int)(x_all.shape[0]*0.6)):((int)(x_all.shape[0]*0.8))] #20% validation
tst7 = idx7[((int)(x_all.shape[0]*0.8)):] #20% testing

model7 = svm_convnet(x_all.reshape(-1,136,80,1)[0].shape)
model7.create()
model7.model.compile(optimizer='nadam', loss='hinge', metrics = ['mae','acc'])

history7 = LossHistory()

loss = model7.model.fit(x_all[trn7],y_all[trn7],
                    callbacks=[EarlyStopping_byvalue(monitor='val_mean_absolute_error', value=0.2, verbose=1),history7],
                    epochs = 50,
                    batch_size = 100,
                    validation_data = (x_all[val7],y_all[val7]),
                    verbose = True)

print("SVM+convnet | Accuracy with a mix of sets and further splitting: {0:2.4f}".format(accuracy_score(y_all[tst7],np.where(model7.model.predict(x_all[tst7])>0.9,1,0))))


h_losses = [history1,
            history2,
            history3,
            history4,
            history5,
            history6]
            #history7,
            #history8,
            #history9]

labels = ['Siamese+convnet sets mixture val loss',
          'Siamese+convnet indpendent test set val loss',
          'Siamese+convnet with genuines added for taining val loss',
          'MLP+convnet sets mixture val loss',
          'MLP+convnet indpendent test set val loss',
          'MLP+convnet with genuines added for taining val loss']
          #'SVM+convnet sets mixture val loss',
          #'SVM+convnet indpendent test set val loss',
          #'SVM+convnet with genuines added for taining val loss',]

def plot_progression(h_losses,labels):
    plt.figure()
    for i,h in enumerate(h_losses):
        plt.plot(np.asarray(h.losses_val),linewidth=2,label=labels[i]) 
    plt.title("Models validation losses progressions")
    plt.xlabel("Epoch")
    plt.ylabel("Total error")
    plt.savefig('losses.png')

plot_progression(h_losses,labels)