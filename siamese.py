import numpy as np
from keras.regularizers import l2
from keras import backend as K
from keras import callbacks
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout
from keras.layers import concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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
   
class batches():
    def __init__(self,files):
        self.files = files
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.d_train = []
        self.d_test = []

    def sets(self):
        for i,f in enumerate(self.files):
            if i==0:
                self.x_train, self.y_train, self.d_train = self.load_data(f)
            else:
                self.x_test, self.y_test, self.d_test = self.load_data(f)

    def load_data(self,file):
        print(file)
        X = []
        with open(file,"r") as f:
            lines = f.readlines()
            for l,line in enumerate(lines):
                blocks = line.strip(",:\n").split(",;")
                X.append([blocks[0].strip().split(":"),blocks[1].strip().split(":")])
        X = [[[(np.asarray(X[i][j][k].strip(",").split(",")).astype(np.float32)/255)\
                                    for k in range(len(X[i][j])) if len(X[i][j][k])>0]\
                                    for j in range(len(X[i]))]\
                                    for i in range(len(X))]
        return self.create_pairs(X)

    def create_pairs(self,X):
        x_pairs = []
        y_pairs = []
        d_pairs = []
        for i in range(len(X)):    # number of authors
            right_pairs = []    # genuine vs genuine
            fake_pairs = []     # genuine vs forgerie
            d_right = []    # distance between genuines
            d_fake = []     # distance between forgeries
            y_right = []    # ones (label)
            y_fake = []     # zeros (label)
            for i0 in range(len(X[i][0])):
                for j in range(len(X[i][0])):
                    if(i0!=j):
                        right_pairs.append([X[i][0][i0],X[i][0][j]]) # genuines pairs
                        d_right.append(np.sum(np.abs(X[i][0][i0]-X[i][0][j]))/X[i][0][i0].size)
                        y_right.append(1)
                for k in range(len(X[i][1])):
                        fake_pairs.append([X[i][0][i0],X[i][1][k]]) # forgeries pairs
                        d_fake.append(np.sum(np.abs(X[i][0][i0]-X[i][1][k]))/X[i][0][i0].size)
                        y_fake.append(0)
            x_pairs += [np.concatenate((np.asarray(right_pairs),np.asarray(fake_pairs)),axis=0)]
            y_pairs += [np.concatenate((y_right,y_fake),axis=0)]
            d_pairs += [np.concatenate((d_right,d_fake),axis=0)]
        return np.asarray(x_pairs),np.asarray(y_pairs),np.asarray(d_pairs)

def covnet_features(sh):
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
    feature_model.summary()
    return feature_model

def similarity(sh,feature_model):
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
    similarity_model.summary()
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
            pairs = xi.reshape(-1,2,136,80)
            labels = yi
            i+=1
        else:
            pairs = np.append(pairs,xi.reshape(-1,2,136,80),axis=0)
            labels = np.append(labels, yi)
    print(pairs.shape,labels.shape)
    pairs_sets.append(pairs)
    labels_sets.append(labels)
print(len(pairs_sets),len(labels_sets))
exit()

##Image verification
#print(ptr[0,0].shape)
#plt.figure()
#plt.imshow(np.transpose(ptr[0,0]),cmap='gray')
#plt.show()
#exit()
"""1st approach: Mix datasets and randomely split for traning, validation and testing"""
full_set_pairs = np.append(pairs_sets[0],pairs_sets[1],axis=0)
full_set_labels = np.append(labels_sets[0],labels_sets[1],axis=0)
idx = np.arange(full_set_labels.shape[0])
np.random.shuffle(idx)
trn1 = idx[:((int)(full_set_pairs[0]*0.6))] #60% training
val1 = idx[((int)(full_set_pairs[0]*0.6)):((int)(full_set_pairs[0]*0.8))] #20% validation
tst1 = idx[((int)(full_set_pairs[0]*0.8)):] #20% testing

feature_model = covnet_features(full_set_pairs[0,0,0:].shape)
sim_model = similarity(full_set_pairs[0,0,0:].shape,feature_model)
sim_model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics = ['mae','acc'])

loss = sim_model.fit([full_set_pairs[trn1,0],full_set_pairs[trn1,1]],full_set_labels[trn1],
                    callbacks=[EarlyStopping_byvalue(monitor='val_mean_absolute_error', value=0.01, verbose=1)],
                    epochs = 50,
                    batch_size = 100,
                    validation_data = ([full_set_pairs[val1,0],full_set_pairs[val1,1]],full_set_labels[val1]),
                    verbose = True)

print("Accuracy with a mix of sets and further splitting: {0:2.4f}".format(accuracy_score(full_set_labels[tst1],sim_model.predict([full_set_pairs[tst1,0],full_set_pairs[tst1,1]]))))

"""2nd approach: Testing with test dataset"""
idx2 = np.arange(ptr.shape[0])
np.random.shuffle(idx2)
trn2 = idx2[:((int)(pairs_sets[0].shape[0]*0.7))]   # 70% training
val2 = idx2[((int)(pairs_sets[0].shape[0]*0.7)):]   # 30% validation

feature_model = covnet_features(pairs_sets[0][0,0,0:].shape)
sim_model = similarity(pairs_sets[0][0,0,0:].shape,feature_model)
sim_model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics = ['mae','acc'])

loss = sim_model.fit([pairs_sets[0][trn2,0],pairs_sets[0][trn2,1]],labels_sets[0][trn2],
                    callbacks=[EarlyStopping_byvalue(monitor='val_mean_absolute_error', value=0.01, verbose=1)],
                    epochs = 50,
                    batch_size = 100,
                    validation_data = ([pairs_sets[0][val2,0],pairs_sets[0][val2,1]],labels_sets[0][val2]),
                    verbose = True)

print("Accuracy with independent testing set: {0:2.4f}".format(accuracy_score(labels_sets[1],sim_model.predict([pairs_sets[1][:,0],pairs_sets[1][:,1]]))))



