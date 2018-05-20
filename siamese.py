import numpy as np
from keras.regularizers import l2
from keras import backend as K
from keras import callbacks
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout
from keras.layers import concatenate
import matplotlib.pyplot as plt

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
        X = []
        with open(file,"r") as f:
            lines = f.readlines()
            for l,line in enumerate(lines):
                blocks = line.strip(" :\n").split(";")
                X.append([blocks[0].strip().split(":"),blocks[1].strip().split(":")])
        X = [[[(np.asarray(X[i][j][k].split()).astype(np.float32)/255)\
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

prueba = ["traingset.txt","traingset.txt"]
dp = batches(prueba)
dp.sets()
for i,x in enumerate(dp.x_train):
    if i == 0:
        ptr = x.reshape(-1,2,136,80)
        ytr = dp.y_train[i]
    else:
        ptr = np.append(ptr,x.reshape(-1,2,136,80),axis=0)
        ytr = np.append(ytr, dp.y_train[i])

##Image verification
#print(ptr[0,0].shape)
#plt.figure()
#plt.imshow(np.transpose(ptr[0,0]),cmap='gray')
#plt.show()
#exit()

"""shuffle data and split training and validaton"""
rdm_index = np.arange(ptr.shape[0])
np.random.shuffle(rdm_index)
tr = rdm_index[:((int)(ptr.shape[0]*0.7))]
val = rdm_index[((int)(ptr.shape[0]*0.7)):]

feature_model = covnet_features(ptr[0,0,0:].shape)
sim_model = similarity(ptr[0,0,0:].shape,feature_model)
sim_model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics = ['mae','acc'])

loss = sim_model.fit([ptr[tr,0],ptr[tr,1]],ytr[tr],
                    callbacks=[EarlyStopping_byvalue(monitor='val_mean_absolute_error', value=0.01, verbose=1)],
                    epochs = 50,
                    batch_size = 100,
                    validation_data = ([ptr[val,0],ptr[val,1]],ytr[val]),
                    verbose = True)



