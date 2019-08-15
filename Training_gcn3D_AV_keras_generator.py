from time import time
from datetime import date
from utils import ArteryVein_data
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from utils import TrainValTensorBoard
import numpy as np
from utils.models import  Av_CNN3D_model, Av_CNN_GCN_model, Av_CNN_GCN_trans_model
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # Select a gpu card
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

DataType = 'CTPA'
orient = 'Orient'

path_data = './data/{}/'.format(orient)

Cases = [
'/DemoRightUpper05{}/right/'.format(DataType), #0
]
Case_train = [Cases[0] ]
Case_val= [Cases[0] ]

batchSz = 128
learning_rate = 1e-3
epochs = 100
Num_classes = 2
Num_neighbors = 2
dp = 0.5
nornalization = 'pnormalization'
[pSx, pSy, pSz] = [32, 32, 5]


tf.set_random_seed(1234)

# model_name = 'AV_CNN3D'
model_name = 'AV_CNN_GCN'
# model_name = 'AV_CNN_GCN_trans'
Tdate = '2019-06-07'
if model_name == 'AV_CNN3D':
    timeR = '{}/{}/{}_ep{}_bt{}_lr{}_dp{}_avs{}'.format(DataType,nornalization,model_name,epochs, batchSz,learning_rate,dp, Tdate)
    usingNeighbors = False
    model = Av_CNN3D_model(patch_sz=(pSx, pSy, pSz), droupout_rate= dp, number_class=Num_classes, )
elif model_name=='AV_CNN_GCN':
    timeR = '{}/{}/{}_ep{}_bt{}_lr{}_dp{}_avs{}'.format(DataType,nornalization, model_name, epochs, batchSz,
                                                                learning_rate, dp, Tdate)
    usingNeighbors = True
    model = Av_CNN_GCN_model(patch_sz=(pSx, pSy, pSz), droupout_rate= dp, number_class=Num_classes, number_neighbors=Num_neighbors)
elif model_name=='AV_CNN_GCN_trans':
    timeR = '{}/{}/{}_ep{}_bt{}_lr{}_dp{}_avs{}'.format(DataType,nornalization, model_name, epochs, batchSz,
                                                                learning_rate, dp, Tdate)
    usingNeighbors = True
    model = Av_CNN_GCN_trans_model(patch_sz=(pSx, pSy, pSz), droupout_rate= dp, number_class=Num_classes, number_neighbors=Num_neighbors)


optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
loss = keras.losses.categorical_crossentropy

cfilepath = "./logs/{}/models_best.h5".format(timeR)
if os.path.isfile(cfilepath):
    print("Resumed modelś weights from {}".format(cfilepath))
    model.load_weights(cfilepath)

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
callbacks = [
    TrainValTensorBoard.TrainValTensorBoard( log_dir="./logs/{}/summary/".format(timeR) ),
    ModelCheckpoint(cfilepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
]


trainGraphes = ArteryVein_data.load_data(path=path_data, case_name=Case_train, Num_neighbor=Num_neighbors, nornalization=nornalization)
valGraphes = ArteryVein_data.load_data(path=path_data, case_name=Case_val, Num_neighbor=Num_neighbors, nornalization=nornalization)

Num_samples = 0
for graph in trainGraphes:
    Num_samples += graph.num_nodes

Num_samples_val = 0
for graph in valGraphes:
    Num_samples_val += graph.num_nodes

Start_time = time()
print('Start training! start time is {}'.format(Start_time))
model.fit_generator(generator=ArteryVein_data.DataGenerator(trainGraphes,batch_sz=batchSz, withNeighbor=usingNeighbors),
                    epochs=epochs, callbacks=callbacks, verbose=1,steps_per_epoch=np.ceil(Num_samples/batchSz),
                    validation_data=ArteryVein_data.DataGenerator(valGraphes,batch_sz=batchSz,withNeighbor=usingNeighbors),
                    validation_steps=np.ceil(Num_samples_val/batchSz)
                    )
End_time = time()
print('Finishing training! Total traing time is {} in sedonds'.format(End_time - Start_time) )



