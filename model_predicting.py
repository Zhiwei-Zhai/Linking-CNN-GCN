from time import time
from utils import ArteryVein_data
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils.models import Av_CNN3D_model, Av_CNN_GCN_model, Av_CNN_GCN_trans_model
# from models_trans import Av_GCN3D_L_trans_model
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


DataType = 'CTPA'
# DataType = 'NCCT'

orient = 'Orient'
#orient = 'no_Orient'

# path_data = '/srv/2-lkeb-17-dl01/zzhai/Data/ArteryVeinData_{}/'.format(orient)
path_data = '/exports/lkeb-hpc/zzhai/Data/Artery_Vein/{}/'.format(orient)

Cases = [
'/The-First-Hospital-of-SYSU02{}/right/'.format(DataType), #0
'/The-First-Hospital-of-SYSU02{}/left/'.format(DataType),
'/The-First-Hospital-of-SYSU03{}/right/'.format(DataType),
'/The-First-Hospital-of-SYSU03{}/left/'.format(DataType),
'/The-First-Hospital-of-SYSU05{}/right/'.format(DataType),  # 4
'/The-First-Hospital-of-SYSU05{}/left/'.format(DataType),
'/The-First-Hospital-of-SYSU07{}/right/'.format(DataType),
'/The-First-Hospital-of-SYSU07{}/left/'.format(DataType),
'/The-First-Hospital-of-SYSU08{}/right/'.format(DataType), # 8
'/The-First-Hospital-of-SYSU08{}/left/'.format(DataType),
'/The-First-Hospital-of-SYSU09{}/right/'.format(DataType),
'/The-First-Hospital-of-SYSU09{}/left/'.format(DataType),
'/The-First-Hospital-of-SYSU10{}/right/'.format(DataType),  # 12
'/The-First-Hospital-of-SYSU10{}/left/'.format(DataType),
'/The-First-Hospital-of-SYSU11{}/right/'.format(DataType),
'/The-First-Hospital-of-SYSU11{}/left/'.format(DataType),
'/The-First-Hospital-of-SYSU12{}/right/'.format(DataType),  # 16
'/The-First-Hospital-of-SYSU12{}/left/'.format(DataType),
'/The-First-Hospital-of-SYSU13{}/right/'.format(DataType),
'/The-First-Hospital-of-SYSU13{}/left/'.format(DataType),
'/The-First-Hospital-of-SYSU14{}/right/'.format(DataType),  #20
'/The-First-Hospital-of-SYSU14{}/left/'.format(DataType)
]
Case_train = Cases[0:16]
Case_val= Cases[16:]

batchSz = 30
Num_classes = 2
[pSx, pSy, pSz] = [32, 32, 5]
Num_neighbors = 2
#nornalization = 'gnormalization'
nornalization = 'pnormalization'


# model_name = 'AV_CNN3D'
# model_name = 'AV_CNN_GCN'
model_name = 'AV_CNN_GCN_trans'
if model_name=='AV_CNN3D':
    model = Av_CNN3D_model(patch_sz=(pSx, pSy, pSz), droupout_rate=0.5, number_class=Num_classes)
    model_h5 = './logs/CTPA/pnormalization/GenAv_CNN3D_L_model_ep100_bt128_lr0.001_dp0.5_avs2019-05-29/models.h5'
    model.load_weights(filepath=model_h5)
    optimizer = keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)
    loss = keras.losses.categorical_crossentropy
    model.compile(loss=loss, optimizer=optimizer)

    for case in Case_val:
        case = [case]
        data = ArteryVein_data.load_data(path=path_data, case_name=case, shuffel=False)
        prediction = model.predict_generator(generator=ArteryVein_data.DataGenerator(data, batchSz, withNeighbor=False),
                                             steps=data[0].num_nodes / batchSz)
        result_file = '{0}{1}predicts_{2}.npy'.format(path_data, case[0], model_name)
        np.save(result_file, prediction[:data[0].num_nodes, ])
        print('Predicting case {} is finished!'.format(case[0]))

elif model_name == 'AV_GCN':
    model = Av_CNN_GCN_model(patch_sz=(pSx, pSy, pSz), droupout_rate= 0.5, number_class=Num_classes, number_neighbors=Num_neighbors)
    model_h5 = './logs/CTPA/pnormalization/GenAV_gcn_L_ep100_bt128_lr0.001_dp0.5_avs2019-06-04/models_best.h5'
    model.load_weights(filepath=model_h5)
    optimizer = keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)
    loss = keras.losses.categorical_crossentropy
    model.compile(loss=loss, optimizer=optimizer)

    for case in Case_val:
        case = [case]
        data = ArteryVein_data.load_data(path=path_data, case_name=case, Num_neighbor=Num_neighbors, nornalization=nornalization)
        prediction = model.predict_generator(generator=ArteryVein_data.DataGenerator(data, batchSz, withNeighbor=True),
                                             steps= data[0].num_nodes/batchSz )
        result_file = '{0}{1}predicts_{2}.npy'.format(path_data, case[0], model_name)
        np.save(result_file, prediction[:data[0].num_nodes,])
        print('Predicting case {} is finished!'.format(case[0]))

elif model_name == 'AV_GCN_trans':
    model = Av_CNN_GCN_trans_model(patch_sz=(pSx, pSy, pSz), droupout_rate=0.5, number_class=Num_classes,
                             number_neighbors=Num_neighbors)
    model_h5 = ''
    model.load_weights(filepath=model_h5)




