""" This is the code for quantitative phase retrieval via deep learning
Author:  Ayan Sinha, Justin Lee, Shuai Li and George Barbastathis
Affliation: Department of Mechanical Engineering, MIT
Date: 2017-6-22
"""


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import numpy as np
#import pylab as plt
import h5py
import tensorflow as tf
import math
import net_conv_deconv_Type1_optica
import keras.backend as K
from keras import optimizers 
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback, CSVLogger
from keras.layers import Permute
from keras.models import load_model
#from keras.utils.visualize_util import plot
from keras.utils.io_utils import HDF5Matrix




def run_experiment(case_data, typnet="res", u_net=True, sh_add=4, ty_2=True,dataname='TIE_iter_face.mat',datanamePh='F:\\Optica_data\\',dataname2='TIE_iter_face.mat',fileend1=42500,filestr=42000,fileend2=45000,imginput=True,normz=1):

        # learning rate schedule

        phase_matlin=HDF5Matrix('ph_val_linp.mat', 'ph_val_lin',0, 256)
        phase_matlin=np.array(phase_matlin)
        #print(phase_matlin.shape)
        
        phase_mattri=HDF5Matrix('ph_val_trip.mat', 'ph_val_tri',0, 256)
        phase_mattri=np.array(phase_mattri)
        #print(phase_mattri.shape)
         
        def step_decay(epoch):
                initial_lrate = 0.001
                drop = 0.5
                epochs_drop = 5.0
                lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
                return lrate

        class LossHistory():
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))


        def normalize_data1(x):
                #x=phase_matlin[np.array(x)]
                x=x*100
                return x

        def normalize_data2(x):
                #x=phase_mattri[np.array(x)]
                x=x*100
                return x

        if case_data==1:  
                file_base="Face_"
        if case_data==2:               
                file_base="Digits_"
        if case_data==3:          
                file_base="Images_"

        if normz==0:
                datanameOp=dataname
                datanameOp2=dataname2
        elif normz==1:
                datanameOp=datanamePh+'Org_'+file_base+'lin.mat'
                datanameOp2=datanamePh+'Org_Test_lin.mat'
        elif normz==2:
                datanameOp=datanamePh+'Org_'+file_base+"tri.mat"
                datanameOp2=datanamePh+'Org_Test_tri.mat'
 
        
                



                

        
        if normz==0:
                output_mat = HDF5Matrix(datanameOp, 'OrgMat',0, fileend1, normalizer=None)
                output_mat_test = HDF5Matrix(datanameOp2, 'OrgMat',filestr, fileend2, normalizer=None)
        elif normz==1:
                output_mat = HDF5Matrix(datanameOp, 'OrgMat',0, fileend1, normalizer=normalize_data1)
                output_mat_test = HDF5Matrix(datanameOp2, 'OrgMat',filestr, fileend2, normalizer=normalize_data1)
        elif normz==2:
                output_mat = HDF5Matrix(datanameOp, 'OrgMat',0, fileend1, normalizer=normalize_data2)
                output_mat_test = HDF5Matrix(datanameOp2, 'OrgMat',filestr, fileend2, normalizer=normalize_data2)
                
        input_mat = HDF5Matrix(dataname, 'CamMat',0, fileend1, normalizer=None)
        input_mat_test = HDF5Matrix(dataname2, 'CamMat',filestr, fileend2, normalizer=None)

              
     


        #test
        


        shape_aux= (1,1024,1024)

        if typnet=="plain":
                model = net_conv_deconv_Type1_optica.ResnetBuilder.build_plainnet(shape_aux, type2=ty_2, shape_add=sh_add, unet=u_net, imginput=imginput)
                batch_sz=3
        else:
                if normz==0:
                        model = net_conv_deconv_Type1_optica.ResnetBuilder.build_resnet(shape_aux, type2=ty_2, shape_add=sh_add, unet=u_net, imginput=imginput)
                elif normz==1:
                        model = net_conv_deconv_Type1_optica.ResnetBuilder.build_resnet(shape_aux, type2=ty_2, shape_add=sh_add, unet=u_net, imginput=imginput)
                elif normz==2:
                        model = net_conv_deconv_Type1_optica.ResnetBuilder.build_resnet(shape_aux, type2=ty_2, shape_add=sh_add, unet=u_net, imginput=imginput)
                batch_sz=3

        if not os.path.exists(dataname[:-4]):
                os.makedirs(dataname[:-4])
        fileall= dataname[:-4]+"\\smOptica_type1_"+file_base+typnet+"_type2"+str(ty_2)+"_shapeadd"+str(sh_add)+"_unet"+ str(u_net)+"_imageinpit"+ str(imginput)+"_normz"+str(normz)
        filepath=fileall+"_weights_type2.{epoch:02d}.hdf5"       
        filename=fileall+'training.log'



        adam=optimizers.Adam(clipvalue=1)
        model.compile(loss='mean_absolute_error', optimizer=adam)
        int_eph=0

        if dataname=='F:\Optica_data1\Imageslinear_optica_distance90000.mat':
                model.load_weights('F:\Optica_data1\Imageslinear_optica_distance90000\Images_res_type2False_shapeadd512_unetTrue_imageinpitTrue_normz1_weights_type2.02.hdf5')
                int_eph=3
        

        
        csv_logger = CSVLogger(filename)
        lrate = LearningRateScheduler(step_decay)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

        callbacks_list = [lrate, checkpoint,csv_logger]

        
        if ty_2 and not imginput:
                model.fit([input_mat,idx_mat], output_mat, batch_size=batch_sz,
                nb_epoch=20, validation_data=([input_mat_test,idx_mat_test], output_mat_test), callbacks=callbacks_list,  shuffle="batch")
        elif ty_2 and imginput:
                model.fit([input_mat,holo_mat], output_mat, batch_size=batch_sz,
                nb_epoch=20, validation_data=([input_mat_test,holo_mat_test], output_mat_test), callbacks=callbacks_list,  shuffle="batch")
        else:
                model.fit(input_mat, output_mat, batch_size=batch_sz,
                nb_epoch=20, validation_data=(input_mat_test, output_mat_test), callbacks=callbacks_list,  shuffle="batch",initial_epoch=int_eph)


if __name__ == '__main__':

    base_folder='F:\\Optica_data2\\'
    subname='tri'
    dist=90000
    normtype=2

    #run_experiment(case_data=1, typnet="res", u_net=True, sh_add=512, ty_2=False,dataname=base_folder+'Faces'+subname+'_optica_distance'+str(dist)+'.mat',datanamePh=base_folder,
    #               dataname2=base_folder+'Test'+subname+'_optica_distance'+str(dist)+'.mat',fileend1=10000,filestr=250, fileend2=350,imginput=True,normz=normtype)
    run_experiment(case_data=3, typnet="res", u_net=True, sh_add=512, ty_2=False,dataname=base_folder+'Images'+subname+'_optica_distance'+str(dist)+'.mat',datanamePh=base_folder,
                   dataname2=base_folder+'Test'+subname+'_optica_distance'+str(dist)+'.mat',fileend1=10000,filestr=350, fileend2=450,imginput=True,normz=normtype)






    
    


