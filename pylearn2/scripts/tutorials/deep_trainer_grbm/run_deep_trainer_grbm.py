#!/usr/bin/env python
"""
See readme.txt

A small example of how to glue shining features of pylearn2 together
to train models layer by layer.
"""

MAX_EPOCHS_UNSUPERVISED_GRBM = 100
MAX_EPOCHS_UNSUPERVISED_RBM = 100
MAX_EPOCHS_SUPERVISED = 100

BATCH_SIZE = 50

USE_UNSUPER = True
USE_SCALE_ALL = True

from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.mlp import Default
from pylearn2.models.rbm import GaussianBinaryRBM
from pylearn2.models.rbm import RBM
from pylearn2.models.softmax_regression import SoftmaxRegression
from pylearn2.training_algorithms.sgd import SGD, LinearDecayOverEpoch
from pylearn2.termination_criteria import EpochCounter, And, Or, MonitorBased
from pylearn2.datasets import cifar10
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1, GRBM_Original
from pylearn2.base import StackedBlocks
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.train import Train
import pylearn2.utils.serial as serial
import os
from optparse import OptionParser
from theano import tensor as T
from theano import function
import numpy as np
import numpy.random
from pylearn2.space import CompositeSpace
from pylearn2.utils.data_specs import DataSpecsMapping

from pylearn2.datasets.csv_dataset import CSVDataset
from pylearn2.datasets.preprocessing import GlobalContrastNormalization, StandardizeAll, ScaleAll
from pylearn2.costs.dbm import BaseCD, VariationalPCD
from pylearn2.models.dbm import BinaryVectorMaxPool
from pylearn2.models.DBNSampler import DBNSampler
from pylearn2.models.dbm import DBM
from pylearn2.costs.ebm_estimation import RBM_Cost
from pylearn2.models.mlp import Tanh, Linear, MLP
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
import pylab as pl
from pylearn2.utils import safe_zip
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor, Momentum

def get_dataset_tfp(name):

    trainset = load_tfp_dataset(name + '-train.csv')
    validset = load_tfp_dataset(name + '-valid.csv')
    testset = load_tfp_dataset(name + '-test.csv')

    print 'trainset before:{}'.format(trainset.X[0:10,:])

    if USE_SCALE_ALL:

        #find max and mins of X and y
        max_X = np.maximum(np.maximum(trainset.X.max(axis=0), validset.X.max(axis=0)),
                                                              testset.X.max(axis=0))
        min_X = np.minimum(np.minimum(trainset.X.min(axis=0), validset.X.min(axis=0)),
                                                              testset.X.min(axis=0))

        max_y = np.maximum(np.maximum(trainset.y.max(axis=0), validset.y.max(axis=0)),
                                                              testset.y.max(axis=0))
        min_y = np.minimum(np.minimum(trainset.y.min(axis=0), validset.y.min(axis=0)),
                                                              testset.y.min(axis=0))

        preprocessor = ScaleAll(min_X = min_X, max_X = max_X, min_y = min_y, max_y = max_y)
        trainset.apply_preprocessor(preprocessor, can_fit=True) 
        validset.apply_preprocessor(preprocessor, can_fit=True) 
        testset.apply_preprocessor(preprocessor, can_fit=True)
    else:
        #preprocessor = ScaleAll(min_X = min_X*0.9, max_X = max_X*1.1, min_y = min_y*0.9, max_y = max_y*1.1)
        trainset.apply_preprocessor(StandardizeAll(), can_fit=True) 
        validset.apply_preprocessor(StandardizeAll(), can_fit=True) 
        testset.apply_preprocessor(StandardizeAll(), can_fit=True)    

    print 'trainset after:{}'.format(trainset.X[0:10,:])
    
    return (trainset, validset, testset)

def load_tfp_dataset(name):
    full_path = os.path.join('${PYLEARN2_DATA_PATH}', name)
    print 'full_path:{}'.format(full_path)
    dataset = CSVDataset(path=full_path, expect_labels=True,
                         reverse_order_of_columns=True, expect_headers=False)

    return dataset

def get_grbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        "irange" : 0.05,
        #"energy_function_class" : GRBM_Type_1,
        "energy_function_class" : GRBM_Original,
        "learn_sigma" : True,
        "init_sigma" : .4,
        "init_bias_hid" : 0,
        "mean_vis" : False,
        "sigma_lr_scale" : 1e-8
        }

    return GaussianBinaryRBM(**config)

def get_layer_trainer_grbm(layer, trainset,save_file='grbm.pkl'):
    train_algo = SGD(
        learning_rate = 1e-5,
        batch_size =  BATCH_SIZE,
        #"batches_per_iter" : 2000,
        monitoring_batches =  10,
        monitoring_dataset =  trainset,
        #cost = SMD(corruptor=GaussianCorruptor(stdev=0.4)),
        cost = RBM_Cost(),
        termination_criterion =  EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED_GRBM),
        )
    model = layer
    #extensions = [MonitorBasedLRAdjuster()]
    extensions = []
    return Train(model = model, algorithm = train_algo,
                 save_path=save_file, save_freq=10,
                 extensions = extensions, dataset = trainset)

def get_rbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        "irange" : 0.05,
        "init_bias_hid" : 0.
        }

    return RBM(**config)

def get_layer_trainer_rbm(layer, trainset,save_file='rbm.pkl'):
    
    train_algo = SGD(
        learning_rate = 1e-4,
        batch_size =  BATCH_SIZE,
        #"batches_per_iter" : 2000,
        monitoring_batches =  10,
        monitoring_dataset =  trainset,
        cost = RBM_Cost(),
        termination_criterion = EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED_RBM),
        )
    model = layer
    #extensions = [MonitorBasedLRAdjuster()]
    extensions = []
    return Train(model = model, algorithm = train_algo,
                 save_path=save_file, save_freq=10,
                 extensions = extensions, dataset = trainset)

def get_tanh(dim, layer_name):    
    layer = Tanh(dim=dim, layer_name=layer_name, irange=0.02)
    return layer

def get_linear(dim, layer_name):
    layer = Linear(dim=dim, layer_name=layer_name, irange=0.02)
    return layer

def get_mlp(nvis, layers):
    mlp = MLP(layers=layers, batch_size=BATCH_SIZE, nvis=nvis)
    return mlp

def get_mlp_trainer(mlp, dataset):
    # configs on sgd
    trainset, validset, testset = dataset

    data = {
        'train': trainset,
        'valid': validset,
        'test': testset
    }

    print 'trainset shape:{}'.format(trainset.get_design_matrix().shape)
    print 'validset shape:{}'.format(validset.get_design_matrix().shape)
    print 'testset shape:{}'.format(testset.get_design_matrix().shape)

    termination_criterion = Or([
                        EpochCounter(max_epochs=MAX_EPOCHS_SUPERVISED),
                        MonitorBased(prop_decrease=0.02, N=10,
                            channel_name='valid_objective')
                ])
    config = {'learning_rate': 0.05,
              'cost' : Default(),
              'batch_size': BATCH_SIZE,
              'monitoring_batches': 1,
              'monitoring_dataset': data,
              'termination_criterion': termination_criterion,
              'update_callbacks': None
              }

    train_algo = SGD(**config)
    model = mlp
    extensions = [MonitorBasedSaveBest(channel_name='valid_objective',
                                       save_path='mlpBS.pkl')]
    return Train(model = model,
            save_path='mlp.pkl',
            save_freq=10,
            dataset = trainset,
            algorithm = train_algo,
            extensions = extensions)

def get_mlp_dropout_trainer(mlp, dataset):
    # configs on sgd
    trainset, validset, testset = dataset

    data = {
        'train': trainset,
        'valid': validset,
        'test': testset
    }

    print 'trainset shape:{}'.format(trainset.get_design_matrix().shape)
    print 'validset shape:{}'.format(validset.get_design_matrix().shape)
    print 'testset shape:{}'.format(testset.get_design_matrix().shape)

    termination_criterion = Or([
                        EpochCounter(max_epochs=MAX_EPOCHS_SUPERVISED),
                        MonitorBased(prop_decrease=0.02, N=10,
                            channel_name='valid_objective')
                ])
    config = {'learning_rate': 0.03,
              'learning_rule':Momentum(.5),
              'cost':Default(),
              # 'cost':Dropout(input_include_probs={'l1': .8,
              #                                     'l2': .5,
              #                                     'l3': .5,
              #                                     'l4': .5,
              #                                     'l5': .5
              #                                    },
              #                input_scales={'l1': 1.,
              #                              'l2': 1.,
              #                              'l3': 1.,
              #                              'l4': 1.,
              #                              'l5': 1.
              #                             }
              #               ),
              'batch_size': BATCH_SIZE,
              'monitoring_batches': 1,
              'monitoring_dataset': data,
              'termination_criterion': termination_criterion,
              'update_callbacks': None
              }

    train_algo = SGD(**config)
    model = mlp

    watcher = MonitorBasedSaveBest(channel_name='valid_objective',
                                       save_path='mlpBS.pkl')

    velocity = MomentumAdjustor(final_momentum=.6,
                                          start=300,
                                          saturate=500)

    decay = LinearDecayOverEpoch(start=300,
                                 saturate=500,
                                 decay_factor=config['learning_rate']*.1)

    extensions = [watcher, velocity, decay]
    #extensions = [watcher]
    return Train(model = model,
            save_path='mlp.pkl',
            save_freq=1,
            dataset = trainset,
            algorithm = train_algo,
            extensions = extensions)

def main(args=None):
    """
    args is the list of arguments that will be passed to the option parser.
    The default (None) means use sys.argv[1:].
    """
    np.set_printoptions(threshold='nan')
    #raw_input('Tweak process affinity then press any key to continue...')

    parser = OptionParser()
    parser.add_option("-d", "--data", dest="dataset", default="cifar10",
                      help="specify the tfp dataset")
    (options,args) = parser.parse_args(args=args)

    trainset, validset, testset = get_dataset_tfp(options.dataset)
   
    print 'TEST data'
    print trainset.y[0:10,:]
    #raw_input('Press...')

    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]
    n_output = 1

    print ' n_input =  %d, n_output =  %d'%(n_input, n_output)

    # build layers
    structure = [[n_input, 64], [64, 64], [64,32], [32, 1]]

    num_unsuper_layers = len(structure)-1
    
    if USE_UNSUPER:
        unsuper_layers = []
        # layer 0: gaussianRBM
        unsuper_layers.append(get_grbm(structure[0]))   
        #layers n: RBM
        for i in range(1, num_unsuper_layers):
            print 'unsuper_layer_init:{}'.format(i)
            unsuper_layers.append(get_rbm(structure[i]))

        #construct training sets for different layers
        trainset_coll = [ trainset ,
                    TransformerDataset( raw = trainset, transformer = unsuper_layers[0] )]                    
        for i in range(2, num_unsuper_layers):
            trainset_coll.append(TransformerDataset(raw = trainset,
                                         transformer = StackedBlocks( unsuper_layers[0:i])))

        # construct layer trainers
        layer_trainers = []
        layer_trainers.append(get_layer_trainer_grbm(unsuper_layers[0], trainset_coll[0],'grbm_l1.pkl'))
        for i in range(1, num_unsuper_layers):
            layer_trainers.append(get_layer_trainer_rbm(unsuper_layers[i], trainset_coll[i], 'rbm_l{}.pkl'.format(i+1)))
          
        #layer_trainers.append(get_layer_trainer_rbm(unsuper_layers[3], trainset_x[3],'rbm_l4.pkl'))
                
        #unsupervised pretraining
        for i, layer_trainer in enumerate(layer_trainers[0:num_unsuper_layers]):
            print '-----------------------------------'
            print ' Unsupervised training layer %d, %s'%(i, unsuper_layers[i].__class__)
            print '-----------------------------------'
            layer_trainer.main_loop()


    print '\n'
    print '------------------------------------------------------'
    print ' Unsupervised training done! Start supervised training...'
    print '------------------------------------------------------'
    print '\n'

    super_layers = []
    for i in range(0, num_unsuper_layers):
        super_layers.append(get_tanh(structure[i][1], 'l{}'.format(i+1)))
    super_layers.append(get_linear(structure[num_unsuper_layers][1], 'l{}'.format(num_unsuper_layers+1)))

    mlp = get_mlp(n_input, super_layers)    
    
    if USE_UNSUPER:
        for i in range(0, num_unsuper_layers):
            super_layers[i].set_weights(unsuper_layers[i].get_weights())
            super_layers[i].set_biases(unsuper_layers[i].bias_hid.get_value(borrow=False))    
    
    #supervised training
    mlp_trainer = get_mlp_trainer(mlp, (trainset, validset, testset))    
    mlp_trainer.main_loop()

    #testing
    calculate_testset_prediction(mlp, testset)

    mlp_bs = serial.load('mlpBS.pkl')
    calculate_testset_prediction(mlp_bs, testset)


def calculate_testset_prediction(mlp, testset):
        testing_set = testset
        x, y_target_test = testing_set.get_data()

        batch_size = x.shape[0]
        mlp.set_batch_size(batch_size)
        X = mlp.get_input_space().make_batch_theano()
        Y = mlp.fprop(X)

        f = function([X], Y)
        y = []
        x_arg = testing_set.X[:,:]
        if X.ndim > 2:
            x_arg = testing_set.get_topological_view(x_arg)
        y.append(f(x_arg.astype(X.dtype)))

        y_prediction_test = np.concatenate(y)

        print 'y_prediction_test:{}'.format(y_prediction_test[0:10])
        print 'y_target_test:{}'.format(y_target_test[0:10])
        #raw_input('Press...')

        preproc = testing_set.get_preprocessor()
        if isinstance(preproc, StandardizeAll):
            y_pred_test_orig = (y_prediction_test) * (preproc._std_eps + preproc._std_y) + preproc._mean_y
            y_targ_test_orig = (y_target_test) * (preproc._std_eps + preproc._std_y) + preproc._mean_y
        elif isinstance(preproc, ScaleAll):
            y_pred_test_orig = (y_prediction_test - preproc.min_range)/(preproc.max_range - preproc.min_range) \
                                *(preproc.max_y - preproc.min_y) + preproc.min_y
            y_targ_test_orig = (y_target_test - preproc.min_range)/(preproc.max_range - preproc.min_range) \
                                *(preproc.max_y - preproc.min_y) + preproc.min_y

        print 'y_pred_test_orig:{}'.format(y_pred_test_orig[0:10])
        #raw_input('Press...')

        n_test = batch_size
        print 'n_test:{}'.format(n_test)

        rmse_val2 = numpy.sqrt(numpy.sum(numpy.square(y_pred_test_orig - y_targ_test_orig))/n_test)
        mape_val2 = (numpy.sum(numpy.abs((y_pred_test_orig - y_targ_test_orig)/y_targ_test_orig))/n_test) * 100.
        
        print 'rmse_val {}'.format(rmse_val2)
        print 'mape_val {}'.format(mape_val2)
        
        y_pred_big = y_pred_test_orig[y_targ_test_orig >= 2000]
        y_target_big = y_targ_test_orig[y_targ_test_orig >= 2000]

        rmse_val_big = numpy.sqrt(numpy.sum(numpy.square(y_pred_big - y_target_big))/y_target_big.shape[0])
        mape_val_big = (numpy.sum(numpy.abs((y_pred_big - y_target_big)/y_target_big))/y_target_big.shape[0]) * 100.
        print 'y_target_big.shape:{}'.format(y_target_big.shape[0])

        print 'rmse_val_big {}'.format(rmse_val_big)
        print 'mape_val_big {}'.format(mape_val_big)    

        pl.plot(range(n_test), y_pred_test_orig)
        pl.plot(range(n_test), y_targ_test_orig)
        pl.show()

if __name__ == '__main__':
    main()