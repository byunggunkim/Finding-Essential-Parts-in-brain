from tensorflow import keras
from dataset import *
import models
from result import *
import numpy as np
import json

def main(model_name, dataset_path, training_model, roi_idx, drop_rate, seed_idx):

    site_list = ['nyu','peking', 'ohsu', 'kki', 'neuroimage']

    # Parameter list
    model_idx = str(len(roi_idx)) + model_name
    batch_size = 32
    l2 = 0.0005
    epoch = 200
    optim = 'adam'
    lr = 1e-4
    kernal_list = [3,3,3,3]
    filter_list = [32,64,96,96]
    cnn_activation = 'leakyrelu'
    activation_param = 0.1
    attention_hidden = 64
    lstm_hidden = 128
    dense_hidden = 128
    dense_activation = 'leakyrelu'
    padding = 'valid'
    drop_rate = drop_rate
    seed = seed_idx
    tf.random.set_seed(seed)
    init = 'glorot_normal'
    
    hyperparameters = {
    'model_name':model_name,
    'roi_idx':roi_idx,
    'batch_size':batch_size,
    "l2":l2,
    'epoch':epoch,
    'optim':optim,
    'lr':lr,
    'dropout':drop_rate,
    'activation_param':activation_param,
    'kernal_list':kernal_list,
    'filter_list':filter_list,
    'padding':padding,
    'cnn_activation':cnn_activation,
    'lstm_hidden':lstm_hidden,
    'attention_hidden':attention_hidden,
    'dense_activation':dense_activation,
    'initialize':init,
    'seed':seed,
    }

    # log list
    train_history = list()
    validbest_history = list()

    # Make hyper_param directory
    hyperparam_dir = './hyparameters/'+ model_name + '/'
    os.makedirs(hyperparam_dir, exist_ok=True)
    with open(hyperparam_dir+model_idx+'.json',"w") as json_file:
        json.dump(hyperparameters, json_file,indent=4)

    for site in site_list:
        # save model
        checkpoint_path = './checkpoint/'+model_name+'/'+ model_idx +'/'+ site +'/cp.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    monitor='val_accuracy',
                                                    save_weights_only=True,
                                                    mode='max',
                                                    save_best_only=True)

        # dataset Preprocessing
        trainset, validset, valid_eval = make_train_testset(dataset_path, site)
        balanced_batch = balanced_resampling(trainset, batch_size)

        # Make model with hyperparameters
        if training_model == 'sccnn_attention':
            print('sccnn + attention Model')
            model = models.SC_CNN_Attention_roi(kernal_list, filter_list, padding, cnn_activation, activation_param, lstm_hidden, attention_hidden, dense_hidden, init, l2, drop_rate, roi_idx)
        elif training_model == 'sccnn_lstm':
            print('sccnn + lstm Model')
            model = models.SC_CNN_LSTM_roi(kernal_list, filter_list, padding, cnn_activation, activation_param, lstm_hidden, attention_hidden, dense_hidden, init, l2, drop_rate, roi_idx)
        elif training_model == 'acrnn':
            print('scrnn + acrnn Model')
            model = models.SC_CNN_acrnn_roi(kernal_list, filter_list, padding, cnn_activation, activation_param, lstm_hidden, attention_hidden, dense_hidden, init, l2, drop_rate, roi_idx)
        elif training_model == 'adrnn':
            print('scrnn + adrnn Model')
            model = models.SC_CNN_adrnn_roi(kernal_list, filter_list, padding, cnn_activation, activation_param, lstm_hidden, attention_hidden, dense_hidden, init, l2, drop_rate, roi_idx)
        elif training_model == 'asrnn':
            print('scrnn + asrnn Model')
            model = models.SC_CNN_asrnn_roi(kernal_list, filter_list, padding, cnn_activation, activation_param, lstm_hidden, attention_hidden, dense_hidden, init, l2, drop_rate, roi_idx)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss=tf.keras.losses.BinaryCrossentropy(True),
                    metrics=['accuracy', tf.keras.metrics.AUC()])

        history = model.fit(balanced_batch, validation_data=validset, epochs=epoch, callbacks=[checkpoint])

        # validation best model evaluation
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        validbest = model.evaluate(valid_eval)
        
        # recording
        train_history.append(history.history)
        validbest_history.append(validbest)
    
        # Clear all trained memorys in GPU
        tf.keras.backend.clear_session()
        del model, trainset, testset, balanced_batch

    # Saving results
    result_dir = './results/'+ model_name + '/'
    os.makedirs(result_dir, exist_ok=True)

    with open(result_dir+model_idx+'.pickle', 'wb') as f:
        pickle.dump(train_history, f)
        pickle.dump(validbest_history, f)

    # ploting results
    result_plt(model_name, str(len(roi_idx)), train_history, validbest_history)


if __name__ == "__main__":

    # model parameters
    model_name = 'roi_test'
    dataset_path = './preprocessed/NIAK_each_site'
    training_model = 'sccnn_attention'
    dropout_rate = [0.2]
    
    print('Traning '+ model_name)

    for idx in range(116):
        main(model_name, dataset_path=dataset_path, training_model=training_model, roi_idx=idx, drop_rate=dropout_rate, seed_idx=None)