from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as KB
from carbontracker.tracker import CarbonTracker

import numpy as np
import matplotlib.pyplot as plt

import os
import time

def CNN(input_shape=(28, 28, 1), class_num=10):
    model = Sequential()
    model.add(Conv2D(6, (5, 5),
                     input_shape=input_shape,
                     strides=(1, 1),
                     padding='valid',
                     data_format='channels_last',
                     activation='relu',))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(16, (5, 5),
                     strides=(1, 1),
                     padding='valid',
                     data_format='channels_last',
                     activation='relu',))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten(data_format='channels_last'))
    model.add(Dense(168, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(class_num, activation='softmax',))
    # model.summary()
    sgd = SGD(learning_rate=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def generate_data(K, frac_norm):
    dataset = mnist.load_data()
    (X_train, Y_train), (X_test, Y_test) = dataset
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test = to_categorical(Y_test, num_classes=10)

    X_train_client = {}
    Y_train_client = {}
    start = 0
    for i in range(K):
        X_train_client[i] = X_train[start:start+frac[i]]
        Y_train_client[i] = Y_train[start:start+frac[i]]
        start = start+frac[i]
    return X_train_client, Y_train_client, X_test, Y_test

def draw(save_path):
    plt.figure()
    acc = np.loadtxt(save_path+'acc.csv', delimiter=',')
    plt.plot(acc)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.savefig(save_path+'acc.png')

if __name__ == "__main__":
    K = 20
    N = 4
    frac = [1000]*10 + [2000]*10
    frac_norm = [i/sum(frac) for i in frac]
    X_train_client, Y_train_client, X_test, Y_test = generate_data(
        K, frac_norm)
    out_epoch = 50
    in_epoch = 5

    save_path = 'result/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    all_model = []
    for i in range(K):
        all_model.append(CNN())

    all_model[-1].save(save_path+'init_model.h5')

    for i in range(K):
        all_model[i] = load_model(save_path+'init_model.h5')
        all_model[i].compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


    print('----BEGIN')
    all_weights = [model.get_weights() for model in all_model]
    acc_list = []
    loss_list = []
    norm_list = []

    tracker = CarbonTracker(epochs=out_epoch, epochs_before_pred=-1, monitor_epochs=-1, interpretable=True)
    for epoch in range (1):
        tracker.epoch_start()
        for t in range(out_epoch):
        # Local model update
            arm = np.random.choice(K, size=N, replace=False, p=frac_norm)
            start_time = time.process_time()
            for a in arm:
                all_model[a].fit(x=X_train_client[a], y=Y_train_client[a],
                                batch_size=32, epochs=in_epoch, verbose=1)
            print('**TRAIN, ', time.process_time() - start_time)

    # For global model update: collect weights
            start_time = time.process_time()
            for a in arm:
                all_weights[a] = all_model[a].get_weights()
            print('**GET, ', time.process_time() - start_time) #weight retrival time

    # For global model update: average weights
            start_time = time.process_time()
            layer_num = len(all_weights[0])
            ave_weights = []
            for layer in range(layer_num):
                init = np.zeros_like(all_weights[0][layer])
                for index, model in enumerate(all_weights):
                    init += frac_norm[index]*model[layer]
                ave_weights.append(init)
            print('**AVE, ', time.process_time() - start_time) #weighted average compute time

    # For global model update: set weights and update all models (including those not participating for the current round)
            start_time = time.process_time()
            for i in range(K):
                all_model[i].set_weights(ave_weights)
            print('**SET, ', time.process_time() - start_time) #model update time

            start_time = time.process_time()
            loss, acc = all_model[0].evaluate(x=X_test, y=Y_test)
            print('**TEST, ', time.process_time() - start_time) #test time

            loss_list.append(loss)
            norm_list.append([np.linalg.norm(i) for i in ave_weights])
            print('----', t, arm, loss, acc) # current iteration, clients selected, loss, accuracy
            print(norm_list[-1]) #L2 norms for each layer

            if t % 10 == 0:
                np.savetxt(save_path+'loss.csv', loss_list, delimiter=',')
                np.savetxt(save_path+'acc.csv', acc_list, delimiter=',')
                np.savetxt(save_path+'norm.csv', norm_list, delimiter=',')
                all_model[0].save(save_path+'model_'+str(t)+'.h5')
        tracker.epoch_end()
        np.savetxt(save_path+'loss.csv', loss_list, delimiter=',')
        np.savetxt(save_path+'acc.csv', acc_list, delimiter=',')
        all_model[0].save(save_path+'model.h5')
        np.savetxt(save_path+'norm.csv', norm_list, delimiter=',')

        draw(save_path)
    tracker.stop() 