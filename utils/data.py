import numpy as np
import tensorflow as tf
import time
import math

PI=math.pi

def sliding_window(data=None, window_size = 3, shift = 1):
    return np.array(list( list(data[(x*shift):(x*shift)+window_size]) for x in range((data.shape[0]-window_size+1)//shift) ))

def np_generate_signal(frequences, sampling_rate):
    f1, f2 = np.copy(frequences)
    f1 /= 2
    f2 /= 2
    sh=0
    if (f1==f2):
        sh = PI/2
    # generate s1、s2
    signalTime = np.arange(0, 6, 1/sampling_rate)
    signalAmplitude_1 = np.sin(2*PI*f1*signalTime+sh)
    signalAmplitude_2 = np.sin(2*PI*f2*signalTime)
    std_1 = np.std(signalAmplitude_1)
    std_2 = np.std(signalAmplitude_2)
    #mu, sigma_1, sigma_2  = 0.0, std_1, std_2
    mu, sigma_1, sigma_2  = 0.0, 0.1,0.1
    
    n1 = np.random.normal(mu, sigma_1, [6*sampling_rate])
    n2 = np.random.normal(mu, sigma_2,[6*sampling_rate])
    
    noiseSignalAmplitude_1 = signalAmplitude_1 + n1
    noiseSignalAmplitude_2 = signalAmplitude_2 + n2
    
    combine = noiseSignalAmplitude_1 + noiseSignalAmplitude_2
    combine = combine - combine.mean()
    _max, _min= combine.max(), combine.min()
    if f1 or f2:
        combine = ((combine-_min)/(_max-_min)) * 2 -1

    return combine


def generate_dataset(_from=0, _to=20, sampling_rate=30, repeat=1, d=0.5):
 
    f1 = np.arange(_from,_to*2, d*2)
    f2 = np.arange(_from, _to*2, d*2)

    mesh = np.array(np.meshgrid(f1,f2))
    # create labels 
    combination = mesh.T.reshape(-1,2)
    combination = np.sort(combination)
    #combination = np.unique(combination, axis=0)
    combination = np.repeat(combination, repeat, axis=0)
    print("num of freq combination", combination.shape)
    combination_onehot = tf.keras.utils.to_categorical(combination, num_classes=40)
    
    
    data = np.zeros(sampling_rate*6)

    i=0

    for f1,f2 in combination:
        start = time.time()
        s = np_generate_signal([f1,f2],sampling_rate)
        data = tf.concat((data, s), axis=0)


        end = time.time()

        remain = combination.shape[0]-i
        print("["+str(remain)+"]|t:"+str(round((end-start)*remain,2)), end="\r")
        i+=1

    data = data[180:]
    print(data.shape)
    
    return data, combination_onehot # X,y

def generate_time_series_dataset(window_size = 60, shift = 10, _from=0, _to=20, sampling_rate=30, repeat=1, d=0.5):
 
    f1 = np.arange(_from,_to*2, d*2)
    f2 = np.arange(_from, _to*2, d*2)

    mesh = np.array(np.meshgrid(f1,f2))
    # create labels 
    combination = mesh.T.reshape(-1,2)
    combination = np.sort(combination)
    #combination = np.unique(combination, axis=0)
    combination = np.repeat(combination, repeat, axis=0)
    print("num of freq combination", combination.shape)
    combination_onehot = tf.keras.utils.to_categorical(combination, num_classes=40)
    
    
    data = np.zeros(sampling_rate*6)

    i=0

    for f1,f2 in combination:
        start = time.time()
        s = np_generate_signal([f1,f2],sampling_rate)
        s_window = sliding_window(data=s, window_size = window_size, shift = shift).flatten()
        data = tf.concat((data, s_window), axis=0)


        end = time.time()

        remain = combination.shape[0]-i
        print("["+str(remain)+"]|t:"+str(round((end-start)*remain,2)), end="\r")
        i+=1

    data = data[180:]
    print(data.shape)
    
    return data, combination_onehot # X,y


def split_dataset(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15/0.85, random_state=42)
    print(y_train.shape[0], y_val.shape[0], y_test.shape[0])
    
    
    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    y_train1_d = tf.data.Dataset.from_tensor_slices(y_train[:,0])
    y_train2_d = tf.data.Dataset.from_tensor_slices(y_train[:,1])
    
    X_val = tf.data.Dataset.from_tensor_slices(X_val)
    y_val1_d = tf.data.Dataset.from_tensor_slices(y_val[:,0])
    y_val2_d = tf.data.Dataset.from_tensor_slices(y_val[:,1])
    
    X_test = tf.data.Dataset.from_tensor_slices(X_test)
    y_test1_d = tf.data.Dataset.from_tensor_slices(y_test[:,0])
    y_test2_d = tf.data.Dataset.from_tensor_slices(y_test[:,1])
    
    
    
    train_dataset = tf.data.Dataset.zip((X_train,(y_train1_d,y_train2_d)))
    val_dataset = tf.data.Dataset.zip((X_val,(y_val1_d,y_val2_d)))
    test_dataset = tf.data.Dataset.zip((X_test,(y_test1_d,y_test2_d)))

    print("train:", tf.data.experimental.cardinality(train_dataset).numpy())
    print("val:",tf.data.experimental.cardinality(val_dataset).numpy())
    print("test:",tf.data.experimental.cardinality(test_dataset).numpy())

    print("new train",tf.data.experimental.cardinality(train_dataset).numpy())

    BATCH_SIZE=4096
    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    
    return train_dataset, val_dataset, test_dataset
