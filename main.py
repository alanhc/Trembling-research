import tensorflow as tf
from utils.data import *
from utils.lstm_rnn import *
import os
import pandas as pd

##### config #####
 
save_dataset = False # 儲存dataset，true: 生成dataset、false: 使用save/ 儲存的


##### config #####


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], enable=True)

repeat = 100
if save_dataset:
    X_raw, y = generate_dataset(repeat=repeat)
    X = tf.reshape(X_raw, [y.shape[0],3,60])
    np.save("save/X.npy", X.numpy())
    np.save("save/y.npy", y)
else:
    X = np.load("save/X.npy")
    y = np.load("save/y.npy")

X = tf.convert_to_tensor(X)
train_dataset, val_dataset, test_dataset = split_dataset(X, y)

try_timesteps=[3,6,9]#list(range(3,18,3))
try_units=[100,150,300]#[50,100,200,300,180]#list(range(10,200,20))
try_epochs=[150]

"""
try_timesteps=[3]
try_units=[180]#list(range(10,200,20))
try_epochs=[150]
"""
df_all = pd.DataFrame([], columns=[["loss", "out_1_loss", "out_2_loss", "out_1_acc", "out_2_acc", "timesteps", "units", "epoch"]])
for timesteps in try_timesteps:
    for units in try_units:
        for epoch in try_epochs:
            #X = tf.reshape(X_raw, [y.shape[0],timesteps,180//timesteps])
            #print(X.shape)
            #train_dataset, val_dataset, test_dataset = split_dataset(X, y)
            model = construct_model(timesteps=timesteps, data_dim = 180//timesteps, units=units)
            #tf.keras.utils.plot_model(model, show_shapes=True)
            model, history = compile_fit_model(model,epochs=epoch,train_dataset=train_dataset,val_dataset=val_dataset)
            model.save("saved_model/lstm-rnn-unit("+str(units)+")-timesteps("+str(timesteps)+")-epoch("+str(epoch)+")")
            results = model.evaluate(test_dataset)
            results.extend([timesteps, units, epoch])
            df = pd.DataFrame([results], columns=[["loss", "out_1_loss", "out_2_loss", "out_1_acc", "out_2_acc", "timesteps", "units", "epoch"]])
            df_all = df_all.append(df)
df_all.to_csv("save/result.csv")