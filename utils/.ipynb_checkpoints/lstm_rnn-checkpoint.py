from tensorflow.keras import layers, Model
import datetime
import tensorflow as tf

def construct_model(timesteps=3, data_dim = 60, units=70):
    
    num_classes = 40

    inputs = layers.Input(shape=(timesteps, data_dim))
    units = units

    output_1, state_h_1, state_c_1 = layers.LSTM(units, return_state=True, return_sequences=True, name="vfc_1_1")(inputs)
    output_2, state_h_2, state_c_2 = layers.LSTM(units, return_state=True, name="vfc_1_2")(output_1)
    encoder_state_1 = [state_h_1, state_c_1]
    encoder_state_2 = [state_h_2, state_c_2]
    fc1_1 = layers.Dense(units, activation="relu")(state_h_2)
    fc1_2 = layers.Dense(num_classes, activation="softmax",name="out_1")(fc1_1)

    lstm_2_1 = layers.LSTM(units, return_sequences=True, name="vfc_2_1")(inputs, initial_state=encoder_state_1)
    lstm_2_2 = layers.LSTM(units, name="vfc_2_2")(lstm_2_1, initial_state=encoder_state_2)
    fc2_1 = layers.Dense(units, activation="relu")(lstm_2_2)
    fc2_2 = layers.Dense(num_classes, activation="softmax",name="out_2")(fc2_1)


    model = Model(inputs=inputs, outputs=[fc1_2, fc2_2])
    print(model.summary())
    
    #tf.keras.utils.plot_model(model, show_shapes=True)
    return model

def compile_fit_model(model, epochs=150,train_dataset=None,val_dataset=None):
    cce = tf.keras.losses.CategoricalCrossentropy()

    #%load_ext tensorboard
    
    #!rm -rf ./logs/
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)


    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=cce,
                  metrics=['acc'])
    history = model.fit(train_dataset,
                        epochs=epochs,
                        validation_data=val_dataset,
                        callbacks=[callback, tensorboard_callback],

                        )
    #%tensorboard --logdir logs/fit --host 0.0.0.0
    return model, history
    