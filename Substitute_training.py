# Python packages:
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l1
from matplotlib import pyplot as plt

def normalization(dataX):
  min_value=-20000
  max_value=20000
  scaled_dataX = (dataX-min_value)/(max_value-min_value)
  return scaled_dataX

def reverse_scaling(x,min_value,max_value):
    unnorm_x = x*(max_value-min_value)+min_value
    return unnorm_x

#loss for discrete model
loss_object = tf.keras.losses.binary_crossentropy
#for probabilistic model: tf.keras.losses.KLD
def get_jacobian(input_data, input_label,origin_result):
    with tf.GradientTape() as g:
        #g.watch(input_data)
        predictions = model(input_data)[:,origin_result]
        #dy_dx = g.gradient(predictions, input_data)
    dy_dx=g.jacobian(predictions, input_data)
   # print(dy_dx)
    return dy_dx

def create_adversarial_pattern(input_data, input_label,origin_result):
    with tf.GradientTape() as g:
        g.watch(input_data)
        #loss = loss_object(input_label, model(input_data))
        loss = model(input_data)[:,origin_result]
        dy_dx = g.gradient(loss, input_data)
        return dy_dx

# Setting values for 5 Original Input, the paper sets them to be constant and changing every 10 mins.
original_training_x = np.zeros([5,60])

#Load NILM model (the oracle model with discrete output)
nilm_model = load_model('target.h5')

#define the substitute model
model = Sequential()
model.add(Dense(128, input_dim=60))
model.add(Dense(256,activation='tanh'))
model.add(Dense(128,activation='tanh'))
model.add(Dense(64,activation='tanh'))
model.add(Dense(32,activation='tanh'))
model.add(Dense(16,activation='tanh'))
model.add(Dense(8,activation='tanh'))
model.add(Dense(2, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-2, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Run this block several epochs to extract the substitute:
'''Model Extraction Start'''
#1. this is the first time querying
train_x = original_training_x.reshape([-1,60,1])

#2. this is the querying data in other epochs
train_x = pd.read_csv('query_data.csv',index_col=0).values.reshape([-1,60,1])
scaled_x = normalization(train_x).reshape([-1,60])
#Query the oracle
train_y_class = nilm_model.predict_classes(scaled_x)
train_y_2_output = to_categorical(train_y_class)
#for probabilistic model
#train_y_prob_1 = nilm_model.predict_proba(norm_train_x)

#3.Train the substitute 
model.fit(scaled_x, train_y_2_output,epochs=1000, batch_size=None)
y_substitute_prob = model.predict_proba(scaled_x)
y_class = model.predict_classes(scaled_x)
model.save('substitute.h5')

#4. New query data generating
adv_data_x = None
ori_data_x = None
for i in range(0,scaled_x.shape[0]):
    signal = scaled_x[i,:].reshape([-1,60])
    ori = signal
    input_data = tf.keras.backend.variable(signal)
    origin_result = model.predict_classes(input_data.numpy())
    counter = 0
    #momentum
    alpha = 0.9
    #velocity
    v = np.zeros([1,60])
    #base learning rate
    epsilon = 1
    #flag=true: gradient ascent; false:gradient descent(by hiding the sign)
    flag = True
    max_iteration = 1000
    while (counter<max_iteration):
        counter = counter+1
        if (flag==True):
          input_label = tf.keras.backend.variable(to_categorical(model.predict_classes(signal),num_classes=2))
        perturbations  = get_jacobian(input_data,input_label,origin_result[0])
        v = alpha*v+epsilon*perturbations
        signal = tf.nn.relu(signal - v).numpy().reshape([-1,60])
        perturbation_result = model.predict_classes(signal)
        perturbation_prob = model.predict_proba(signal)
        if (perturbation_result!=origin_result):
          flag = False
          if(abs(perturbation_prob[:,0]-perturbation_prob[:,1])>0):
            print('We got here '+str(i))
            if adv_data_x is None:
                adv_data_x = signal
            else:
                adv_data_x = np.vstack([adv_data_x,signal])
            if ori_data_x is None:
                ori_data_x = ori
            else:
                ori_data_x = np.vstack([ori_data_x,ori])
            break;
        input_data = tf.keras.backend.variable(signal)

#5. Check the substitute performnce
adv_test = nilm_model.predict_classes(adv_data_x.reshape([-1,60,1])).flatten()
#compare adv_test with train_y_class, if better than threshold epsilon, then stop.
# Save the adversarial data:
new_train_x = np.vstack([train_x.reshape([-1,60]),reverse_scaling(adv_data_x,-20000,20000)])
df = pd.DataFrame(new_train_x)
df.to_csv('query_data.csv')
'''Model Extraction End'''
