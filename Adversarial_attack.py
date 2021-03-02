import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler

def calc_euclidean_distance(ori_signal,adv_signal):
  el_dsts = np.sqrt(np.sum(np.square(ori_signal-adv_signal)))
  return el_dsts

def reverse_scaling(x,min_value,max_value):
    unnorm_x = x*(max_value-min_value)+min_value
    return unnorm_x

def reverse_norm(scaled_dataX):
    min_value=-20000
    max_value=20000
    #mean_value=-20000
    dataX = scaled_dataX*((max_value-min_value))+min_value
    return dataX

loss_object = tf.keras.losses.KLD
#binary_crossentropy
def create_adversarial_pattern(input_data, input_label):
    with tf.GradientTape() as g:
        g.watch(input_data)
        loss = loss_object(input_label, model(input_data))
        #print(loss)
        dy_dx = g.gradient(loss, input_data)
        return dy_dx


'''Sliding window function'''
def create_dataset(dataset, look_back=1):
    dataX = []
    for i in range(len(dataset)-look_back+1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
    return np.array(dataX)

def minmax_scaling(x,min_value,max_value):
    norm_x = (x-min_value)/(max_value-min_value)
    return norm_x

def normalization(dataX):
  min_value=-20000
  max_value=20000
  scaled_dataX = (dataX-min_value)/(max_value-min_value)
  return scaled_dataX

def norm_infinite(ori_signal,adv_signal):
  diff_matrix = ori_signal - adv_signal
  norm_infinite = np.max(abs(diff_matrix))
  return norm_infinite

def norm_percentage(ori_signal,adv_signal):
  diff_matrix = ori_signal - adv_signal
  diff_percentage = diff_matrix/ori_signal
  norm_percentage = np.max(abs(diff_percentage))
  print(norm_percentage)
  return norm_percentage

def lball_projection(ori_signal,adv_signal,per):
  rever_ori = reverse_norm(ori_signal)
  rever_adv = reverse_norm(adv_signal)
  index_max = np.where(rever_adv>rever_ori*(1+per))[1]
  index_min = np.where(rever_adv<rever_ori*(1+per))[1]
  adv = tf.Variable(adv_signal)
  for idx in index_max:
    adv[:,idx].assign(rever_ori[:,idx]*(1+per))
  for idx in index_min:
    adv[:,idx].assign(rever_ori[:,idx]*(1-per))
  return normalization(adv)

#Load test data
X_test_raw = pd.read_csv('X_test.csv').values[:,1:]
y_test_raw = pd.read_csv('y_test.csv').values[:,1:]

#Sample data for testing
tl = RandomUnderSampler(sampling_strategy={0:10000,1:10000})
X_test, y_test = tl.fit_resample(X_test_raw, y_test_raw.flatten())
X_test_norm = normalization(X_test)

#load target model
nilm_model = load_model('target.h5')
y_nilm_test = nilm_model.predict_proba(X_test_norm.reshape([-1,60,1]))
train_y_class = nilm_model.predict_classes(X_test_norm.reshape([-1,60,1]))
#load substitute
model=load_model('substitute.h5')
sub_y_class = model.predict_classes(X_test_norm)
true_pred = np.sum(train_y_class==sub_y_class)

'''r-ratio projected_gradient_ascent algorithm'''
adv_data_x_g = None
ori_data_x_g = None
y_classes = []

#allowed ratio of perturbation
per = 0.3
#max iteration
max_iter = 1000
#step size
epsilon = 0.1

for i in range(0,scaled_x.shape[0]):
    #given a signal from the input data
    signal = scaled_x[i,:].reshape([-1,60])
    signal_y = to_categorical(nilm_y_class[i],num_classes=2)
    ori = signal
    #get original label
    origin_result = model.predict_classes(signal)
    #initilize the iterator
    iteration=0
    #confidence margin
    margin = 0.3

    #perturbation_percentage = 0
    while (iteration<=max_iter):
        iteration = iteration+1
        epsilon = epsilon+a
        input_data = tf.keras.backend.variable(signal)
        perturbations = create_adversarial_pattern(input_data,signal_y)
        pertubed = signal + epsilon*perturbations
        pro_signal = lball_projection(ori,pertubed,per)
        signal_1 = tf.nn.relu(pro_signal)
        perturbation_result = model.predict_classes(signal_1)
        perturbation_prob = model.predict_proba(signal_1)
        signal = signal_1
        if (perturbation_result!=origin_result):
          print('reach here')
          print(abs(perturbation_prob[:,0]-perturbation_prob[:,1]))
          if(abs(perturbation_prob[:,0]-perturbation_prob[:,1])>margin):
            if adv_data_x_g is None:
                adv_data_x_g = signal_1.numpy()
            else:
                adv_data_x_g = np.vstack([adv_data_x_g,signal_1.numpy()])
            if ori_data_x_g is None:
                ori_data_x_g = ori
            else:
                ori_data_x_g = np.vstack([ori_data_x_g,ori])
            y_classes.append(sub_y_class[i])
            break; 
