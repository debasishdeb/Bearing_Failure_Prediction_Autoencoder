#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data manipulation library
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

#Data visualization librari
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats as sci
from scipy import signal
from statsmodels.tsa.stattools import acf
#Data preprocessing library
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

#ML Model
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Input, Dense,Dropout,LSTM,TimeDistributed, RepeatVector
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import keras.backend as K
from keras_visualizer import visualizer 
#Model Evaluation
from sklearn.metrics import mean_squared_error
import pydot
import graphviz


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


def load_data(data_dir):
    data = pd.DataFrame()
    for filename in tqdm(os.listdir(data_dir)):
        dataset = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
        dataset_mean_abs = dataset.abs().mean().values.reshape(1, -1)
        dataset_mean_abs = pd.DataFrame(dataset_mean_abs, index=[filename])
        data = data.append(dataset_mean_abs)
    return data


# In[4]:


#location of input data file
data_dir='C:/Users/user/Documents/DataScience/self/datasets/2nd_test/2nd_test'
#data_dir='C:/Users/user/Documents/DataScience/self/datasets/3rd_test/4th_test/txt'


# In[5]:


data=load_data(data_dir)


# In[6]:


df_org=data


# In[7]:


df_org.columns=['brg1','brg2','brg3','brg4']


# In[8]:


df_org.shape


# In[9]:


df_org.head()


# In[10]:


df_copy=df_org.copy()


# In[11]:


df_copy.info()


# In[12]:


df_copy.shape


# ## Creating Custom Summary

# In[13]:


#Provide custom summary
def custom_summary(my_df):
    result = []
    for col in my_df.columns:
        if my_df[col].dtypes != 'object':
            stats = OrderedDict({
                'feature_name' : col,
                'count' : my_df[col].count(),
                'quartile1' : my_df[col].quantile(.25),
                'quartile2' : my_df[col].quantile(.50),
                'quartile3' : my_df[col].quantile(.75),
                'mean' : my_df[col].mean(),
                'max' : my_df[col].max(),
                'variance' : round(my_df[col].var()),
                'standard_deviation' : my_df[col].std(),
                'skewness' : my_df[col].skew(),
                'kurtosis' : my_df[col].kurt()
            })
            result.append(stats)
    result_df = pd.DataFrame(result)
    # skewness type :
    skewness_label = []
    for i in result_df['skewness']:
        if i <= -1:
            skewness_label.append('Highly negatively skewed')
        elif -1 < i <= -0.5:
            skewness_label.append('Moderately negatively skewed')
        elif -0.5 < i < 0:
            skewness_label.append('Fairly negatively skewed')
        elif 0 <= i <= 0.5:
            skewness_label.append('Fairly Positively skewed')
        elif 0.5 <= i < 1:
            skewness_label.append('Moderately Positively skewed')
        elif i >= 1:
            skewness_label.append('Highly Positively skewed')
    result_df['skewness_comment'] = skewness_label
    
    # kurtosis type :
    kurtosis_label = []
    for i in result_df['kurtosis']:
        if i >= 1:
            kurtosis_label.append('Leptokurtic Curve')
        elif i <= -1:
            kurtosis_label.append('Platykurtic Curve')
        else:
            kurtosis_label.append('Mesokurtic Curve')
    result_df['kurtosis_comment'] = kurtosis_label
    
    # Outliers :
    outliers_label = []
    for col in  my_df.columns:
        if  my_df[col].dtypes != 'object':
            q1 =  my_df[col].quantile(.25)
            q2 =  my_df[col].quantile(.50)
            q3 =  my_df[col].quantile(.75)
            iqr = q3-q1
            lower_whisker = q1-1.5*iqr
            upper_whisker = q3+1.5*iqr
            if len( my_df[( my_df[col] < lower_whisker) | ( my_df[col] > upper_whisker) ]) > 0:
                outliers_label.append('Have outliers')
            else:
                outliers_label.append('No outliers')
    result_df['Outlier Comment'] = outliers_label
    return result_df

        


# In[14]:


custom_summary(df_copy)


# In[15]:


df=df_copy.copy()


# In[16]:


# df.index=pd.to_datetime(df.index,format='%Y-%m-%d %H:%M:%S')
df.index=pd.to_datetime(df.index,format='%Y.%m.%d.%H.%M.%S')


# In[17]:


# Split the dataset into train and test sets
train_size = int(0.5 * df.shape[0])
train_data = df[:train_size]
test_data = df[train_size:]


# In[18]:


import random
random.seed(0)


# In[19]:


def r_square(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred)) 
    ss_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - ss_res/(ss_tot + K.epsilon()))


# In[20]:


# reshape inputs for LSTM [samples, timesteps, features]
train_X = np.array(train_data).reshape(train_data.shape[0], 1, train_data.shape[1])
print("Training data shape:", train_X.shape)
test_X = np.array(test_data).reshape(test_data.shape[0], 1, test_data.shape[1])
print("Test data shape:", test_X.shape)


# In[21]:


# Define the autoencoder model
def create_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(4, activation='relu', return_sequences=True, 
              kernel_regularizer=l2(.001))(inputs)
    L2 = LSTM(2, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(3, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(4, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=[r_square])
    return model


# In[22]:


final_autoencoder = create_model(train_X)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history=final_autoencoder.fit(train_X, train_X,
                      epochs=100,
                      batch_size=10,
                      shuffle=True,
                      validation_split=.05,
                      callbacks=[early_stopping])


# In[23]:


print(final_autoencoder.summary())


# In[24]:


#plot training and validation loss
plt.plot(history.history['loss'],
        'b',
        label='Training loss')
plt.plot(history.history['val_loss'],
        'r',
        label='Validation loss')
plt.ylim([0,.02])
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:





# In[25]:


#Vizualize loss distribution
train_pred=final_autoencoder.predict(train_X)
train_pred = train_pred.reshape(train_pred.shape[0], train_pred.shape[2])
train_pred=pd.DataFrame(train_pred,columns=train_data.columns,index=train_data.index)
train_result=pd.DataFrame(index=train_data.index)
train_result['loss_mae']=np.mean(np.abs(train_pred-train_data),axis=1)
plt.figure()
sns.histplot(train_result['loss_mae'],kde=True,color='blue')
plt.xlim([0,.004])
plt.ylim([0,100])


# In[60]:


# Define threshold for anomaly detection
mae=np.mean(np.abs(train_pred-train_data),axis=1)
# Define threshold
threshold = np.mean(mae) + 5.1 * np.std(mae)
#threshold=np.percentile(mae,95)
threshold


# In[61]:


#Marking anomalous Training data 
train_pred=final_autoencoder.predict(train_X)
train_pred = train_pred.reshape(train_pred.shape[0], train_pred.shape[2])
train_pred=pd.DataFrame(train_pred,columns=train_data.columns,index=train_data.index)
train_result['loss_mae']=np.mean(np.abs(train_pred-train_data),axis=1)
train_result['Threshold']=threshold
train_result['Anomoly']=train_result['loss_mae']>train_result['Threshold']
train_result.tail()


# In[62]:


#Marking anomalous Test data 
test_pred=final_autoencoder.predict(test_X)
test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[2])
test_pred=pd.DataFrame(test_pred,columns=test_data.columns,index=test_data.index)
test_result=pd.DataFrame(index=test_data.index)
test_result['loss_mae']=np.mean(np.abs(test_pred-test_data),axis=1)
test_result['Threshold']=threshold
test_result['Anomoly']=test_result['loss_mae']>test_result['Threshold']
test_result.tail()


# In[63]:


#merging model predicted data for train and test 
result=pd.concat([train_result,test_result])


# In[64]:


#Visualizing model output along with theshold for anomaly detection
result.plot(logy=True,figsize=(10,6),ylim=[0,10],color=['cyan','red'],title='Model Prediction ',xlabel='Date Time',ylabel='Reconstruction Loss')
data.plot(figsize=(20,10))


# In[65]:


final_autoencoder.summary()

