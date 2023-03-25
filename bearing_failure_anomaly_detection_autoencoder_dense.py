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
from keras.models import Sequential
from keras.layers import Input, Dense,Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras.backend as K
from keras_visualizer import visualizer 

#Model Evaluation
from sklearn.metrics import mean_squared_error


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
train_size = int(0.8 * df.shape[0])
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


# Define the autoencoder model
def create_model():
    model = Sequential()
    model.add(Dense(3, activation='LeakyReLU', kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(0.0),input_shape=(train_data.shape[1],)))
    model.add(Dense(2, activation='LeakyReLU',kernel_initializer='glorot_uniform'))
    model.add(Dense(3, activation='LeakyReLU',kernel_initializer='glorot_uniform'))
    model.add(Dense(train_data.shape[1],kernel_initializer='glorot_uniform'))
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=[r_square])
    return model


# In[21]:


final_autoencoder = create_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history=final_autoencoder.fit(train_data, train_data,
                      epochs=100,
                      batch_size=10,
                      shuffle=True,
                      validation_split=.05,
                      callbacks=[early_stopping])


# In[22]:


print(final_autoencoder.summary())


# In[23]:


#plot training and validation loss
plt.plot(history.history['loss'],
        'b',
        label='Training loss')
plt.plot(history.history['val_loss'],
        'r',
        label='Validation loss')
plt.ylim([0,.002])
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[24]:


#Vizualize loss distribution
train_pred=final_autoencoder.predict(np.array(train_data))
train_pred=pd.DataFrame(train_pred,columns=train_data.columns,index=train_data.index)
train_result=pd.DataFrame(index=train_data.index)
train_result['loss_mae']=np.mean(np.abs(train_pred-train_data),axis=1)
plt.figure()
sns.histplot(train_result['loss_mae'],kde=True,color='blue')
plt.xlim([0,.005])
plt.ylim([0,200])


# In[32]:


# Define threshold for anomaly detection
mae=np.mean(np.abs(train_pred-train_data),axis=1)
# Define threshold
threshold = np.mean(mae) + 4 * np.std(mae)
#threshold=np.percentile(mae,95)
threshold


# In[33]:


#Marking anomalous Training data 
train_pred=final_autoencoder.predict(np.array(train_data))
train_pred=pd.DataFrame(train_pred,columns=train_data.columns,index=train_data.index)
train_result['loss_mae']=np.mean(np.abs(train_pred-train_data),axis=1)
train_result['Threshold']=threshold
train_result['Anomoly']=train_result['loss_mae']>train_result['Threshold']
train_result.tail()


# In[34]:


#Marking anomalous Test data 
test_pred=final_autoencoder.predict(np.array(test_data))
test_pred=pd.DataFrame(test_pred,columns=test_data.columns,index=test_data.index)
test_result=pd.DataFrame(index=test_data.index)
test_result['loss_mae']=np.mean(np.abs(test_pred-test_data),axis=1)
test_result['Threshold']=threshold
test_result['Anomoly']=test_result['loss_mae']>test_result['Threshold']
test_result.tail()


# In[35]:


#merging model predicted data for train and test 
result=pd.concat([train_result,test_result])


# In[36]:


#Visualizing model output along with theshold for anomaly detection
result.plot(logy=True,figsize=(10,6),ylim=[0,10],color=['cyan','red'],title='Model Prediction ',xlabel='Date Time',ylabel='Reconstruction Loss')
data.plot(figsize=(10,6))


# In[37]:


result.to_csv('anomaly_dense.csv')


# In[ ]:




