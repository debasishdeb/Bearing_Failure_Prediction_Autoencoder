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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#ML Model
from keras.models import Sequential
from keras.layers import Input, Dense,Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras.backend as K

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
data_dir='2nd_test'
#data_dir='3rd_test/4th_test/txt'


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


# Compute the first-order difference to make the data stationary
diff_df = df.diff().dropna()


# In[18]:


diff_df.shape


# In[19]:


diff_df.head()


# In[20]:


# Detrend the data using the scipy.signal.detrend() function
detrended_df = pd.DataFrame(signal.detrend(diff_df, axis=0), index=diff_df.index, columns=diff_df.columns)


# In[21]:


detrended_df.head()


# In[22]:


detrended_df.shape


# In[23]:


# Remove autocorrelation using the acf() and lfilter() functions
autocorr = [acf(detrended_df.iloc[:,i], nlags=len(detrended_df)) for i in range(detrended_df.shape[1])]
acorr_filtered = [signal.lfilter(np.concatenate(([1], -ac)), 1, detrended_df.iloc[:,i]) for i, ac in enumerate(autocorr)]
df_filtered = pd.DataFrame(np.column_stack(acorr_filtered), index=detrended_df.index, columns=detrended_df.columns)


# In[24]:


df_filtered.shape


# In[25]:


df_filtered.head()


# In[26]:


df_filtered=df_filtered.sort_index()


# In[27]:


df_filtered.head()


# In[28]:


# Split the dataset into train and test sets
train_size = int(0.8 * df_filtered.shape[0])
train_data = df_filtered[:train_size]
test_data = df_filtered[train_size:]


# In[29]:


# Scale the features using StandardScaler
scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)
train_scaled = scaler.fit_transform(train_data)
train_scaled =pd.DataFrame(train_scaled,columns =train_data.columns,index =train_data.index )
test_scaled = scaler.transform(test_data)
test_scaled =pd.DataFrame(test_scaled,columns =test_data.columns,index =test_data.index )


# In[30]:


import random
random.seed(0)


# In[31]:


def r_square(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred)) 
    ss_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - ss_res/(ss_tot + K.epsilon()))


# In[32]:


# Define the autoencoder model
def create_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01),input_shape=(train_data.shape[1],)))
    model.add(Dense(train_data.shape[1], activation='linear'))
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=[r_square])
    return model


# In[33]:


final_autoencoder = create_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history=final_autoencoder.fit(train_scaled, train_scaled,
                      epochs=100,
                      batch_size=10,
                      shuffle=True,
                      validation_split=.05,
                      callbacks=[early_stopping])


# In[34]:


#plot training and validation loss
plt.plot(history.history['loss'],
        'b',
        label='Training loss')
plt.plot(history.history['val_loss'],
        'r',
        label='Validation loss')
plt.ylim([0,2])
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[35]:


#Vizualize loss distribution
train_pred=final_autoencoder.predict(np.array(train_scaled))
train_pred=pd.DataFrame(train_pred,columns=train_scaled.columns,index=train_scaled.index)
train_result=pd.DataFrame(index=train_scaled.index)
train_result['loss_mae']=np.mean(np.abs(train_pred-train_scaled),axis=1)
plt.figure()
sns.histplot(train_result['loss_mae'],kde=True,color='blue')
plt.xlim([0,.25])
plt.ylim([0,200])


# In[36]:


# Define threshold for anomaly detection
mae=np.mean((train_pred-train_scaled),axis=1)
# Define threshold
threshold = np.mean(mae) + 3 * np.std(mae)
threshold


# In[37]:


#Marking anomalous Training data 
train_pred=final_autoencoder.predict(np.array(train_scaled))
train_pred=pd.DataFrame(train_pred,columns=train_scaled.columns,index=train_scaled.index)
train_result['loss_mae']=np.mean(np.abs(train_pred-train_scaled),axis=1)
train_result['Threshold']=threshold
train_result['Anomoly']=train_result['loss_mae']>train_result['Threshold']
train_result.tail()


# In[38]:


#Marking anomalous Test data 
test_pred=final_autoencoder.predict(np.array(test_scaled))
test_pred=pd.DataFrame(test_pred,columns=test_scaled.columns,index=test_scaled.index)
test_result=pd.DataFrame(index=test_scaled.index)
test_result['loss_mae']=np.mean(np.abs(test_pred-test_scaled),axis=1)
test_result['Threshold']=threshold
test_result['Anomoly']=test_result['loss_mae']>test_result['Threshold']
test_result.tail()


# In[39]:


#merging model predicted data for train and test 
result=pd.concat([train_result,test_result])


# In[40]:


#Visualizing model output along with theshold for anomaly detection
result.plot(logy=True,figsize=(10,6),ylim=[0,100],color=['blue','red'],title='Model Prediction ',xlabel='Date Time',ylabel='Reconstruction Loss')
data.plot


# In[ ]:




