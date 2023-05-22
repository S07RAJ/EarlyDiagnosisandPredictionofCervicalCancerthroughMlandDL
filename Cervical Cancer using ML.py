#!/usr/bin/env python
# coding: utf-8

# In[1]:


#detection cancer 


# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[5]:


cc_df = pd.read_csv(r'E:\Cancer PRJ\data.csv')


# In[6]:


cc_df.head()


# In[7]:


cc_df.shape


# In[8]:


cc_df.info()


# In[9]:


cc_df.describe()


# In[10]:


cc_df = cc_df.replace('?', np.nan)
cc_df


# In[11]:


cc_df.isnull()


# In[12]:


cc_df.info()


# In[13]:


cc_df = cc_df.drop(columns = ['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])
cc_df


# In[14]:


cc_df = cc_df.apply(pd.to_numeric)
cc_df.info()


# In[15]:


cc_df.describe()


# In[16]:


cc_df.mean()


# In[17]:


cc_df = cc_df.fillna(cc_df.mean())
cc_df


# In[18]:


corr_matrix = cc_df.corr()
corr_matrix


# In[19]:


plt.figure(figsize =(30,30))
sns.heatmap(corr_matrix, annot = True)
plt.show()


# In[20]:


cc_df.info()


# In[21]:


# Change all the datatype to be float 64
cc_df['Age'] = cc_df['Age'].astype(float)
cc_df['STDs: Number of diagnosis'] = cc_df['STDs: Number of diagnosis'].astype(float)
cc_df['Dx:Cancer'] = cc_df['Dx:Cancer'].astype(float)
cc_df['Dx:CIN'] = cc_df['Dx:CIN'].astype(float)
cc_df['Dx:HPV'] = cc_df['Dx:HPV'].astype(float)
cc_df['Dx'] = cc_df['Dx'].astype(float)
cc_df['Hinselmann'] = cc_df['Hinselmann'].astype(float)
cc_df['Schiller'] = cc_df['Schiller'].astype(float)
cc_df['Citology'] = cc_df['Citology'].astype(float)
cc_df['Biopsy'] = cc_df['Biopsy'].astype(float)


# In[22]:


cc_df.info()


# In[23]:


sns.countplot(x="Biopsy",data=cc_df);
counts = cc_df['Biopsy'].value_counts()
counts


# In[24]:


sns.countplot(x="Biopsy",data=cc_df);
counts = cc_df['Biopsy'].value_counts()
counts


# In[25]:


y = cc_df['Biopsy']
X = cc_df.drop(columns = ['Biopsy'])


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
print(X_test.shape)
print(X_train.shape)


# In[27]:


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train,y_train.values.ravel())


# In[28]:


RandomForestClassifier()


# In[29]:


feat_scores= pd.DataFrame({"Fraction of variables affected" : rf.feature_importances_},index = X.columns)
feat_scores= feat_scores.sort_values(by = "Fraction of variables affected")
feat_scores.plot(kind = "barh", figsize = (10, 5))
sns.despine()


# In[30]:


# Positivity by Schiller
ax = sns.kdeplot(cc_df.Schiller[(cc_df["Biopsy"] == 0)],
               color = "Red", shade = True)
ax = sns.kdeplot(cc_df.Schiller[(cc_df["Biopsy"] == 1)],
               color = "Blue", shade = True)

ax.legend(["Negative", "Positive"], loc = "upper right")
ax.set_ylabel("Density")
ax.set_xlabel("Schiller")
ax.set_title("Distribution of Schiller by positivity")


# In[31]:


ax = sns.kdeplot(cc_df["First sexual intercourse"][(cc_df["Biopsy"] == 0)],
               color = "Red", shade = True)
ax = sns.kdeplot(cc_df["First sexual intercourse"][(cc_df["Biopsy"] == 1)],
               color = "Blue", shade = True)

ax.legend(["Negative", "Positive"], loc = "upper right")
ax.set_ylabel("Density")
ax.set_xlabel("First sexual intercourse")
ax.set_title("Distribution of First sexual intercourse by positivity")


# In[32]:


from sklearn.ensemble import RandomForestClassifier

model_rf=RandomForestClassifier()
model_rf.fit(X_train,y_train)


# In[33]:


RandomForestClassifier()


# In[34]:


y_predict1=model_rf.predict(X_test)


# In[35]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_predict1))


# In[36]:


cm=confusion_matrix(y_test,y_predict1)
sns.heatmap(cm,annot=True)


# In[37]:


from sklearn.calibration import CalibratedClassifierCV 
from sklearn.svm import LinearSVC

model_svm=LinearSVC(max_iter=10000)
model_svm=CalibratedClassifierCV(model_svm)
model_svm.fit(X_train,y_train)


# In[38]:


CalibratedClassifierCV(base_estimator=LinearSVC(max_iter=10000))


# In[39]:


y_predict2=model_svm.predict(X_test)


# In[40]:


print(classification_report(y_test,y_predict2))


# In[41]:


cm=confusion_matrix(y_test,y_predict2)
sns.heatmap(cm, annot=True)


# In[42]:


from sklearn.neigh()
KNeighborsClassifier()
KNeighborsClassifier()bors import KNeighborsClassifier

model_knn=KNeighborsClassifier()
model_knn.fit(X_train,y_train)


# In[46]:


KNeighborsClassifier()


# In[45]:


y_predict3=model_knn.predict(X_test)


# In[47]:


print(classification_report(y_test,y_predict3))


# In[48]:


cm=confusion_matrix(y_test,y_predict3)
sns.heatmap(cm,annot=True)


# In[49]:


from sklearn.naive_bayes import GaussianNB

model_gnb=GaussianNB()
model_gnb.fit(X_train, y_train)


# In[50]:


GaussianNB()


# In[51]:


y_predict4=model_gnb.predict(X_test)


# In[52]:


print(classification_report(y_test, y_predict4))


# In[53]:


cm = confusion_matrix(y_test, y_predict4)
sns.heatmap(cm, annot = True)


# In[54]:


from sklearn.metrics import roc_curve

fpr1, tpr1, thresh1 = roc_curve(y_test, model_rf.predict_proba(X_test)[:, 1], pos_label = 1)
fpr2, tpr2, thresh2 = roc_curve(y_test, model_svm.predict_proba(X_test)[:, 1], pos_label = 1)
fpr3, tpr3, thresh3 = roc_curve(y_test, model_knn.predict_proba(X_test)[:, 1], pos_label = 1)
fpr4, tpr4, thresh4 = roc_curve(y_test, model_gnb.predict_proba(X_test)[:, 1], pos_label = 1)


# In[55]:


plt.plot(fpr1, tpr1, linestyle = "--", color = "green", label = "Random Forest")
plt.plot(fpr2, tpr2, linestyle = "--", color = "red", label = "SVM")
plt.plot(fpr3, tpr3, linestyle = "--", color = "purple", label = "KNN")
plt.plot(fpr4, tpr4, linestyle = "--", color = "orange", label = "Naive bayes")


plt.title('Receiver Operator Characteristics (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')

plt.legend(loc = 'best')
plt.savefig('ROC', dpi = 300)
plt.show()


# In[56]:


fig,ax=plt.subplots(figsize=(25,15))
ax=sns.boxplot(y='Age', x='First sexual intercourse', hue='Biopsy',data=cc_df)


# In[57]:


fig,ax=plt.subplots(figsize=(20,15))
ax=sns.barplot(y='Age',x='First sexual intercourse', hue='Biopsy', data=cc_df)
ax.set(xlabel='First sexual intercourse', ylabel='Age')
plt.show()


# In[ ]:




