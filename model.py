
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB , MultinomialNB
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pickle


df =pd.read_csv('data\Air Pollution.csv')


#drop duplicates values
df.drop_duplicates(inplace=True)





# drop the rows that have Philippines as Country Name
index_names = df[ df['Country Name'] == 'Philippines' ].index
df.drop(index_names, inplace = True)

#drop the rows that have the Taylor City as City
index_canada = df[ df['City'] == 'Taylor' ].index
df.drop(index_canada, inplace = True)





#Take the PM10 mean based on Country to fill the null values
df['PM10 (μg/m3)'] = df.apply(lambda row: df.groupby(['Country Name'])['PM10 (μg/m3)'].mean()[row['Country Name']] if pd.isna(row['PM10 (μg/m3)']) else row['PM10 (μg/m3)'],axis=1)

#Take the PM2.5 based on Country to fill the null values
df['PM2.5 (μg/m3)'] = df.apply(lambda row: df.groupby(['Country Name'])['PM2.5 (μg/m3)'].mean()[row['Country Name']] if pd.isna(row['PM2.5 (μg/m3)']) else row['PM2.5 (μg/m3)'],axis=1)

#Take the NO2 based on Country to fill the null values
df['NO2 (μg/m3)'] = df.apply(lambda row: df.groupby(['Country Name'])['NO2 (μg/m3)'].mean()[row['Country Name']] if pd.isna(row['NO2 (μg/m3)']) else row['NO2 (μg/m3)'],axis=1)

#Take the PM25 temporal coverage mean based on Country to fill the null values
df['PM25 temporal coverage (%)'] = df.apply(lambda row: df.groupby(['Country Name'])['PM25 temporal coverage (%)'].mean()[row['Country Name']] if pd.isna(row['PM25 temporal coverage (%)']) else row['PM25 temporal coverage (%)'],axis=1)

#Take the PM10 temporal coverage mean based on Country to fill the null values
df['PM10 temporal coverage (%)'] = df.apply(lambda row: df.groupby(['Country Name'])['PM10 temporal coverage (%)'].mean()[row['Country Name']] if pd.isna(row['PM10 temporal coverage (%)']) else row['PM10 temporal coverage (%)'],axis=1)

#Take the NO2 temporal coverage mean based on Country to fill the null values
df['NO2 temporal coverage (%)'] = df.apply(lambda row: df.groupby(['Country Name'])['NO2 temporal coverage (%)'].mean()[row['Country Name']] if pd.isna(row['NO2 temporal coverage (%)']) else row['NO2 temporal coverage (%)'],axis=1)



#Based on Year to fill the null values
df['PM2.5 (μg/m3)'] = df.apply(lambda row: df.groupby(['Year'])['PM2.5 (μg/m3)'].mean()[row['Year']] if pd.isna(row['PM2.5 (μg/m3)']) else row['PM2.5 (μg/m3)'],axis=1)
df['PM10 (μg/m3)'] = df.apply(lambda row: df.groupby(['Year'])['PM10 (μg/m3)'].mean()[row['Year']] if pd.isna(row['PM10 (μg/m3)']) else row['PM10 (μg/m3)'],axis=1)
df['NO2 (μg/m3)'] = df.apply(lambda row: df.groupby(['Year'])['NO2 (μg/m3)'].mean()[row['Year']] if pd.isna(row['NO2 (μg/m3)']) else row['NO2 (μg/m3)'],axis=1)
df['PM25 temporal coverage (%)'] = df.apply(lambda row: df.groupby(['Year'])['PM25 temporal coverage (%)'].mean()[row['Year']] if pd.isna(row['PM25 temporal coverage (%)']) else row['PM25 temporal coverage (%)'],axis=1)
df['PM10 temporal coverage (%)'] = df.apply(lambda row: df.groupby(['Year'])['PM10 temporal coverage (%)'].mean()[row['Year']] if pd.isna(row['PM10 temporal coverage (%)']) else row['PM10 temporal coverage (%)'],axis=1)
df['NO2 temporal coverage (%)'] = df.apply(lambda row: df.groupby(['Year'])['NO2 temporal coverage (%)'].mean()[row['Year']] if pd.isna(row['NO2 temporal coverage (%)']) else row['NO2 temporal coverage (%)'],axis=1)




#make categories with the 6th columns
category=[1,2, 3,4]
df['level_pm2.5']=pd.cut(df['PM2.5 (μg/m3)'],[0,10.38,14.63,23.16,191.90], labels=category)
df['level_pm2.5']= pd.to_numeric(df['level_pm2.5'])

df['level_pm10']=pd.cut(df['PM10 (μg/m3)'],[0,18,23,46,540], labels=category)
df['level_pm10']=pd.to_numeric(df['level_pm10'])

df['level_no2']=pd.cut(df['NO2 (μg/m3)'],[0,14,20.05,30.57,210.68], labels=category)
df['level_no2'] =pd.to_numeric(df['level_no2'])

df['level_pm2.5coverage']=pd.cut(df['PM25 temporal coverage (%)'],[0,90.97,92.87,96.54,100], labels=category)
df['level_pm2.5coverage']=pd.to_numeric(df['level_pm2.5coverage'])

df['level_pm10coverage']=pd.cut(df['PM10 temporal coverage (%)'],[0,91.61,94.02,95.88,100], labels=category)
df['level_pm10coverage']=pd.to_numeric(df['level_pm10coverage'])

df['level_no2temporal']=pd.cut(df['NO2 temporal coverage (%)'],[0,93,95,98,100], labels=category)
df['level_no2temporal'] =pd.to_numeric(df['level_no2temporal'])



#The contamination_rank column
df['contamination_rank'] = (df[['level_pm2.5','level_pm10','level_no2','level_pm2.5coverage','level_pm10coverage','level_no2temporal']].sum(axis=1))/6


labels=[1,2,3]
df['contamination_rank']=pd.qcut(df['contamination_rank'],q=4,labels=False)





#PREPROCESSING

#Using StandarScaler to normalize the Columns
Sc=StandardScaler()

column=['PM2.5 (μg/m3)','PM10 (μg/m3)','NO2 (μg/m3)','PM25 temporal coverage (%)',
                    'PM10 temporal coverage (%)','NO2 temporal coverage (%)']

for i in column:
    df[i] = Sc.fit_transform(df[[i]])



#OneHotEncoder Year and Country Name
Ohe = OneHotEncoder(sparse=False)

transformed = Ohe.fit_transform(df[['Year','Country Name']])

#Ohe Column names
encoded_cols = Ohe.get_feature_names(['Year','Country Name'])



# Creating the DF
df_encoded = pd.DataFrame(transformed, columns= encoded_cols)

df.reset_index(drop=True, inplace=True)
df_encoded.reset_index(drop=True, inplace=True)

df_def = pd.concat([df,df_encoded],axis=1)




#Selecting the train_test_split
X = df_def.drop(['Country Name','City','Year','Updated Year', 'level_pm2.5', 'level_pm10', 'level_no2',
       'level_pm2.5coverage', 'level_pm10coverage', 'level_no2temporal',
       'contamination_rank'],axis=1)
y = df_def['contamination_rank']


X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


classifier = RandomForestClassifier()

classifier.fit(X_train,y_train)

#Make pickle file of the model
pickle.dump(classifier,open("model.pkl","wb"))

target_names =['0','1','2','3']
    

