#!/usr/bin/env python
# coding: utf-8

# #EDA to understand how consumer attributes and loan attributes influence the tendency of defaulters.

# Identify patterns which indicate if a client has difficulty paying their
# installments which may be used for taking actions such as:
# 
# 
# *   denying the loan
# *   reducing the amount of loan
# *   lending (to risky applicants) at a higher interest rate
# 
# This will ensure that the consumers capable of repaying the loan are not rejected. 
# Identification of such applicants using EDA is the aim of this case study.

# 
# 1.   Identify the driving factors (or driver variables)behind loan default, i.e. the variables which are strong indicators of default. 
# 2.   Utilise the knowledge for company portfolio and risk assessment.
# 
# 

# The various risk variables are:
# 
# 1.   income level
# 2.   Employment status
# 3.   clients with payment difficulties
# 
# 
# 
# 

# This dataset has 3 files :
# 1. 'application_data.csv' contains all the information of the client at the time of application.
# The data is about whether a client has payment difficulties.
# 2. 'previous_application.csv' contains information about the client’s previous loan data. 
# It contains the data whether the previous application had been Approved, Cancelled, Refused
# or Unused offer.
# 3. 'columns_description.csv' is a data dictionary which describes the meaning of the variables.

# **RESULTS EXPECTED BY LEARNERS**
#  
# 
# *   Present the overall approach of the analysis in a presentation. Mention the problem
# statement and the analysis approach briefly.
# 
# *   Identify the missing data and use appropriate methods to deal with it. (Remove columns/or replace it with an appropriate value) 
# 
# 
# 

# Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt


# In[ ]:





# Hint : Note that in EDA, since it is not necessary to replace the missing value, but if you have
# to replace the missing value, what should be the approach. Clearly mention the approach.
# ● Identify if there are outliers in the dataset. Also, mention why do you think it is an outlier.
# Again, remember that for this exercise, it is not necessary to remove any data points.
# 

# In[ ]:





# ● Explain the results of univariate, segmented univariate, bivariate analysis, etc. in business
# terms.
# 

# ● Identify if there is data imbalance in the data. Find the ratio of data imbalance.
# Hint : How will you analyse the data in case of data imbalance? You can plot more than one
# type of plot to analyse the different aspects due to data imbalance. For example, you can
# choose your own scale for the graphs, i.e. one can plot in terms of percentage or absolute
# value. Do this analysis for the ‘Target variable’ in the dataset ( clients with payment difficulties
# and all other cases). Use a mix of univariate and bivariate analysis etc.
# Hint : Since there are a lot of columns, you can run your analysis in loops for the appropriate
# columns and find the insights.
# 

# In[ ]:





# ● Find the top 10 correlation for the Client with payment difficulties and all other cases
# (Target variable). Note that you have to find the top correlation by segmenting the data frame
# w.r.t to the target variable and then find the top correlation for each of the segmented data
# and find if any insight is there. Say, there are 5+1(target) variables in a dataset: Var1, Var2,
# Var3, Var4, Var5, Target. And if you have to find a top 3 correlation, it can be: Var1 & Var2,
# Var2 & Var3, Var1 & Var3. Target variable will not feature in this correlation as it is a
# categorical variable and not a continuous variable which is increasing or decreasing.
# 

# 

# Reading the previous file

# Include visualisations and summarise the most important results in the presentation. You
# are free to choose the graphs which explain the numerical/categorical variables. Insights
# should explain why the variable is important for differentiating the clients with payment
# difficulties with all other cases.

# # Dataset Info: Sample Data Set containing application information

# In[2]:


# To read the previous application
application_df=pd.read_csv('application_data.csv')


# In[3]:


application_df


# Top 5 records of data

# In[4]:



application_df.head()


# Bottom 5 records of data

# In[5]:



application_df.tail()


# Check the various attributes of data like shape (rows and cols)

# In[6]:



application_df.shape


# List the column names

# In[7]:



application_df.columns


# Checking the data types of all the columns

# In[8]:



application_df.dtypes


#  To check duplicate records

# In[9]:



duplicate = application_df[application_df.duplicated()]
print('There are ' + str(len(duplicate)) + ' duplicate values in the datasets.')


# Check the descriptive statistics of numeric variables

# In[10]:



application_df.describe()


# To locate null values in data frame

# In[11]:


all_columns_list=[]
for i in application_df.columns:
    all_columns_list.append(i)
print('The columns in the dataset are: \n', all_columns_list)


# In[12]:


application_df.isnull().sum()


# To find columns with missing values

# In[13]:



columns_list=[]
for i in application_df.columns:
  if application_df[i].isnull().sum()>0:
    columns_list.append(i)
print('Columns having missing values are: \n', columns_list)


# In[14]:


missing = pd.DataFrame((application_df.isnull().sum())*100/application_df.shape[0]).reset_index()
print(missing)
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# 
# 
# 

# 

# # Data cleaning

# 1.Create a copy of base data for manupulation & processing

# In[15]:


application_df_c = application_df.copy()
application_df_c.head()

# Hence, dropping the rows of total 55374 have 'XNA' values in the organization type column

#new_pre_df=new_pre_df.drop(new_pre_df.loc[new_pre_df['NAME_CONTRACT_TYPE']=='XNA'].index)
#new_pre_df['NAME_CONTRACT_TYPE'].value_counts()


# Deleting unwanted columns

# In[16]:



unwanted_columns=['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21','FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL','REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']
application_df_c.drop(labels=unwanted_columns, axis=1, inplace=True )
application_df_c


# Removing rows having null values greater than or equal to 30%

# In[17]:


empty_rows=application_df_c.isnull().sum(axis=1)
empty_rows=list(empty_rows[empty_rows.values>=0.3*len(application_df_c)].index)
application_df_c.drop(labels=empty_rows,axis=0,inplace=True)
print(len(empty_rows))


# In[18]:


columns_list_missing=[]
for i in application_df_c.columns:
    if application_df_c[i].isnull().sum()>0:
        columns_list_missing.append(i)
print('The columns in the dataset are: \n', columns_list_missing)


# Fill missing values 

# In[19]:


# let's find these categorical columns having these 'XNA' values
    
# For Gender column

# Updating the column 'CODE_GENDER' with "F" for the dataset

application_df_c.loc[application_df_c['CODE_GENDER']=='XNA','CODE_GENDER']='F'
application_df_c['CODE_GENDER'].value_counts()


# In[20]:


# For Organization column

application_df_c[application_df_c['ORGANIZATION_TYPE']=='XNA'].shape
application_df_c=application_df_c.drop(application_df_c.loc[application_df_c['ORGANIZATION_TYPE']=='XNA'].index)


# In[21]:



# Used median method to fill the missing annuity amount, cnt_patment
application_df_c['AMT_ANNUITY'].fillna(application_df_c['AMT_ANNUITY'].median(), inplace = True)
application_df_c['AMT_GOODS_PRICE'].fillna(application_df_c['AMT_GOODS_PRICE'].median(), inplace = True)


# In[22]:


application_df_c.isna().sum()


# In[23]:


#To find the correlation after data cleaning


# In[24]:


def corrFilters(x: pd.DataFrame, bound: float):
    xCorr = x.corr()
    xFiltered = xCorr[((xCorr >= bound) | (xCorr <= -bound)) & (xCorr !=1.000)]
    xFlattened = xFiltered.unstack().sort_values().drop_duplicates()
    return xFlattened

new_corr_columns = corrFilters(application_df_c, .7)
print('Columns having high correlation after the data cleaning are: \n', new_corr_columns)


# In[ ]:





# # Feature Selection

# In[25]:


features_with_na =[features for features in application_df_c.columns if application_df_c[features].isnull().sum()>1]
features_with_na
for feature in features_with_na:
    print(feature,np.round(application_df_c[feature].isnull().median(),4),'% missing values')


# In[26]:


application_df_c.corr()


# In[27]:


#Creating bins for Amount annuity
# Get the max tenure
print(application_df_c['AMT_ANNUITY'].max())
bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000]
slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000']
application_df_c['AMT_ANNUITY_RANGE']=pd.cut(application_df_c['AMT_ANNUITY'],bins=bins,labels=slot)
application_df_c['AMT_ANNUITY_RANGE'].value_counts()


# In[28]:


#Creating bins for AMT_Income
print(application_df_c['AMT_INCOME_TOTAL'].max())

bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']
application_df_c['AMT_INCOME_TOTAL_RANGE']=pd.cut(application_df_c['AMT_INCOME_TOTAL'],bins=bins,labels=slot)


# In[29]:


#Creating bins for AMT_credit
print(application_df_c['AMT_CREDIT'].max())
bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']
application_df_c['AMT_CREDIT_RANGE']=pd.cut(application_df_c['AMT_CREDIT'],bins=bins,labels=slot)


# Imbalance ratio (IR), defined as the ratio of the number of instances in the majority class to the number of examples in the minority class.

# In[30]:


target0_df_application=application_df_c.loc[application_df_c["TARGET"]==0]
target1_df_application=application_df_c.loc[application_df_c["TARGET"]==1]
#1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample.
#0 - all other cases


# In[31]:


# Calculating Imbalance percentage
    
# Since the majority is target0 and minority is target1

Imbalance = round(len(target0_df_application)/len(target1_df_application),2)
print("The imbalance ratio is " +str(Imbalance))


# # Data exploration

# # Univariate analysis

# Categorical Univariate Analysis in logarithmic scale for target=0(client with no payment difficulties)

# In[131]:




# Count plotting in logarithmic scale

def uniplot(application_df_c,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(application_df_c[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = application_df_c, x= col, order=application_df_c[col].value_counts().index,hue = hue,palette='rocket') 
        
    plt.show()


# In[132]:


# PLotting for income range

uniplot(target0_df_application,col='AMT_INCOME_TOTAL_RANGE',title='Distribution of income range',hue='CODE_GENDER')


# In[133]:


# PLotting for credit range

uniplot(target0_df_application,col='AMT_CREDIT_RANGE',title='Distribution of credit range',hue='CODE_GENDER')


# In[35]:


uniplot(target0_df_application,col='AMT_ANNUITY_RANGE',title='Distribution of annuity amount range',hue='CODE_GENDER')


# In[134]:


#plotting contract type
uniplot(target0_df_application,col='NAME_CONTRACT_TYPE',title='Distribution of contract type',hue='CODE_GENDER')


# In[135]:


#plotting occupation type
uniplot(target0_df_application,col='OCCUPATION_TYPE',title='Distribution of occupation type',hue='CODE_GENDER')


# In[136]:


# Plotting for Organization type in logarithmic scale

sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30

plt.title("Distribution of Organization type for TARGET ==0")

plt.xticks(rotation=90)
plt.xscale('log')

sns.countplot(data=target0_df_application,y='ORGANIZATION_TYPE',order=target0_df_application['ORGANIZATION_TYPE'].value_counts().index,palette='crest')

plt.show()


# Categorical Univariate Analysis in logarithmic scale for target=1(client with  payment difficulties)

# In[147]:


uniplot(target1_df_application,col='AMT_INCOME_TOTAL_RANGE',title='Distribution of income range',hue='CODE_GENDER')


# In[137]:


#plotting occupation type
uniplot(target1_df_application,col='OCCUPATION_TYPE',title='Distribution of occupation type',hue='CODE_GENDER')


# In[138]:


#plotting contract type
uniplot(target1_df_application,col='NAME_CONTRACT_TYPE',title='Distribution of contract type',hue='CODE_GENDER')


# In[139]:


uniplot(target1_df_application,col='AMT_ANNUITY_RANGE',title='Distribution of annuity amount range',hue='CODE_GENDER')


# In[140]:


# PLotting for credit range

uniplot(target1_df_application,col='AMT_CREDIT_RANGE',title='Distribution of credit range',hue='CODE_GENDER')


# In[141]:


# Plotting for Organization type in logarithmic scale

sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30

plt.title("Distribution of Organization type for TARGET ==1")

plt.xticks(rotation=90)
plt.xscale('log')

sns.countplot(data=target1_df_application,y='ORGANIZATION_TYPE',order=target0_df_application['ORGANIZATION_TYPE'].value_counts().index,palette='crest')

plt.show()


# In[44]:


# Finding some correlation for numerical columns for both target 0 and 1 

target0_corr=target1_df_application.iloc[0:,2:]
target1_corr=target1_df_application.iloc[0:,2:]

target0=target0_corr.corr(method='spearman')
target1=target1_corr.corr(method='spearman')


# In[142]:


# Now, plotting the above correlation with heat map as it is the best choice to visulaize

# figure size

def targets_corr(data,title):
    plt.figure(figsize=(15, 10))
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['axes.titlepad'] = 70

# heatmap with a color map of choice


    sns.heatmap(data, cmap="YlOrBr",annot=False)

    plt.title(title)
    plt.yticks(rotation=0)
    plt.show()


# In[143]:


# For Target 0

targets_corr(data=target0,title='Correlation for target 0')


# In[144]:


# For Target 1

targets_corr(data=target1,title='Correlation for target 1')


# In[48]:


#For Target 0 - Finding any outliers


# In[49]:


# Box plotting for univariate variables analysis in logarithmic scale

def univariate_numerical(data,col,title):
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    plt.title(title)
    plt.yscale('log')
    sns.boxplot(data =target0_df_application, x=col,orient='v')
    plt.show()


# In[ ]:





# In[50]:


# Distribution of income amount

sns.boxplot(data=target0_df_application, x='AMT_INCOME_TOTAL')


# In[51]:


# Disrtibution of credit amount
sns.boxplot(data=target0_df_application, x='AMT_CREDIT')


# In[52]:


sns.boxplot(data=target0_df_application, x='AMT_ANNUITY')


# In[53]:


#For Target 1 - Finding any outliers


# In[54]:


sns.boxplot(data=target1_df_application, x='AMT_ANNUITY')


# In[55]:


sns.boxplot(data=target1_df_application, x='AMT_INCOME_TOTAL')


# In[56]:


sns.boxplot(data=target1_df_application, x='AMT_CREDIT')


# # Bivariate analysis for numerical variables
# 
# For Target 0

# In[146]:


# Box plotting for Credit amount

plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
sns.boxplot(data =target0_df_application, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Credit amount vs Education Status')
plt.show()


# In[58]:


plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
sns.boxplot(data =target0_df_application, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Income vs Education Status')
plt.show()


# In[59]:


plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
sns.boxplot(data =target0_df_application, x='NAME_CONTRACT_TYPE',y='AMT_ANNUITY', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Amout annuity vs Name contract type')
plt.show()


# For Target 1

# In[60]:


plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
sns.boxplot(data =target1_df_application, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Credit amount vs Education Status')
plt.show()


# In[61]:


plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
sns.boxplot(data =target1_df_application, x='NAME_CONTRACT_TYPE',y='AMT_ANNUITY', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Amount annuity vs Name contract type')
plt.show()


# In[62]:


plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
sns.boxplot(data =target1_df_application, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Income vs Education Status')
plt.show()


# In[63]:


appl_df=application_df.copy()


# # Reading the dataset of previous application

# In[64]:


previous_application_df=pd.read_csv("previous_application.csv")


# In[65]:


previous_application_df.head()


# In[ ]:





# In[66]:


empty_values = previous_application_df.isnull().sum()


# In[67]:


empty_values= list(empty_values[empty_values.values>=0.3].index)
previous_application_df.drop(labels=empty_values,axis=1,inplace=True)


# In[104]:


previous_application_df=previous_application_df.drop(previous_application_df[previous_application_df['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)
previous_application_df=previous_application_df.drop(previous_application_df[previous_application_df['NAME_CASH_LOAN_PURPOSE']=='XAP'].index)
previous_application_df=previous_application_df.drop(previous_application_df[previous_application_df['NAME_PAYMENT_TYPE']=='XNA'].index)


# In[105]:


pre_df1=previous_application_df.copy()
pre_df1


# In[106]:


# Now merging the Application dataset with previous appliaction dataset

#new_df=pd.merge(left=application_df_c,right=previous_application_df,how='inner',on='SK_ID_CURR')
df=pd.merge(left=appl_df,right=pre_df1,how='inner',on='SK_ID_CURR',suffixes=('_x'))


# In[107]:


df


# In[ ]:


#Renaming the column


# In[121]:


new_df1 =df.rename({'NAME_CONTRACT_TYPE':'NAME_CONTRACT_TYPE_','AMT_INCOME_TOTAL_':'AMT_INCOME_TOTAL','AMT_CREDIT_':'AMT_CREDIT','AMT_ANNUITY_':'AMT_ANNUITY','NAME_PAYMENT_TYPE_':'NAME_PAYMENT_TYPE','NAME_CLIENT_TYPE_':'NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY_':'NAME_GOODS_CATEGORY','AMT_CREDITx':'AMT_CREDIT_PREV','AMT_ANNUITYx':'AMT_ANNUITY_PREV'},axis=1)


# In[122]:


new_df1


# In[123]:


new_df1.drop(['SK_ID_CURR'],axis=1,inplace=True)


# Univariate analysis

# In[124]:


# Distribution of contract status in logarithmic scale

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of contract status with purposes')
ax = sns.countplot(data = new_df1, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=new_df1['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS',palette='mako') 


# In[125]:


# Distribution of payment type in logarithmic scale

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of payment type with purposes')
ax = sns.countplot(data = new_df1, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=new_df1['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_PAYMENT_TYPE',palette='crest') 


# Bivariate analysis

# In[128]:


# Box plotting for Credit amount in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
plt.yscale('log')
sns.boxplot(data =new_df1, x='NAME_CASH_LOAN_PURPOSE',hue='NAME_INCOME_TYPE',y='AMT_CREDIT',orient='v')
plt.title('Prev Credit amount vs Loan Purpose')
plt.show()


# In[129]:


# Box plotting for Credit amount prev vs Housing type in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
sns.barplot(data =new_df1, y='AMT_CREDIT',hue='TARGET',x='NAME_HOUSING_TYPE')
plt.title('Prev Credit amount vs Housing type')
plt.show()


# In[130]:


# Box plotting for Credit amount prev vs contract status in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
sns.barplot(data =new_df1, y='AMT_CREDIT',hue='TARGET',x='NAME_CONTRACT_STATUS')
plt.title('Prev Credit amount vs contract status')
plt.show()


# In[ ]:




