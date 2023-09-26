#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Libraries

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.express as px


# # 2. Joining the two DataFrames

# In[5]:


# Creating DataFrame 1
df1 = pd.read_csv(r"C:\Users\dnclv\OneDrive\Dokumenter\Data Analyst\Data Analyst Training\Datasets\Customer Churn E-Commerce Kaggle\ecommerce_customer_data_custom_ratios.csv")
df1


# In[6]:


# Creating DataFrame 2
df2 = pd.read_csv(r"C:\Users\dnclv\OneDrive\Dokumenter\Data Analyst\Data Analyst Training\Datasets\Customer Churn E-Commerce Kaggle\ecommerce_customer_data_large.csv")
df2


# In[7]:


# Merging DataFrame 1 and DataFrame 2 using a Full Outer join
df = df1.merge(df2, how="outer")


# In[8]:


df


# # 3. Understanding and Manipulation the Data

# In[9]:


# Number of columns and rows
df.shape


# In[10]:


# Info regarding the datasets
df.info()


# In[11]:


# Statistical summary of the data set
df.describe()


# In[12]:


# Renaming coloumns
df = df.rename(columns={'Customer ID': 'Customer_ID'})
df = df.rename(columns={'Purchase Date': 'Purchase_Date'})
df = df.rename(columns={'Product Price': 'Product_Price'})
df = df.rename(columns={'Total Purchase Amount': 'Total_Purchase_Amount'})
df = df.rename(columns={'Customer Age': 'Customer_Age'})
df = df.rename(columns={'Product Category': 'Product_Category'})
df = df.rename(columns={'Payment Method': 'Payment_Method'})
df = df.rename(columns={'Customer Name': 'Customer_Name'})
df.head()


# In[216]:


# Columns 'Customer_Age' and 'Age' is the same
df = df.drop('Age', axis='columns')


# In[154]:


# Purchase date is of type object, and should be datetime type
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])
# Print dtype of column 'Purchase Data'
df['Purchase_Date'].dtype


# In[155]:


# Find NA values
df.isna().sum()


# In[156]:


# Counting the unique vavlues in the dataset (E.g.Number of customers, Number of product categories)
df.nunique()


# In[157]:


# Check for duplicates
df.duplicated().sum()


# # 4. Data Analysis

# ## 4.1 RFM Analysis

# In[158]:


# To conduct the customer churn analysis, there will be focused on Recency, Frequency and Monetery Value (RFM model) to
# group customers into segments based on their purchasing behavior.


# In[159]:


# Gathering the current date time
today = dt.datetime(2023,9,18)


# In[160]:


# Creating a new dataframe called "rfm" with recency, frequency and monetary value
rfm = df.groupby('Customer_ID').agg({'Purchase_Date': lambda Purchase_Date: (today - Purchase_Date.max()).days,
                                     'Customer_ID': lambda Customer_ID: Customer_ID.value_counts(),
                                     'Total_Purchase_Amount': lambda Total_Purchase_Amount: Total_Purchase_Amount.sum()})


# In[161]:


rfm.head()


# In[1]:


# Renaming columns to the correct names.
rfm.columns = ['Recency', 'Frequency', 'Monetary_Value']


# In[163]:


rfm.head()


# In[164]:


# Creating labels for segmentation

rfm["recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['Monetary_Value'], 5, labels=[1, 2, 3, 4, 5])

rfm["rfm_score"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))


# In[165]:


# Creating the segmentation map (The map values is from https://documentation.bloomreach.com/engagement/docs/rfm-segmentation)
segmentation_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Cant Loose',
    r'3[1-2]': 'About to sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}
# Creating a High Level grouping of the segmentation map
high_level_cat = {
    "Low_Value": [
        segmentation_map[r'[1-2][1-2]'],
        segmentation_map[r'[1-2][3-4]'],
        segmentation_map[r'[1-2]5'],
        segmentation_map[r'3[1-2]']
    ],
    "Mid_Value": [
        segmentation_map[r'33'],
        segmentation_map[r'[3-4][4-5]'],
        segmentation_map[r'41'],
        segmentation_map[r'51']
    ],
    "High_Value": [
        segmentation_map[r'[4-5][2-3]'],
        segmentation_map[r'5[4-5]']
    ]
}


# In[166]:


# Creating segment column in rfm dataframe based on segmentation_map dictionary
rfm['Segment'] = rfm['rfm_score'].replace(segmentation_map, regex=True)


# In[167]:


# Creating category column in rfm dataframe based on high_level_cat dictionary
for category, segments in high_level_cat.items():
    for segment in segments:
        # Create a Boolean mask for customers in the current 'Segment'
        mask = rfm['Segment'] == segment
        
        # Assign the 'Category' value to customers in the 'Segment'
        rfm.loc[mask, 'Category'] = category


# In[168]:


# Creating a visualisation of the counts on each category
rfm_category_countplot = sns.countplot(data=rfm, x='Category', order=['Low_Value', 'Mid_Value', 'High_Value'])
rfm_category_countplot.set_xticklabels(rfm_category_countplot.get_xticklabels(), rotation=45)


# In[169]:


# Creating a visualisation of the counts on each segment
rfm_seg_map_countplot = sns.countplot(data=rfm, x='Segment')
rfm_seg_map_countplot.set_xticklabels(rfm_seg_map_countplot.get_xticklabels(), rotation=45)


# In[170]:


# Creating another visualisation of distribution - now with both category and segments
rfm_treemap_df = rfm.groupby(['Category', 'Segment']).size().reset_index(name='Count')

# Create a treemap
fig = px.treemap(rfm_treemap_df, 
                 path=['Category', 'Segment'], 
                 values='Count',
                 color='Count', 
                 color_continuous_scale='YlGnBu',
                 title='RFM Segment Treemap')

# Show the treemap
fig.show()


# In[171]:


# Creating a visualisation of the relationship between Frequency and Recency using Monetary Value as HUE
sns.scatterplot(data=rfm, x='Frequency', y='Recency', hue='Monetary_Value')


# In[172]:


# Subplots with three histograms side by side to show distribution in the rfm dateframe
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Recency Histogram
axes[0].hist(rfm['Recency'], bins=10, color='skyblue', alpha=0.7)
axes[0].set_title('Recency Histogram')
axes[0].set_xlabel('Recency')
axes[0].set_ylabel('Frequency')
axes[0].tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels

# Frequency Histogram
axes[1].hist(rfm['Frequency'], bins=10, color='lightgreen', alpha=0.7)
axes[1].set_title('Frequency Histogram')
axes[1].set_xlabel('Frequency')
axes[1].set_ylabel('Frequency')
axes[1].tick_params(axis='x', labelrotation=45) 

# Monetary Histogram
axes[2].hist(rfm['Monetary_Value'], bins=10, color='salmon', alpha=0.7)
axes[2].set_title('Monetary Histogram')
axes[2].set_xlabel('Monetary')
axes[2].set_ylabel('Frequency')
axes[2].tick_params(axis='x', labelrotation=45)

plt.tight_layout()
plt.show()


# In[173]:


rfm.describe()


# In[174]:


# Getting the median of the column "Recency" due to the right skewed distribution to see the average through median rather than mean
rfm["Recency"].median()


# ## 4.2 Customer  Analysis

# In[175]:


# The second part of the analysis is focused on the customer data, highlighting important insights regarding churn


# In[176]:


df.head()


# In[177]:


# Number of unique customers in the dataframe
df['Customer_ID'].nunique()


# In[217]:


# Copying the cleaned dataframe
df_cust = df.copy()


# In[218]:


df_cust.head()


# In[219]:


# Dropping dupicates on 'Customer_ID' to get unique rows
df_cust = df_cust.drop_duplicates(subset=['Customer_ID'])
df_cust


# In[220]:


# The number of churned customers (value = 1)
df_churned_values = df_cust['Churn'].value_counts()
print(df_churned_values)


# In[232]:


churned = df_churned_values[1]
active_cust = df_churned_values[0]
total_cust = churned + active_cust

print("customers who have churned:", churned)
print("active customers:", active_cust)
print("total customers:", total_cust)


# In[225]:


# churned and active customers divided into age three age groups
df_cust['Age_Range'] = pd.cut(df_cust['Customer_Age'], bins = [0, 19, 39, 59, 70], labels = ['0-19', '19-39', '40-59', '60+'])
df_cust_age_range = df_cust.groupby(['Age_Range', 'Churn'])['Customer_Age'].count()
print(df_cust_age_range)


# In[235]:


# Make additional date columns before exporting to CSV
df_cust['Day'] = df['Purchase_Date'].dt.day
df_cust['Month'] = df['Purchase_Date'].dt.month
df_cust['Year'] = df['Purchase_Date'].dt.year


# In[229]:


rfm.to_csv(r'C:\Users\dnclv\OneDrive\Dokumenter\Data Analyst\Data Analyst Training\PortfolioProjects Python\RFM', sep=',', index = None, header=True)
df_cust.to_csv(r'C:\Users\dnclv\OneDrive\Dokumenter\Data Analyst\Data Analyst Training\PortfolioProjects Python\dfcust', sep=',', index = None, header=True)


# In[ ]:




