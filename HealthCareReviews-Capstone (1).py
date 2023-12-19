#!/usr/bin/env python
# coding: utf-8

# # Capstone Requirements
# 
# Problem: Analyze sentiments in healthcare reviews.
# Objective: Develop a model to classify sentiments.
# Tasks:
# Data Preprocessing
# Sentiment Analysis Model
# Model Evaluation
# Insights & Visualization

# Data : healthcare_reviews

# # Load File using Read CSV command

# # Machine learning Pipeline
# 
# ### Step 1 : EDA 
# ### Step 2 : Check for a null and Imput with default values
# ### Step 3 : Punctuation removal, tokenization, Stop words removal
# ### Step 4 : Stemming or lemmatizing 
# ### Step 5 : Vectorize the data , represent in sparse matric format
# ### Step 6 : Choose a model, fit and train 
# ### Step 6.5 :Hyper parameter tuning if required 
# ### Step 7 : Test the model. 
# 

# In[1]:


import numpy as np
import pandas as pd
import string
import nltk


# In[2]:


raw_review_data = pd.read_csv('healthcare_reviews.csv')


# In[3]:


raw_review_data.head()


# In[4]:


raw_review_data.info()


# In[5]:


raw_review_data.describe()


# In[6]:


# What is the shape of the dataset?

print("Input data has {} rows and {} columns".format(len(raw_review_data), len(raw_review_data.columns)))

# How many rating 2, 3, 4, 5 are there out of 1000 ?

print("Out of {} rows, {} are 5, {} are 4,, {} are 3, {} are 2 and {} are 1".format(len(raw_review_data),
                                                       len(raw_review_data[raw_review_data['Rating']==5]),
                                                       len(raw_review_data[raw_review_data['Rating']==4]),
                                                       len(raw_review_data[raw_review_data['Rating']==3]),   
                                                       len(raw_review_data[raw_review_data['Rating']==2]),
                                                       len(raw_review_data[raw_review_data['Rating']==1]),
                                                     ))


# In[7]:


# How much missing data is there?

print("Number of null in Review_Text: {}".format(raw_review_data['Review_Text'].isnull().sum()))
print("Number of null in Rating: {}".format(raw_review_data['Rating'].isnull().sum()))


# In[8]:


# Fill missing values in 'Review_Text' column with a default value ('No Review')
default_value = 'No Review'
raw_review_data['Review_Text'].fillna(default_value, inplace=True)


# In[9]:


# check values in 'Review_Text' column with default value ('No Review')
len(raw_review_data[raw_review_data['Review_Text'] == 'No Review'])


# In[10]:


# How much missing data is there after imputation?

print("Number of null in Review_Text: {}".format(raw_review_data['Review_Text'].isnull().sum()))
print("Number of null in Rating: {}".format(raw_review_data['Rating'].isnull().sum()))


# # EDA through visualisation

# In[11]:


# Calculate the length of 'Review_Text' and store it in a new column 'Review_Length'
raw_review_data['Review_Length'] = raw_review_data['Review_Text'].apply(len)


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


g = sns.FacetGrid(raw_review_data,col='Rating')
g.map(plt.hist,'Review_Length')


# #### Looks like the distribution is similar for all ratings
# #### text length for all star rating ranges mostly ranges between 40 and 100 
# #### all ratings have a length <20 for 10-20 messages, they can be considered as outliers 

# In[14]:


#Create a boxplot of text length for each Rating category.
sns.boxplot(x='Rating',y='Review_Length',data=raw_review_data,palette='rainbow')


# In[15]:


# Create a boolean mask for rows where 'Review_Text' length is less than 30
#mask = raw_review_data['Review_Length'] < 30

# Use np.where to create a subset of the DataFrame based on the mask
#subset_df = raw_review_data[np.where(mask)]
subset_df = raw_review_data[raw_review_data['Review_Length'] < 30]
subset_df.head()


# In[16]:


# Create a bar plot using Seaborn
sns.barplot(x='Rating', y='Review_Length', data=subset_df)

# Set labels for the x and y axes
plt.xlabel('Rating')
plt.ylabel('Review Length')

# Show the plot
plt.show()


# In[17]:


sns.countplot(x='Rating',data=subset_df,palette='rainbow')


# In[18]:


# To use the ratings more effectively we are going change rating >4 to to "Good rating" 3 to neutral and < 3 to "Not good rating"
# so we can use them for our modelling appropriately
# Create a boolean mask for rows where 'Rating' is greater than 4 and 'Review_Text' is 'No Review'
mask = (raw_review_data['Rating'] >= 4) & (raw_review_data['Review_Text'] == 'No Review')

# Use np.where to conditionally update 'Review_Text'
raw_review_data['Review_Text'] = np.where(mask, 'this is a good rating', raw_review_data['Review_Text'])
len(raw_review_data[raw_review_data['Review_Text'] == 'this is a good rating'])


# In[19]:


# Create a boolean mask for rows where 'Rating' is equal to 3 and 'Review_Text' is 'No Review'
mask = (raw_review_data['Rating'] == 3) & (raw_review_data['Review_Text'] == 'No Review')

# Use np.where to conditionally update 'Review_Text'
raw_review_data['Review_Text'] = np.where(mask, 'this is a neutral rating', raw_review_data['Review_Text'])
len(raw_review_data[raw_review_data['Review_Text'] == 'this is a neutral rating'])


# In[20]:


# Create a boolean mask for rows where 'Rating' is equal to 3 and 'Review_Text' is 'No Review'
mask = (raw_review_data['Rating'] <=2) & (raw_review_data['Review_Text'] == 'No Review')

# Use np.where to conditionally update 'Review_Text'
raw_review_data['Review_Text'] = np.where(mask, 'this is a bad rating', raw_review_data['Review_Text'])
len(raw_review_data[raw_review_data['Review_Text'] == 'this is a bad rating'])


# In[21]:


g = sns.FacetGrid(raw_review_data,col='Rating')
g.map(plt.hist,'Review_Length')


# # Cleaning the text

# In[22]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[23]:


# Define the clean_text function
def cleaned_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    print(tokens)
    tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    cleaned_text = ' '.join(tokens)
    return cleaned_text


# In[24]:


#Apply clean_text function to Data set
raw_review_data['Review_Text_Cleaned'] = raw_review_data['Review_Text'].apply(cleaned_text)
raw_review_data.head(15)


# ##  Stemming and Lemmatization
# ### We will go with lemmatization for this project as the data set is not really huge

# In[25]:


# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to perform lemmatization on a text
def lemmatize_text(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    
    # Lemmatize each word and join them back into a sentence
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    
    return lemmatized_text


# In[26]:


raw_review_data['Review_Text_Cleaned'] = raw_review_data['Review_Text_Cleaned'].apply(lemmatize_text)


# ## Feature Engineering 

# In[27]:


# Define conditions and corresponding values for Review_Sentiment
conditions = [
    (raw_review_data['Rating'] > 3),
    (raw_review_data['Rating'] == 3),
    (raw_review_data['Rating'] < 3)
]

values = ['Positive', 'Neutral', 'Negative']

# Use numpy.where to create the Review_Sentiment column
raw_review_data['Review_Sentiment'] = np.select(conditions, values, default='Neutral')


# In[28]:


raw_review_data.head(20)


# ## Use TF_IDF vectorizer fit and transform the data 

# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[30]:


# Initialize the TF-IDF vectorizer with analyzer='word'
tfidf_vectorizer = TfidfVectorizer(analyzer='word')
# Fit and transform the 'Review_Text_Cleaned' column
tfidf_matrix = tfidf_vectorizer.fit_transform(raw_review_data['Review_Text_Cleaned'])


# In[31]:


# Convert the TF-IDF matrix into a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray())
tfidf_df.head(20)
print(tfidf_df.shape)


# In[44]:


tfidf_df.head(40)


# In[32]:


X_features = pd.concat([raw_review_data['Rating'], raw_review_data['Review_Length'], tfidf_df], axis=1)
X_features.head()


# ## Explore RandomForestClassifier through Holdout Set

# In[33]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X_features, raw_review_data['Review_Sentiment'], test_size=0.2)


# In[35]:


X_train


# In[36]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1)
X_train.columns = X_train.columns.astype(str)
rf_model = rf.fit(X_train, y_train)


# In[37]:


X_test.columns = X_test.columns.astype(str)
y_pred = rf_model.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred,average='macro')


# In[38]:


print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))


# In[39]:


X_test


# In[40]:


test_string = "I have mixed feelings about my experience"
test_data =[test_string]
pred_vec = tfidf_vectorizer.transform(test_data)
test_df = pd.DataFrame(pred_vec.toarray())
#test_df
df = pd.DataFrame({'Rating': [4], 'Review_Length': [len(test_string)]})
X_features_input =pd.concat([df,test_df],axis=1)
X_features_input.columns = X_features_input.columns.astype(str)
predicted_out = rf_model.predict(X_features_input)
predicted_out


# In[41]:


test_string = "I did not like my stay, was not good, no customer care"
test_data =[test_string]
pred_vec = tfidf_vectorizer.transform(test_data)
test_df = pd.DataFrame(pred_vec.toarray())
#test_df
df = pd.DataFrame({'Rating': [2], 'Review_Length': [len(test_string)]})
X_features_input =pd.concat([df,test_df],axis=1)
X_features_input.columns = X_features_input.columns.astype(str)
predicted_out = rf_model.predict(X_features_input)
predicted_out


# In[42]:


test_string = "Its a netural feeling, seems ok"
test_data =[test_string]
pred_vec = tfidf_vectorizer.transform(test_data)
test_df = pd.DataFrame(pred_vec.toarray())
#test_df
df = pd.DataFrame({'Rating': [3], 'Review_Length': [len(test_string)]})
X_features_input =pd.concat([df,test_df],axis=1)
X_features_input.columns = X_features_input.columns.astype(str)
predicted_out = rf_model.predict(X_features_input)
predicted_out


# In[43]:


# if tuning is required we can do it througn Grid search cv or random search cv 
#rf = RandomForestClassifier()
#param = {'n_estimators': [10, 150, 300], 'max_depth': [30, 60, 90, None]}

#gs = GridSearchCV(rf, param, cv=5, n_jobs=-1)
#gs_fit = gs.fit(Features,target)
#pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]

