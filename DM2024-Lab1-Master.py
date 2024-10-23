#!/usr/bin/env python
# coding: utf-8

# # Data Mining Lab 1
# In this lab session we will focus on the use of scientific computing libraries to efficiently process, transform, and manage data. We will also provide best practices and introduce visualization tools for effectively conducting big data analysis. Furthermore, we will show you how to implement basic classification techniques.

# ---

# ## Table of Contents
# 1. Data Source
# 2. Data Preparation
# 3. Data Transformation
#  - 3.1 Converting Dictionary into Pandas dataframe
#  - 3.2 Familiarizing yourself with the Data
# 4. Data Mining using Pandas
#  - 4.1 Dealing with Missing Values
#  - 4.2 Dealing with Duplicate Data
# 5. Data Preprocessing
#  - 5.1 Sampling
#  - 5.2 Feature Creation
#  - 5.3 Feature Subset Selection
#  - 5.4 Atrribute Transformation / Aggregation
#  - 5.5 Dimensionality Reduction
#  - 5.6 Discretization and Binarization
# 6. Data Exploration
# 7. Data Classification
# 8. Conclusion
# 9. References

# ---

# ## Introduction
# In this notebook I will explore a text-based, document-based [dataset](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) using scientific computing tools such as Pandas and Numpy. In addition, several fundamental Data Mining concepts will be explored and explained in details, ranging from calculating distance measures to computing term frequency vectors. Coding examples, visualizations and demonstrations will be provided where necessary. Furthermore, additional exercises are provided after special topics. These exercises are geared towards testing the proficiency of students and motivate students to explore beyond the techniques covered in the notebook. 

# ---

# ### Requirements
# Here are the computing and software requirements
# 
# #### Computing Resources
# - Operating system: Preferably Linux or MacOS
# - RAM: 8 GB
# - Disk space: Mininium 8 GB
# 
# #### Software Requirements
# Here is a list of the required programs and libraries necessary for this lab session:
# 
# ##### Language:
# - [Python 3+](https://www.python.org/download/releases/3.0/) (Note: coding will be done strictly on Python 3)
#     - We are using Python 3.9.6.
#     - You can use newer version, but use at your own risk.
#     
# ##### Environment:
# Using an environment is to avoid some library conflict problems. You can refer this [Setup Instructions](http://cs231n.github.io/setup-instructions/) to install and setup.
# 
# - [Anaconda](https://www.anaconda.com/download/) (recommended but not required)
#     - Install anaconda environment
#     
# - [Python virtualenv](https://virtualenv.pypa.io/en/stable/userguide/) (recommended to Linux/MacOS user)
#     - Install virtual environment
# 
# - [Kaggle Kernel](https://www.kaggle.com/kernels/)
#     - Run on the cloud  (with some limitations)
#     - Reference: [Kaggle Kernels Instructions](https://github.com/omarsar/data_mining_lab/blob/master/kagglekernel.md)
#     
# ##### Necessary Libraries:
# - [Jupyter](http://jupyter.org/) (Strongly recommended but not required)
#     - Install `jupyter` and Use `$jupyter notebook` in terminal to run
# - [Scikit Learn](http://scikit-learn.org/stable/index.html)
#     - Install `sklearn` latest python library
# - [Pandas](http://pandas.pydata.org/)
#     - Install `pandas` python library
# - [Numpy](http://www.numpy.org/)
#     - Install `numpy` python library
# - [Matplotlib](https://matplotlib.org/)
#     - Install `maplotlib` for python (version 3.7.3 recommended, pip install matplotlib==3.7.3)
# - [Plotly](https://plot.ly/)
#     - Install and signup for `plotly`
# - [Seaborn](https://seaborn.pydata.org/)
#     - Install and signup for `seaborn`
# - [NLTK](http://www.nltk.org/)
#     - Install `nltk` library
# - [PAMI](https://github.com/UdayLab/PAMI?tab=readme-ov-file)
#     - Install `PAMI` library
# - [UMAP](https://umap-learn.readthedocs.io/en/latest/)
#     - Install `UMAP` library

# ---

# In[2]:


# TEST necessary for when working with external scripts
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## 1. The Data
# In this notebook we will explore the popular 20 newsgroup dataset, originally provided [here](http://qwone.com/~jason/20Newsgroups/). The dataset is called "Twenty Newsgroups", which means there are 20 categories of news articles available in the entire dataset. A short description of the dataset, provided by the authors, is provided below:
# 
# - *The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. To the best of our knowledge, it was originally collected by Ken Lang, probably for his paper “Newsweeder: Learning to filter netnews,” though he does not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.*
# 
# If you need more information about the dataset please refer to the reference provided above. Below is a snapshot of the dataset already converted into a table. Keep in mind that the original dataset is not in this nice pretty format. That work is left to us. That is one of the tasks that will be covered in this notebook: how to convert raw data into convenient tabular formats using Pandas. 

# ![pic1.png](attachment:pic1.png)

# ---

# ## 2. Data Preparation
# In the following we will use the built-in dataset loader for 20 newsgroups from scikit-learn. Alternatively, it is possible to download the dataset manually from the website and use the sklearn.datasets.load_files function by pointing it to the 20news-bydate-train sub-folder of the uncompressed archive folder.
# 
# In order to get faster execution times for this first example we will work on a partial dataset with only 4 categories out of the 20 available in the dataset:

# In[3]:


# categories
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']


# In[4]:


# obtain the documents containing the categories provided
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                  shuffle=True, random_state=42) 
#This command also shuffles the data randomly, but with random_state we can bring the same distribution of data everytime 
#if we choose the same number, in this case "42". This is good for us, it means we can reproduce the same results every time
#we want to run the code.


# Let's take a look at some of the records that are contained in our subset of the data

# In[5]:


twenty_train.data[0:2]


# **Note** the `twenty_train` is just a bunch of objects that can be accessed as python dictionaries; so, you can do the following operations on `twenty_train`

# In[6]:


twenty_train.target_names


# In[7]:


len(twenty_train.data)


# In[8]:


len(twenty_train.filenames)


# #### We can also print an example from the subset

# In[9]:


# An example of what the subset contains
print("\n".join(twenty_train.data[0].split("\n")))


# ... and determine the label of the example via `target_names` key value

# In[10]:


print(twenty_train.target_names[twenty_train.target[0]])


# In[11]:


twenty_train.target[0]


# ... we can also get the category of 10 documents via `target` key value 

# In[12]:


# category of first 10 documents.
twenty_train.target[0:10]


# **Note:** As you can observe, both approaches above provide two different ways of obtaining the `category` value for the dataset. Ideally, we want to have access to both types -- numerical and nominal -- in the event some particular library favors a particular type. 
# 
# As you may have already noticed as well, there is no **tabular format** for the current version of the data. As data miners, we are interested in having our dataset in the most convenient format as possible; something we can manipulate easily and is compatible with our algorithms, and so forth.

# Here is one way to get access to the *text* version of the label of a subset of our training data:

# In[13]:


for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])


# ---

# ### **>>> Exercise 1 (Watch Video):**  
# In this exercise, please print out the *text* data for the first three samples in the dataset. (See the above code for help)

# In[14]:


# Answer here
for text in twenty_train.data[:3]:
    print(text)


# ---

# ## 3. Data Transformation
# So we want to explore and understand our data a little bit better. Before we do that we definitely need to apply some transformations just so we can have our dataset in a nice format to be able to explore it freely and more efficient. Lucky for us, there are powerful scientific tools to transform our data into that tabular format we are so farmiliar with. So that is what we will do in the next section--transform our data into a nice table format.

# ---

# ### 3.1 Converting Dictionary into Pandas Dataframe
# Here we will show you how to convert dictionary objects into a pandas dataframe. And by the way, a pandas dataframe is nothing more than a table magically stored for efficient information retrieval.

# In[15]:


import pandas as pd

# my functions
import helpers.data_mining_helpers as dmh

# construct dataframe from a list
X = pd.DataFrame.from_records(dmh.format_rows(twenty_train), columns= ['text'])


# In[16]:


len(X)


# In[17]:


X[0:2]


# In[18]:


for t in X["text"][:2]:
    print(t)


# ### Adding Columns

# One of the great advantages of a pandas dataframe is its flexibility. We can add columns to the current dataset programmatically with very little effort.

# In[19]:


# add category to the dataframe
X['category'] = twenty_train.target


# In[20]:


# add category label also
X['category_name'] = X.category.apply(lambda t: dmh.format_labels(t, twenty_train))


# Now we can print and see what our table looks like. 

# In[21]:


X[0:10]


# Nice! Isn't it? With this format we can conduct many operations easily and efficiently since Pandas dataframes provide us with a wide range of built-in features/functionalities. These features are operations which can directly and quickly be applied to the dataset. These operations may include standard operations like **removing records with missing values** and **aggregating new fields** to the current table (hereinafter referred to as a dataframe), which is desirable in almost every data mining project. Go Pandas!

# ---

# ### 3.2 Familiarizing yourself with the Data

# To begin to show you the awesomeness of Pandas dataframes, let us look at how to run a simple query on our dataset. We want to query for the first 10 rows (documents), and we only want to keep the `text` and `category_name` attributes or fields.

# In[22]:


# a simple query
X[:10][["text","category_name"]]


# Let us look at a few more interesting queries to familiarize ourselves with the efficiency and conveniency of Pandas dataframes.

# #### Let's query the last 10 records

# In[23]:


X[-10:]


# Ready for some sourcery? Brace yourselves! Let us see if we can query the first 10th record in our dataframe. For this we will use the build-in function called `loc`. This allows us to explicity define the columns you want to query.

# In[24]:


# using loc (by label)
X.loc[:10, 'text']


# You can also use the `iloc` function to query a selection of our dataset by position. Take a look at this [great discussion](https://stackoverflow.com/questions/28757389/pandas-loc-vs-iloc-vs-ix-vs-at-vs-iat/43968774) on the differences between the `iloc` and `loc` functions.

# In[25]:


# using iloc (by position)
X.iloc[:10, 0]


# ### **>>> Exercise 2 (take home):** 
# Experiment with other querying techniques using pandas dataframes. Refer to their [documentation](https://pandas.pydata.org/pandas-docs/stable/indexing.html) for more information. 

# In[26]:


#Answer here

# 1. Boolean indexing
city_emails = X[X['text'].str.contains('city.ac.uk')]
print(city_emails)
print("-----------------------------------------------------------------------------------")

# 2. Multiple conditions
condition = (X['category_name'].str.contains('sci.med')) & (X['text'].str.contains('city.ac.uk'))
michael_city = X[condition]
print(michael_city)
print("-----------------------------------------------------------------------------------")

# 3. String-based querying
long_subjects = X.query("text.str.len() > 50")
print(long_subjects)
print("-----------------------------------------------------------------------------------")

# Aggregate data by groupby
domain_counts = X['text'].str.split('@').str[1].value_counts()
print(domain_counts)
print("-----------------------------------------------------------------------------------")
# Pivot table
pivot_table = pd.pivot_table(X,  index='category_name', aggfunc='count')
print(pivot_table)
print("-----------------------------------------------------------------------------------")


# ---

# ### **>>> Exercise 3 (Watch Video):**  
# Try to fetch records belonging to the ```sci.med``` category, and query every 10th record. Only show the first 5 records.

# In[27]:


# Answer here
print(X[X['category_name'] == 'sci.med'].iloc[::10][0:5])


# ---

# ## 4. Data Mining using Pandas

# Let's do some serious work now. Let's learn to program some of the ideas and concepts learned so far in the data mining course. This is the only way we can convince ourselves of the true power of Pandas dataframes. 

# ### 4.1 Missing Values

# First, let us consider that our dataset has some *missing values* and we want to remove those values. In its current state our dataset has no missing values, but for practice sake we will add some records with missing values and then write some code to deal with these objects that contain missing values. You will see for yourself how easy it is to deal with missing values once you have your data transformed into a Pandas dataframe.
# 
# Before we jump into coding, let us do a quick review of what we have learned in the Data Mining course. Specifically, let's review the methods used to deal with missing values.
# 
# The most common reasons for having missing values in datasets has to do with how the data was initially collected. A good example of this is when a patient comes into the ER room, the data is collected as quickly as possible and depending on the conditions of the patients, the personal data being collected is either incomplete or partially complete. In the former and latter cases, we are presented with a case of "missing values". Knowing that patients data is particularly critical and can be used by the health authorities to conduct some interesting analysis, we as the data miners are left with the tough task of deciding what to do with these missing and incomplete records. We need to deal with these records because they are definitely going to affect our analysis or learning algorithms. So what do we do? There are several ways to handle missing values, and some of the more effective ways are presented below (Note: You can reference the slides - Session 1 Handout for the additional information).
# 
# - **Eliminate Data Objects** - Here we completely discard records once they contain some missing values. This is the easiest approach and the one we will be using in this notebook. The immediate drawback of going with this approach is that you lose some information, and in some cases too much of it. Now imagine that half of the records have at least one or more missing values. Here you are presented with the tough decision of quantity vs quality. In any event, this decision must be made carefully, hence the reason for emphasizing it here in this notebook. 
# 
# - **Estimate Missing Values** - Here we try to estimate the missing values based on some criteria. Although this approach may be proven to be effective, it is not always the case, especially when we are dealing with sensitive data, like **Gender** or **Names**. For fields like **Address**, there could be ways to obtain these missing addresses using some data aggregation technique or obtain the information directly from other databases or public data sources.
# 
# - **Ignore the missing value during analysis** - Here we basically ignore the missing values and proceed with our analysis. Although this is the most naive way to handle missing values it may proof effective, especially when the missing values includes information that is not important to the analysis being conducted. But think about it for a while. Would you ignore missing values, especially when in this day and age it is difficult to obtain high quality datasets? Again, there are some tradeoffs, which we will talk about later in the notebook.
# 
# - **Replace with all possible values** - As an efficient and responsible data miner, we sometimes just need to put in the hard hours of work and find ways to makes up for these missing values. This last option is a very wise option for cases where data is scarce (which is almost always) or when dealing with sensitive data. Imagine that our dataset has an **Age** field, which contains many missing values. Since **Age** is a continuous variable, it means that we can build a separate model for calculating the age for the incomplete records based on some rule-based approach or probabilistic approach.  

# As mentioned earlier, we are going to go with the first option but you may be asked to compute missing values, using a different approach, as an exercise. Let's get to it!
# 
# First we want to add the dummy records with missing values since the dataset we have is perfectly composed and cleaned that it contains no missing values. First let us check for ourselves that indeed the dataset doesn't contain any missing values. We can do that easily by using the following built-in function provided by Pandas.  

# In[28]:


# check missing values
X.isnull()


# The `isnull` function looks through the entire dataset for null values and returns `True` wherever it finds any missing field or record. As you will see above, and as we anticipated, our dataset looks clean and all values are present, since `isnull` returns **False** for all fields and records. But let us start to get our hands dirty and build a nice little function to check each of the records, column by column, and return a nice little message telling us the amount of missing records found. This excerice will also encourage us to explore other capabilities of pandas dataframes. In most cases, the build-in functions are good enough, but as you saw above when the entire table was printed, it is impossible to tell if there are missing records just by looking at preview of records manually, especially in cases where the dataset is huge. We want a more reliable way to achieve this. Let's get to it!

# In[29]:


X.isnull().apply(lambda x: dmh.check_missing_values(x))


# Okay, a lot happened there in that one line of code, so let's break it down. First, with the `isnull` we tranformed our table into the **True/False** table you see above, where **True** in this case means that the data is missing and **False** means that the data is present. We then take the transformed table and apply a function to each row that essentially counts to see if there are missing values in each record and print out how much missing values we found. In other words the `check_missing_values` function looks through each field (attribute or column) in the dataset and counts how many missing values were found. 
# 
# There are many other clever ways to check for missing data, and that is what makes Pandas so beautiful to work with. You get the control you need as a data scientist or just a person working in data mining projects. Indeed, Pandas makes your life easy!

# ---

# ### >>> **Exercise 4 (Watch Video):** 
# Let's try something different. Instead of calculating missing values by column let's try to calculate the missing values in every record instead of every column.  
# $Hint$ : `axis` parameter. Check the documentation for more information.

# In[30]:


# Answer here
X.isnull().apply(lambda x: dmh.check_missing_values(x), axis = 1)


# ---

# We have our function to check for missing records, now let us do something mischievous and insert some dummy data into the dataframe and test the reliability of our function. This dummy data is intended to corrupt the dataset. I mean this happens a lot today, especially when hackers want to hijack or corrupt a database.
# 
# We will insert a `Series`, which is basically a "one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.). The axis labels are collectively called index.", into our current dataframe.

# In[31]:


dummy_series = pd.Series(["dummy_record", 1], index=["text", "category"])


# In[32]:


dummy_series


# In[33]:


dummy_series.to_frame().T
# .to_frame() -> Convert Series to DataFrame
# .T          -> Transpose


# In[34]:


result_with_series = pd.concat([X, dummy_series.to_frame().T], ignore_index=True)


# In[35]:


# check if the records was commited into result
len(result_with_series)


# Now we that we have added the record with some missing values. Let try our function and see if it can detect that there is a missing value on the resulting dataframe.

# In[36]:


result_with_series.isnull().apply(lambda x: dmh.check_missing_values(x))


# Indeed there is a missing value in this new dataframe. Specifically, the missing value comes from the `category_name` attribute. As I mentioned before, there are many ways to conduct specific operations on the dataframes. In this case let us use a simple dictionary and try to insert it into our original dataframe `X`. Notice that above we are not changing the `X` dataframe as results are directly applied to the assignment variable provided. But in the event that we just want to keep things simple, we can just directly apply the changes to `X` and assign it to itself as we will do below. This modification will create a need to remove this dummy record later on, which means that we need to learn more about Pandas dataframes. This is getting intense! But just relax, everything will be fine!

# In[37]:


# dummy record as dictionary format
dummy_dict = [{'text': 'dummy_record',
               'category': 1
              }]


# In[38]:


X = pd.concat([X, pd.DataFrame(dummy_dict)], ignore_index=True)


# In[39]:


len(X)


# In[40]:


X.isnull().apply(lambda x: dmh.check_missing_values(x))


# So now that we can see that our data has missing values, we want to remove the records with missing values. The code to drop the record with missing that we just added, is the following:

# In[41]:


X.dropna(inplace=True)


# ... and now let us test to see if we gotten rid of the records with missing values. 

# In[42]:


X.isnull().apply(lambda x: dmh.check_missing_values(x))


# In[43]:


len(X)


# And we are back with our original dataset, clean and tidy as we want it. That's enough on how to deal with missing values, let us now move unto something more fun. 

# But just in case you want to learn more about how to deal with missing data, refer to the official [Pandas documentation](http://pandas.pydata.org/pandas-docs/stable/missing_data.html#missing-data).

# ---

# ### >>> **Exercise 5 (take home)** 
# There is an old saying that goes, "The devil is in the details." When we are working with extremely large data, it's difficult to check records one by one (as we have been doing so far). And also, we don't even know what kind of missing values we are facing. Thus, "debugging" skills get sharper as we spend more time solving bugs. Let's focus on a different method to check for missing values and the kinds of missing values you may encounter. It's not easy to check for missing values as you will find out in a minute.
# 
# Please check the data and the process below, describe what you observe and why it happened.   
# $Hint$ :  why `.isnull()` didn't work?

# In[44]:


import numpy as np

NA_dict = [{ 'id': 'A', 'missing_example': np.nan },
           { 'id': 'B'                    },
           { 'id': 'C', 'missing_example': 'NaN'  },
           { 'id': 'D', 'missing_example': 'None' },
           { 'id': 'E', 'missing_example':  None  },
           { 'id': 'F', 'missing_example': ''     }]

NA_df = pd.DataFrame(NA_dict, columns = ['id','missing_example'])
NA_df


# In[45]:


NA_df['missing_example'].isnull()


# In[46]:


# Answer here
"""
Data structure:
'A': np.nan (NumPy NaN)
'B': missing key
'C': 'NaN' (string)
'D': 'None' (string)
'E': None (Python None)
'F': '' (empty string)

Result:
'A' and 'B' return True (actual NaN values)
'E' return True (Python None)
'C', 'D', and 'F' return False (string)

Summary:
isnull() function identifies np.nan and Python None as null.
It doesn't recognize string representations of null values ('NaN', 'None') as null.
It doesn't consider empty strings as null.
"""


# ---

# ### 4.2 Dealing with Duplicate Data
# Dealing with duplicate data is just as painful as dealing with missing data. The worst case is that you have duplicate data that has missing values. But let us not get carried away. Let us stick with the basics. As we have learned in our Data Mining course, duplicate data can occur because of many reasons. The majority of the times it has to do with how we store data or how we collect and merge data. For instance, we may have collected and stored a tweet, and a retweet of that same tweet as two different records; this results in a case of data duplication; the only difference being that one is the original tweet and the other the retweeted one. Here you will learn that dealing with duplicate data is not as challenging as missing values. But this also all depends on what you consider as duplicate data, i.e., this all depends on your criteria for what is considered as a duplicate record and also what type of data you are dealing with. For textual data, it may not be so trivial as it is for numerical values or images. Anyhow, let us look at some code on how to deal with duplicate records in our `X` dataframe.

# First, let us check how many duplicates we have in our current dataset. Here is the line of code that checks for duplicates; it is very similar to the `isnull` function that we used to check for missing values. 

# In[47]:


X.duplicated()


# We can also check the sum of duplicate records by simply doing:

# In[48]:


sum(X.duplicated())


# Based on that output, you may be asking why did the `duplicated` operation only returned one single column that indicates whether there is a duplicate record or not. So yes, all the `duplicated()` operation does is to check per records instead of per column. That is why the operation only returns one value instead of three values for each column. It appears that we don't have any duplicates since none of our records resulted in `True`. If we want to check for duplicates as we did above for some particular column, instead of all columns, we do something as shown below. As you may have noticed, in the case where we select some columns instead of checking by all columns, we are kind of lowering the criteria of what is considered as a duplicate record. So let us only check for duplicates by only checking the `text` attribute. 

# In[49]:


sum(X.duplicated('text'))


# Now let us create some duplicated dummy records and append it to the main dataframe `X`. Subsequenlty, let us try to get rid of the duplicates.

# In[50]:


dummy_duplicate_dict = [{
                             'text': 'dummy record',
                             'category': 1, 
                             'category_name': "dummy category"
                        },
                        {
                             'text': 'dummy record',
                             'category': 1, 
                             'category_name': "dummy category"
                        }]


# In[51]:


X = pd.concat([X, pd.DataFrame(dummy_duplicate_dict)], ignore_index=True)


# In[52]:


len(X)


# In[53]:


sum(X.duplicated())


# We have added the dummy duplicates to `X`. Now we are faced with the decision as to what to do with the duplicated records after we have found it. In our case, we want to get rid of all the duplicated records without preserving a copy. We can simply do that with the following line of code:

# In[54]:


X.drop_duplicates(keep=False, inplace=True) # inplace applies changes directly on our dataframe


# In[55]:


len(X)


# Check out the Pandas [documentation](http://pandas.pydata.org/pandas-docs/stable/indexing.html?highlight=duplicate#duplicate-data) for more information on dealing with duplicate data.

# ---

# ## 5.  Data Preprocessing
# In the Data Mining course we learned about the many ways of performing data preprocessing. In reality, the list is quiet general as the specifics of what data preprocessing involves is too much to cover in one course. This is especially true when you are dealing with unstructured data, as we are dealing with in this particular notebook. But let us look at some examples for each data preprocessing technique that we learned in the class. We will cover each item one by one, and provide example code for each category. You will learn how to perform each of the operations, using Pandas, that cover the essentials to Preprocessing in Data Mining. We are not going to follow any strict order, but the items we will cover in the preprocessing section of this notebook are as follows:
# 
# - Aggregation
# - Sampling
# - Dimensionality Reduction
# - Feature Subset Selection
# - Feature Creation
# - Discretization and Binarization
# - Attribute Transformation

# ---

# ### 5.1 Sampling
# The first concept that we are going to cover from the above list is sampling. Sampling refers to the technique used for selecting data. The functionalities that we use to  selected data through queries provided by Pandas are actually basic methods for sampling. The reasons for sampling are sometimes due to the size of data -- we want a smaller subset of the data that is still representatitive enough as compared to the original dataset. 
# 
# We don't have a problem of size in our current dataset since it is just a couple thousand records long. But if we pay attention to how much content is included in the `text` field of each of those records, you will realize that sampling may not be a bad idea after all. In fact, we have already done some sampling by just reducing the records we are using here in this notebook; remember that we are only using four categories from the all the 20 categories available. Let us get an idea on how to sample using pandas operations.

# In[56]:


X_sample = X.sample(n=1000) #random state


# In[57]:


len(X_sample)


# In[58]:


X_sample[0:4]


# ---

# ### >>> Exercise 6 (take home):
# Notice any changes from the `X` dataframe to the `X_sample` dataframe? What are they? Report every change you noticed as compared to the previous state of `X`. Feel free to query and look more closely at the dataframe for these changes.

# In[59]:


# Answer here
"""
1. Size reduction
2. Random selection：The rows in X_sample are randomly selected from X
3. Preserved structure：The column structure remains the same.
4. Content diversity：Sampling maintained the same distribution of original.
5. Randomness：Unless setting random seed, each sample are differents
"""
# Analyze
print("1. Size reduction")
print("Original X shape:", X.shape)
print("X_sample shape:", X_sample.shape)

print("2. Random selection：The rows in X_sample are randomly selected from X")
print(X_sample.head())

print("3. Preserved structure：The column structure remains the same.")
print("Columns in X:", X.columns.tolist())
print("Columns in X_sample:", X_sample.columns.tolist())

print("4. Content diversity：Sampling maintained the same distribution of original.")
print("Category distribution in X:")
print(X['category_name'].value_counts(normalize=True))
print("Category distribution in X_sample:")
print(X_sample['category_name'].value_counts(normalize=True))

print("5. Randomness：Unless setting random seed, each sample are differents")
X_sample2 = X.sample(n=1000, random_state=None)
print("Are two samples identical?", X_sample.equals(X_sample2))

print("Additional Information:")
print("X info:")
X.info()
print("X_sample info:")
X_sample.info()
print("X describe:")
print(X.describe())
print("X_sample describe:")
print(X_sample.describe())


# ---

# Let's do something cool here while we are working with sampling! Let us look at the distribution of categories in both the sample and original dataset. Let us visualize and analyze the disparity between the two datasets. To generate some visualizations, we are going to use `matplotlib` python library. With matplotlib, things are faster and compatability-wise it may just be the best visualization library for visualizing content extracted from dataframes and when using Jupyter notebooks. Let's take a loot at the magic of `matplotlib` below.

# In[60]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[61]:


categories


# In[62]:


print(X.category_name.value_counts())

# plot barchart for X
X.category_name.value_counts().plot(kind = 'bar',
                                    title = 'Category distribution',
                                    ylim = [0, 700],        
                                    rot = 0, fontsize = 11, figsize = (8,3))


# In[63]:


print(X_sample.category_name.value_counts())

# plot barchart for X_sample
X_sample.category_name.value_counts().plot(kind = 'bar',
                                           title = 'Category distribution',
                                           ylim = [0, 300], 
                                           rot = 0, fontsize = 12, figsize = (8,3))


# You can use following command to see other available styles to prettify your charts.
# ```python
# print(plt.style.available)```

# ---

# ### >>> **Exercise 7 (Watch Video):**
# Notice that for the `ylim` parameters we hardcoded the maximum value for y. Is it possible to automate this instead of hard-coding it? How would you go about doing that? (Hint: look at code above for clues)

# In[64]:


# Answer here
upper_bound = max(X_sample.category_name.value_counts() + 20)

print(X_sample.category_name.value_counts())
plt.style.use('dark_background')
# plot barchart for X_sample
X_sample.category_name.value_counts().plot(kind = 'bar',
                                           title = 'Category distribution',
                                           ylim = [0, upper_bound], 
                                           rot = 0, fontsize = 12, figsize = (8,3))


# ---

# ### >>> **Exercise 8 (take home):** 
# We can also do a side-by-side comparison of the distribution between the two datasets, but maybe you can try that as an excerise. Below we show you an snapshot of the type of chart we are looking for. 

# ![alt txt](https://i.imgur.com/9eO431H.png)

# In[65]:


import numpy as np
plt.style.use('default')

categories = X.category_name.value_counts().index
counts_X = X.category_name.value_counts().values
counts_X_sample = X_sample.category_name.value_counts().reindex(categories, fill_value=0).values

x = np.arange(len(categories))
width = 0.2

plt.bar(x - width/2, counts_X, width, label='Original Data', color='blue')
plt.bar(x + width/2, counts_X_sample, width, label='Sample Data', color='orange')

plt.title('Category Distribution Comparison', fontsize=16)
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(x, categories, ha='right')

plt.legend()
plt.tight_layout()
plt.show()


#  

# One thing that stood out from the both datasets, is that the distribution of the categories remain relatively the same, which is a good sign for us data scientist. There are many ways to conduct sampling on the dataset and still obtain a representative enough dataset. That is not the main focus in this notebook, but if you would like to know more about sampling and how the `sample` feature works, just reference the Pandas documentation and you will find interesting ways to conduct more advanced sampling.

# ---

# ### 5.2 Feature Creation
# The other operation from the list above that we are going to practise on is the so-called feature creation. As the name suggests, in feature creation we are looking at creating new interesting and useful features from the original dataset; a feature which captures the most important information from the raw information we already have access to. In our `X` table, we would like to create some features from the `text` field, but we are still not sure what kind of features we want to create. We can think of an interesting problem we want to solve, or something we want to analyze from the data, or some questions we want to answer. This is one process to come up with features -- this process is usually called `feature engineering` in the data science community. 
# 
# We know what feature creation is so let us get real involved with our dataset and make it more interesting by adding some special features or attributes if you will. First, we are going to obtain the **unigrams** for each text. (Unigram is just a fancy word we use in Text Mining which stands for 'tokens' or 'individual words'.) Yes, we want to extract all the words found in each text and append it as a new feature to the pandas dataframe. The reason for extracting unigrams is not so clear yet, but we can start to think of obtaining some statistics about the articles we have: something like **word distribution** or **word frequency**.
# 
# Before going into any further coding, we will also introduce a useful text mining library called [NLTK](http://www.nltk.org/). The NLTK library is a natural language processing tool used for text mining tasks, so might as well we start to familiarize ourselves with it from now (It may come in handy for the final project!). In partcular, we are going to use the NLTK library to conduct tokenization because we are interested in splitting a sentence into its individual components, which we refer to as words, emojis, emails, etc. So let us go for it! We can call the `nltk` library as follows:
# 
# ```python
# import nltk
# ```

# In[66]:


import nltk


# In[67]:


nltk.download('punkt_tab')


# In[68]:


# takes a like a minute or two to process
X['unigrams'] = X['text'].apply(lambda x: dmh.tokenize_text(x))


# In[69]:


X[0:4]["unigrams"]


# If you take a closer look at the `X` table now, you will see the new columns `unigrams` that we have added. You will notice that it contains an array of tokens, which were extracted from the original `text` field. At first glance, you will notice that the tokenizer is not doing a great job, let us take a closer at a single record and see what was the exact result of the tokenization using the `nltk` library.

# In[70]:


X[0:4]


# In[71]:


list(X[0:1]['unigrams'])


# The `nltk` library does a pretty decent job of tokenizing our text. There are many other tokenizers online, such as [spaCy](https://spacy.io/), and the built in libraries provided by [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). We are making use of the NLTK library because it is open source and because it does a good job of segmentating text-based data. 

# ---

# ### 5.3 Feature subset selection
# Okay, so we are making some headway here. Let us now make things a bit more interesting. We are going to do something different from what we have been doing thus far. We are going use a bit of everything that we have learned so far. Briefly speaking, we are going to move away from our main dataset (one form of feature subset selection), and we are going to generate a document-term matrix from the original dataset. In other words we are going to be creating something like this. 

# ![alt txt](https://docs.google.com/drawings/d/e/2PACX-1vS01RrtPHS3r1Lf8UjX4POgDol-lVF4JAbjXM3SAOU-dOe-MqUdaEMWwJEPk9TtiUvcoSqTeE--lNep/pub?w=748&h=366)

# Initially, it won't have the same shape as the table above, but we will get into that later. For now, let us use scikit learn built in functionalities to generate this document. You will see for yourself how easy it is to generate this table without much coding. 

# In[72]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(X.text) #learn the vocabulary and return document-term matrix
print(X_counts[0])


# Now you can also see some examples of what each feature is based on their index in the vector:

# In[73]:


count_vect.get_feature_names_out()[14887]


# In[74]:


count_vect.get_feature_names_out()[29022]


# In[75]:


count_vect.get_feature_names_out()[8696]


# In[76]:


count_vect.get_feature_names_out()[4017]


# What we did with those two lines of code is that we transformed the articles into a **term-document matrix**. Those lines of code tokenize each article using a built-in, default tokenizer (often referred to as an `analyzer`) and then produces the word frequency vector for each document. We can create our own analyzers or even use the nltk analyzer that we previously built. To keep things tidy and minimal we are going to use the default analyzer provided by `CountVectorizer`. Let us look closely at this analyzer. 

# In[77]:


analyze = count_vect.build_analyzer()
analyze("I am craving for a hawaiian pizza right now")

# tokenization, remove stop words (e.g i, a, the), create n-gram (or unigram)


# ---

# ### **>>> Exercise 9 (Watch Video):**
# Let's analyze the first record of our X dataframe with the new analyzer we have just built. Go ahead try it!

# In[78]:


# Answer here
# How do we turn our array[0] text document into a tokenized text using the build_analyzer()?
analyze(X.text[0])


# ---

# Now let us look at the term-document matrix we built above.

# In[79]:


# We can check the shape of this matrix by:
X_counts.shape


# In[80]:


# We can obtain the feature names of the vectorizer, i.e., the terms
# usually on the horizontal axis
count_vect.get_feature_names_out()[0:10]


# ![alt txt](https://i.imgur.com/57gA1sd.png)

# Above we can see the features found in the all the documents `X`, which are basically all the terms found in all the documents. As I said earlier, the transformation is not in the pretty format (table) we saw above -- the term-document matrix. We can do many things with the `count_vect` vectorizer and its transformation `X_counts`. You can find more information on other cool stuff you can do with the [CountVectorizer](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction). 
# 
# Now let us try to obtain something that is as close to the pretty table I provided above. Before jumping into the code for doing just that, it is important to mention that the reason for choosing the `fit_transform` for the `CountVectorizer` is that it efficiently learns the vocabulary dictionary and returns a term-document matrix.
# 
# In the next bit of code, we want to extract the first five articles and transform them into document-term matrix, or in this case a 2-dimensional array. Here it goes. 

# In[81]:


X_counts.shape


# In[82]:


# we convert from sparse array to normal array
X_counts[0:5, 0:100].toarray()


# As you can see the result is just this huge sparse matrix, which is computationally intensive to generate and difficult to visualize. But we can see that the fifth record, specifically, contains a `1` in the beginning, which from our feature names we can deduce that this article contains exactly one `00` term.

# ---

# ### **>>> Exercise 10 (take home):**
# We said that the `1` at the beginning of the fifth record represents the `00` term. Notice that there is another 1 in the same record. Can you provide code that can verify what word this 1 represents from the vocabulary. Try to do this as efficient as possible.

# In[83]:


# Get all feature names
feature_names = count_vect.get_feature_names_out()
fifth_record = X_counts[4]
non_zero_indices = fifth_record.nonzero()[1]
words = feature_names[non_zero_indices]
answer = words[1]

print(f"The second '1' in the fifth record represents the word: '{answer}'")


# In[84]:


# find the second record index and print it 
second_one_index = X_counts[4].nonzero()[1][1]
count_vect.get_feature_names_out()[second_one_index]


# ---

# To get you started in thinking about how to better analyze your data or transformation, let us look at this nice little heat map of our term-document matrix. It may come as a surpise to see the gems you can mine when you start to look at the data from a different perspective. Visualization are good for this reason.

# In[85]:


# first twenty features only
plot_x = ["term_"+str(i) for i in count_vect.get_feature_names_out()[0:20]]


# In[86]:


# obtain document index
plot_y = ["doc_"+ str(i) for i in list(X.index)[0:20]]


# In[87]:


plot_z = X_counts[0:20, 0:20].toarray() #X_counts[how many documents, how many terms]
plot_z


# For the heat map, we are going to use another visualization library called `seaborn`. It's built on top of matplotlib and closely integrated with pandas data structures. One of the biggest advantages of seaborn is that its default aesthetics are much more visually appealing than matplotlib. See comparison below.

# ![alt txt](https://i.imgur.com/1isxmIV.png)

# The other big advantage of seaborn is that seaborn has some built-in plots that matplotlib does not support. Most of these can eventually be replicated by hacking away at matplotlib, but they’re not built in and require much more effort to build.
# 
# So without further ado, let us try it now!

# In[88]:


import seaborn as sns

df_todraw = pd.DataFrame(plot_z, columns = plot_x, index = plot_y)
plt.subplots(figsize=(9, 7))
ax = sns.heatmap(df_todraw,
                 cmap="PuRd",
                 vmin=0, vmax=1, annot=True)


# Check out more beautiful color palettes here: https://python-graph-gallery.com/197-available-color-palettes-with-matplotlib/

# ---

# ### **>>> Exercise 11 (take home):** 
# From the chart above, we can see how sparse the term-document matrix is; i.e., there is only one terms with **FREQUENCY** of `1` in the subselection of the matrix. By the way, you may have noticed that we only selected 20 articles and 20 terms to plot the histrogram. As an excersise you can try to modify the code above to plot the entire term-document matrix or just a sample of it. How would you do this efficiently? Remember there is a lot of words in the vocab. Report below what methods you would use to get a nice and useful visualization

# In[89]:


# 1. Randomly sample documents and terms
sampled_docs_indices = np.random.choice(range(X_counts.shape[0]), size=50, replace=False)
sampled_terms_indices = np.random.choice(range(X_counts.shape[1]), size=50, replace=False)

plot_x = ["term_"+str(i) for i in count_vect.get_feature_names_out()[sampled_terms_indices]]
plot_y = ["doc_"+ str(i) for i in sampled_docs_indices]
plot_z = X_counts[sampled_docs_indices, :][:, sampled_terms_indices].toarray()

df_todraw = pd.DataFrame(plot_z, columns = plot_x, index = plot_y)
plt.subplots(figsize=(12, 10))
ax = sns.heatmap(df_todraw, cmap="PuRd", vmin=0, vmax=1, annot=False)
plt.show()


# In[90]:


# 2: Filter terms based on frequency

freq_threshold = 10
term_frequencies = np.array(X_counts.sum(axis=0)).flatten()
frequent_terms = np.where(term_frequencies > freq_threshold)[0]
sampled_docs_indices = np.random.choice(range(X_counts.shape[0]), size=30, replace=False)

plot_x = ["term_" + str(i) for i in count_vect.get_feature_names_out()[frequent_terms]]
plot_y = ["doc_" + str(i) for i in sampled_docs_indices]
plot_z = X_counts[sampled_docs_indices, :][:, frequent_terms].toarray()

# Create the DataFrame for heatmap
df_todraw = pd.DataFrame(plot_z, columns=plot_x, index=plot_y)
plt.subplots(figsize=(12, 10))
ax = sns.heatmap(df_todraw, cmap="PuRd", vmin=0, vmax=1, annot=False)
plt.show()


# ---

# The great thing about what we have done so far is that we now open doors to new problems. Let us be optimistic. Even though we have the problem of sparsity and a very high dimensional data, we are now closer to uncovering wonders from the data. You see, the price you pay for the hard work is worth it because now you are gaining a lot of knowledge from what was just a list of what appeared to be irrelevant articles. Just the fact that you can blow up the data and find out interesting characteristics about the dataset in just a couple lines of code, is something that truly inspires me to practise Data Science. That's the motivation right there!

# ---

# ### 5.4 Attribute Transformation / Aggregation
# We can do other things with the term-vector matrix besides applying dimensionality reduction technique to deal with sparsity problem. Here we are going to generate a simple distribution of the words found in all the entire set of articles. Intuitively, this may not make any sense, but in data science sometimes we take some things for granted, and we just have to explore the data first before making any premature conclusions. On the topic of attribute transformation, we will take the word distribution and put the distribution in a scale that makes it easy to analyze patterns in the distrubution of words. Let us get into it!

# First, we need to compute these frequencies for each term in all documents. Visually speaking, we are seeking to add values of the 2D matrix, vertically; i.e., sum of each column. You can also refer to this process as aggregation, which we won't explore further in this notebook because of the type of data we are dealing with. But I believe you get the idea of what that includes.  

# ![alt txt](https://docs.google.com/drawings/d/e/2PACX-1vTMfs0zWsbeAl-wrpvyCcZqeEUf7ggoGkDubrxX5XtwC5iysHFukD6c-dtyybuHnYigiRWRlRk2S7gp/pub?w=750&h=412)

# In[91]:


# note this takes time to compute. You may want to reduce the amount of terms you want to compute frequencies for
term_frequencies = []
for j in range(0,X_counts.shape[1]):
    term_frequencies.append(sum(X_counts[:,j].toarray()))

#[3, 8, 5, 2, 5, 8, 2, 5, 3, 2]


# In[92]:


term_frequencies = np.asarray(X_counts.sum(axis=0))[0]


# In[93]:


term_frequencies[0] #sum of first term: 00


# In[94]:


plt.subplots(figsize=(100, 10))
g = sns.barplot(x=count_vect.get_feature_names_out()[:300], 
            y=term_frequencies[:300])
g.set_xticklabels(count_vect.get_feature_names_out()[:300], rotation = 90);


# ---

# ### >>> **Exercise 12 (take home):**
# If you want a nicer interactive visualization here, I would encourage you try to install and use plotly to achieve this.

# In[95]:


# Answer here
import plotly.express as px

terms_to_plot = count_vect.get_feature_names_out()[:300]
frequencies = term_frequencies[:300]

fig = px.bar(
    x=terms_to_plot,
    y=frequencies,
    labels={'x': 'Terms', 'y': 'Frequencies'},
    title="Term Frequencies (Top 300 Terms)"
)

fig.update_layout(
    xaxis_tickangle=-90,
    height=600,
    width=1000 
)

fig.show()


# ---

# ### >>> **Exercise 13 (take home):** 
# The chart above only contains 300 vocabulary in the documents, and it's already computationally intensive to both compute and visualize. Can you efficiently reduce the number of terms you want to visualize as an exercise. 
# 

# In[96]:


# Answer here
import plotly.express as px
import numpy as np

min_freq = 500
max_freq = 7000 
term_frequencies = np.asarray(X_counts.sum(axis=0)).flatten()
filtered_indices = np.where((term_frequencies >= min_freq) & (term_frequencies <= max_freq))[0]

top_terms_indices = filtered_indices[:100]

terms_to_plot = count_vect.get_feature_names_out()[top_terms_indices]
frequencies_to_plot = term_frequencies[top_terms_indices]

fig = px.bar(
    x=terms_to_plot,
    y=frequencies_to_plot,
    labels={'x': 'Terms', 'y': 'Frequencies'},
    title=f"Term Frequencies (Top 100 Terms)"
)

fig.update_layout(
    xaxis_tickangle=-90,
    height=600,
    width=1000 
)

fig.show()


# ---

# ### >>> **Exercise 14 (take home):** 
# Additionally, you can attempt to sort the terms on the `x-axis` by frequency instead of in alphabetical order. This way the visualization is more meaninfgul and you will be able to observe the so called [long tail](https://en.wikipedia.org/wiki/Long_tail) (get familiar with this term since it will appear a lot in data mining and other statistics courses). see picture below
# 
# ![alt txt](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Long_tail.svg/1000px-Long_tail.svg.png)

# In[97]:


# Answer here
import plotly.express as px

min_freq = 500
max_freq = 7000
term_frequencies = np.asarray(X_counts.sum(axis=0)).flatten()
filtered_indices = np.where((term_frequencies >= min_freq) & (term_frequencies <= max_freq))[0]
sorted_indices = filtered_indices[np.argsort(-term_frequencies[filtered_indices])]

# Limit to top N terms
top_n = 100
top_terms_indices = sorted_indices[:top_n]

terms_to_plot = count_vect.get_feature_names_out()[top_terms_indices]
frequencies_to_plot = term_frequencies[top_terms_indices]

fig = px.bar(
    x=terms_to_plot,
    y=frequencies_to_plot,
    labels={'x': 'Terms', 'y': 'Frequencies'},
    title=f"Term Frequencies (Top {top_n} Terms)"
)

fig.update_layout(
    xaxis_tickangle=-90,
    height=600,
    width=1000 
)

fig.show()


# ---

# Since we already have those term frequencies, we can also transform the values in that vector into the log distribution. All we need is to import the `math` library provided by python and apply it to the array of values of the term frequency vector. This is a typical example of attribute transformation. Let's go for it. The log distribution is a technique to visualize the term frequency into a scale that makes you easily visualize the distribution in a more readable format. In other words, the variations between the term frequencies are now easy to observe. Let us try it out!

# In[98]:


import math
term_frequencies_log = [math.log(i) for i in term_frequencies]


# In[99]:


plt.subplots(figsize=(100, 10))
g = sns.barplot(x=count_vect.get_feature_names_out()[:300],
                y=term_frequencies_log[:300])
g.set_xticklabels(count_vect.get_feature_names_out()[:300], rotation = 90);


# Besides observing a complete transformation on the disrtibution, notice the scale on the y-axis. The log distribution in our unsorted example has no meaning, but try to properly sort the terms by their frequency, and you will see an interesting effect. Go for it!

# ### >>> **Exercise 15 (take home):** 
# You can copy the code from the previous exercise and change the 'term_frequencies' variable for the 'term_frequencies_log', comment about the differences that you observe and talk about other possible insights that we can get from a log distribution.

# In[100]:


# Answer here
"""
The key differences between original and log-transformed term frequencies are:
1. Scaling: Log transformation compresses the scale, reducing the dominance of high-frequency terms and making it easier to compare less common terms.
2. Spread: The original distribution is skewed, with a few dominant terms. Log transformation flattens this, spreading out the term frequencies more evenly.
3. Visibility: High-frequency terms in the original plot obscure lower-frequency ones, while the log-transformed plot reveals the entire distribution, including rare terms.

This technique enhances insights by highlighting mid-frequency terms and significant rare terms, improving the comparison of term distributions for deeper analysis.
"""
plt.subplots(figsize=(100, 10))
g = sns.barplot(x=count_vect.get_feature_names_out()[:300],
                y=term_frequencies[:300])
g.set_xticklabels(count_vect.get_feature_names_out()[:300], rotation = 90);


# ###  Finding frequent patterns
# Perfect, so now that we know how to interpret a document-term matrix from our text data, we will see how to get extra insight from it, we will do this by mining frequent patterns. For this we will be using the PAMI library that we previously installed.

# **Introduction to PAMI**
# 
# PAMI (PAttern MIning) is a Python-based library designed to empower data scientists by providing the necessary tools to uncover hidden patterns within large datasets. Unlike other pattern mining libraries that are Java-based (such as WEKA and SPMF), PAMI caters specifically to the Python environment, making it more accessible for data scientists working with Big Data. The goal of PAMI is to streamline the process of discovering patterns that are often hidden within large datasets, offering a unified platform for applying various pattern mining techniques. In the library you can find a lot of implementations from current state-of-the-art algorithms, all of them cater to different type of data, they can be: transactional data, temporal data, utility data and some others. You can find more information in the following github: [PAMI](https://github.com/UdayLab/PAMI?tab=readme-ov-file). For the purpose of our lab we will be modeling our text data as a transactional type. So let's get into it.
# 
# **Transactional Data**
# 
# In order to apply pattern mining techniques, we first need to convert our text data into transactional data. A transactional database is a set of transactions where each transaction consists of a unique identifier (TID) and a set of items. For instance, think of a transaction as a basket of items purchased by a customer, and the TID is like the receipt number. Each transaction could contain items such as "apple", "banana", and "orange".
# 
# Here's an example of a transactional database:
# 
# TID	Transactions
# 1	a, b, c
# 2	d, e
# 3	a, e, f
# 
# In this structure:
# TID refers to the unique identifier of each transaction (often ignored by PAMI to save storage space).
# Items refer to the elements in each transaction, which could be either integers or strings (e.g., products, words, etc.).
# When preparing text data, we need to transform sentences or documents into a similar format, where each sentence or document becomes a transaction, and the words within it become the items.
# 
# **Frequent Pattern Mining**
# 
# After converting the text into a transactional format, we can then apply frequent pattern mining. This process identifies patterns or combinations of items that occur frequently across the dataset. For example, in text data, frequent patterns might be common word pairs or phrases that appear together across multiple documents. Important term to learn: **Minimum Support**: It refers to the minimum frequency that a transaction has to have to be considered a pattern in our scenario.
# 
# PAMI allows us to mine various types of patterns, but for the purpuse of this lab we will explore the following types:
# 
# 
# **Patterns Above Minimum Support:** These are all patterns that meet a specified minimum support threshold. The result set can be quite large as it includes all frequent patterns, making it ideal for comprehensive analysis but potentially complex.
# 
# **Maximal Frequent Patterns:** These are the largest frequent patterns that cannot be extended by adding more items without reducing their frequency below the minimum support threshold. The result set is smaller and more concise, as it only includes the largest patterns, reducing redundancy.
# 
# **Top-K Frequent Patterns:** These patterns represent the K most frequent patterns, regardless of the minimum support threshold. The result set is highly focused and concise, with a fixed number of patterns, making it ideal when prioritizing the most frequent patterns.
# 
# ![freq_patterns_alg.png](attachment:freq_patterns_alg.png)
# 
# In the following steps, we will guide you through how to convert text data into transactional form and mine frequent patterns from it.
# 

# In our scenario, what we need is to mine patterns that can be representative to **each category**, in this way we will be able to differentiate each group of data more easily, for that we will need to first modify our document-term matrix to be able to work for each category, for this we will do the following:

# In[101]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#Create separate DataFrames for each category
categories = X['category_name'].unique()  # Get unique category labels
category_dfs = {}  # Dictionary to store DataFrames for each category

for category in categories:
    # Filter the original DataFrame by category
    category_dfs[category] = X[X['category_name'] == category].copy()

# Function to create term-document frequency DataFrame for each category
def create_term_document_df(df):
    count_vect = CountVectorizer()  # Initialize the CountVectorizer
    X_counts = count_vect.fit_transform(df['text'])  # Transform the text data into word counts
    
    # Get the unique words (vocabulary) from the vectorizer
    words = count_vect.get_feature_names_out()
    
    # Create a DataFrame where rows are documents and columns are words
    term_document_df = pd.DataFrame(X_counts.toarray(), columns=words)
    
    return term_document_df

# Create term-document frequency DataFrames for each category
filt_term_document_dfs = {}  # Dictionary to store term-document DataFrames for each category

for category in categories:
    filt_term_document_dfs[category] = create_term_document_df(category_dfs[category])


# In[102]:


# Display the filtered DataFrame for one of the categories, feel free to change the number in the vector
category_number=0 #You can change it from 0 to 3
print(f"Filtered Term-Document Frequency DataFrame for Category {categories[category_number]}:")
filt_term_document_dfs[categories[category_number]]


# Now we can see the number of unique words per category based on the column number in the new dataframe, feel free to **explore the changes of each category changing the vector number at the end**.

# In the past sections we saw the behaviour of each word frequency in the documents, but we still want to generalize a little bit more so we can observe and determine the data that we are going to use to mine the patterns. For this we will group the terms in bins and we are going to plot their frequency. Again, feel free to change the category number to explore the different results.

# In[103]:


# Sum over all documents to get total frequency for each word
category_number=0 #You can change it from 0 to 3
word_counts = filt_term_document_dfs[categories[category_number]].sum(axis=0).to_numpy()

# Visualize the frequency distribution
plt.figure(figsize=(10, 6))
plt.hist(word_counts, bins=5000, color='blue', edgecolor='black')
plt.title(f'Term Frequency Distribution for Category {categories[category_number]}')
plt.xlabel('Frequency')
plt.ylabel('Number of Terms')
plt.xlim(1, 200)
plt.show()


# From this graph, we can see that most of the words appear very infrequently across the entire dataset, while a small number of words appear quite often. When we're trying to find patterns in text data, we focus on combinations of words that are most helpful for classifying the documents. However, very rare words or extremely common words (like stopwords: 'the,' 'in,' 'a,' 'of,' etc.) don’t usually give us much useful information. To improve our results, we can filter out these words. Specifically, we'll remove the **bottom 1%** of the least frequent words and the **top 5%** of the most frequent ones. This helps us focus on words that might reveal more valuable patterns.
# 
# In this case, the choice of filtering the top 5% and bottom 1% is **arbitrary**, but in other applications, domain knowledge might guide us to filter words differently, depending on the type of text classification we're working on.
# 
# Let us look first at the words that we will be filtering based on the set percentage threshold.

# In[104]:


category_number=0 #You can change it from 0 to 3
word_counts = filt_term_document_dfs[categories[category_number]].sum(axis=0).to_numpy()

# Sort the term frequencies in descending order
sorted_indices = np.argsort(word_counts)[::-1]  # Get indices of sorted frequencies
sorted_counts = np.sort(word_counts)[::-1]  # Sort frequencies in descending order

# Calculate the index corresponding to the top 5% most frequent terms
total_terms = len(sorted_counts)
top_5_percent_index = int(0.05 * total_terms)

# Get the indices of the top 5% most frequent terms
top_5_percent_indices = sorted_indices[:top_5_percent_index]

# Filter terms that belong to the top 5% based on their rank
filtered_words = [filt_term_document_dfs[categories[category_number]].iloc[:, i].name for i in top_5_percent_indices]

print(f"Category: {categories[category_number]}")
print(f"Number of terms in top 5%: {top_5_percent_index}")
print(f"Filtered terms: {filtered_words}")


# Here we can explore the frequencies of the **top 5%** words:

# In[105]:


sorted_counts #We can see the frequencies sorted in a descending order


# In[106]:


sorted_indices #This are the indices corresponding to the words after being sorted in a descending order


# In[107]:


filt_term_document_dfs[categories[category_number]].loc[:,'the'].sum(axis=0) #Here we can sum up the column corresponding to the top 5% words, we just specify which one first.


# In[108]:


category_number=0 #You can change it from 0 to 3
word_counts = filt_term_document_dfs[categories[category_number]].sum(axis=0).to_numpy()

# Sort the term frequencies in ascending order and get sorted indices
sorted_indices = np.argsort(word_counts)  # Get indices of sorted frequencies
sorted_counts = word_counts[sorted_indices]  # Sort frequencies

# Calculate the index corresponding to the bottom 1% least frequent terms
total_terms = len(sorted_counts)
bottom_1_percent_index = int(0.01 * total_terms)

# Get the indices of the bottom 1% least frequent terms
bottom_1_percent_indices = sorted_indices[:bottom_1_percent_index]

# Filter terms that belong to the bottom 1% based on their rank
filtered_words = [filt_term_document_dfs[categories[category_number]].iloc[:, i].name for i in bottom_1_percent_indices]

print(f"Category: {categories[category_number]}")
print(f"Number of terms in bottom 1%: {bottom_1_percent_index}")
print(f"Filtered terms: {filtered_words}")


# Here we can explore the frequencies of the **bottom 1%** words:

# In[109]:


sorted_counts #We can see the frequencies sorted in an ascending order


# In[110]:


sorted_indices #This are the indices corresponding to the words after being sorted in an ascending order


# In[111]:


filt_term_document_dfs[categories[category_number]].loc[:,'l14h11'].sum(axis=0) #Here we can sum up the column corresponding to the bottom 1% words, we just specify which one first.


# Well done, now that we have seen what type of words are inside the thresholds we set, then we can procede to **filter them out of the dataframe**. If you want to experiment after you complete the lab, you can return to try different percentages to filter, or not filter at all to do all the subsequent tasks for the pattern minings, and see if there is a significant change in the result.

# In[112]:


category_number=0 #You can change it from 0 to 3

# Filter the bottom 1% and top 5% words based on their sum across all documents
def filter_top_bottom_words_by_sum(term_document_df, top_percent=0.05, bottom_percent=0.01):
    # Calculate the sum of each word across all documents
    word_sums = term_document_df.sum(axis=0)
    
    # Sort the words by their total sum
    sorted_words = word_sums.sort_values()
    
    # Calculate the number of words to remove
    total_words = len(sorted_words)
    top_n = int(top_percent * total_words)
    bottom_n = int(bottom_percent * total_words)
    
    # Get the words to remove from the top 5% and bottom 1%
    words_to_remove = pd.concat([sorted_words.head(bottom_n), sorted_words.tail(top_n)]).index
    print(f'Bottom {bottom_percent*100}% words: \n{sorted_words.head(bottom_n)}') #Here we print which words correspond to the bottom percentage we filter
    print(f'Top {top_percent*100}% words: \n{sorted_words.tail(top_n)}') #Here we print which words correspond to the top percentage we filter
    # Return the DataFrame without the filtered words
    return term_document_df.drop(columns=words_to_remove)

# Apply the filtering function to each category
term_document_dfs = {}

for category in categories:
    print(f'\nFor category {category} we filter the following words:')
    term_document_dfs[category] = filter_top_bottom_words_by_sum(filt_term_document_dfs[category])

# Example: Display the filtered DataFrame for one of the categories
print(f"Filtered Term-Document Frequency DataFrame for Category {categories[category_number]}:")
term_document_dfs[categories[category_number]]


# ### >>> **Exercise 16 (take home):** 
# Review the words that were filtered in each category and comment about the differences and similarities that you can see.

# #### # Answer here
# ##### **Category 0:**
# The filtered terms include many technical words such as "jpeg," "mpeg," "unix," and various acronyms and file formats commonly associated with computing or digital media. Additionally, there are numbers and codes, possibly referring to screen resolutions or memory addresses. A few common words like "good," "book," and "user" were also filtered, because they are either too general. 
# ##### **Category 1:**
# The filtered terms largely consist of religious and spiritual language, including terms like "Christian," "Bible," "Jesus," and "church." There are also doctrinal words like "sin", "salvation", "faith", and "priest", these terms are removed because they appear too often.
# ##### **Category 2:**
# The filtered terms are primarily related to science, medicine, and biology, with words such as "disease," "patients," "treatment," and "research" standing out. Terms like "immune," "protein," and "symptoms" suggest a focus on biomedical topics, while neutral, technical, and clinical words like "data" and "process" were also filtered. 
# ##### **Category 3:**
# The filtered terms in this category include personal names, geographical locations, and common conversational words like "today," "because," and "actually." Some seemingly random words like "New York" and "Obama" appear as well, likely due to their frequent use in everyday discourse or news contexts. This category seems to remove overly specific names or places, ensuring a more generalized approach to content.
# ##### **Summary:**
# Different categories remove the words that are commonly used in their field, and remove some digits and rare used words.

# In[110]:


"""
# Answer here
Category 0:
    The filtered terms include many technical words such as "jpeg," "mpeg," "unix," and various acronyms and file formats commonly associated with computing or digital media. Additionally, there are numbers and codes, possibly referring to screen resolutions or memory addresses. A few common words like "good," "book," and "user" were also filtered, because they are either too general. 
Category 1:
    The filtered terms largely consist of religious and spiritual language, including terms like "Christian," "Bible," "Jesus," and "church." There are also doctrinal words like "sin", "salvation", "faith", and "priest", these terms are removed because they appear too often.
Category 2:
    The filtered terms are primarily related to science, medicine, and biology, with words such as "disease," "patients," "treatment," and "research" standing out. Terms like "immune," "protein," and "symptoms" suggest a focus on biomedical topics, while neutral, technical, and clinical words like "data" and "process" were also filtered. 
Category 3:
    The filtered terms in this category include personal names, geographical locations, and common conversational words like "today," "because," and "actually." Some seemingly random words like "New York" and "Obama" appear as well, likely due to their frequent use in everyday discourse or news contexts. This category seems to remove overly specific names or places, ensuring a more generalized approach to content.
Summary:
    Different categories remove the words that are commonly used in their field, and remove some digits and rare used words."""


# In[136]:


category_number = 1
filtered_shape = filt_term_document_dfs[categories[category_number]].shape
original_shape = term_document_dfs[categories[category_number]].shape
print(f"Filtered DataFrame shape: {filtered_shape}")
print(f"Original DataFrame shape: {original_shape}")

removed_terms = set(filt_term_document_dfs[categories[category_number]].columns) - set(term_document_dfs[categories[category_number]].columns)
print(f"Terms removed in filtering: {removed_terms}")


# Great! Now that our document-term frequency dataframe is ready, we can proceed with the frequent pattern mining process. To do this, we first need to convert our dataframe into a transactional database that the PAMI library can work with. We will generate a CSV file for each category to create this database.
# 
# A key step in this process is defining the threshold that determines when a value in the data is considered a transaction. As we observed in the previous cell, there are **many zeros** in our dataframe, which indicate that certain words do not appear in specific documents. With this in mind, we'll set the transactional threshold to be **greater than or equal to 1**. This means that for each document/transaction, we will include all the words that occur at least once (after filtering), ensuring that only relevant words are included in the pattern mining process. For your reference you can also check the following real world example that the PAMI library provides to review how they chose the threshold to generate the transactional data: [Air Pollution Analytics - Japan](https://colab.research.google.com/github/udayLab/PAMI/blob/main/notebooks/airPollutionAnalytics.ipynb). 
# 
# #### The next part of the code will take a couple of minutes to execute, for simplicity I already shared the resulting files from it, to continue onwards
# 
# #### Given that some students have been experiencing some errors with recent newer versions of PAMI after Oct 11, where they changed some directories of these functions, you can try to run the following block of code or uncomment the lines indicated inside to run the older version of the functions:

# In[ ]:


"""
from PAMI.extras.DF2DB import DenseFormatDF as db      #Uncomment this line and comment the line below if this block of code 
                                                        #gives you trouble
#from PAMI.extras.convert.DF2DB import DF2DB            

# Loop through the dictionary of term-document DataFrames
for category in term_document_dfs:
    # Replace dots with underscores in the category name to avoid errors in the file creation
    category_safe = category.replace('.', '_')
    
    # Create the DenseFormatDF object and convert to a transactional database
    obj = db.DenseFormatDF(term_document_dfs[category]) #Uncomment this line and comment the line below if this block of code 
                                                         #gives you trouble
    #obj = DF2DB(term_document_dfs[category])           
        
    obj.convert2TransactionalDatabase(f'td_freq_db_{category_safe}.csv', '>=', 1)
"""


# Now let us look into the stats of our newly created transactional databases, we will observe the following:
# 
# - **Database Size (Total Number of Transactions)**: Total count of transactions in the dataset.
# 
# - **Number of Items**: Total count of unique items available across all transactions.
# 
# - **Minimum Transaction Size**: Smallest number of items in any transaction, indicating the simplest transaction.
# 
# - **Average Transaction Size**: Mean number of items per transaction, showing the typical complexity.
# 
# - **Maximum Transaction Size**: Largest number of items in a transaction, representing the most complex scenario.
# 
# - **Standard Deviation of Transaction Size**: Measures variability in transaction sizes; higher values indicate greater diversity.
# 
# - **Variance in Transaction Sizes**: Square of the standard deviation, providing a broader view of transaction size spread.
# 
# - **Sparsity**: Indicates the proportion of possible item combinations that do not occur, with values close to 1 showing high levels of missing combinations.
# 
# With regards to the graphs we will have: 
# 
# - **Item Frequency Distribution**
#     - Y-axis (Frequency): Number of transactions an item appears in.
#     - X-axis (Number of Items): Items ranked by frequency.
# 
# - **Transaction Length Distribution**
#     - Y-axis (Frequency): Occurrence of transaction lengths.
#     - X-axis (Transaction Length): Number of items per transaction.

# In[145]:


from PAMI.extras.dbStats import TransactionalDatabase as tds
obj = tds.TransactionalDatabase('td_freq_db_comp_graphics.csv')
obj.run()
obj.printStats()
obj.plotGraphs()


# In[146]:


from PAMI.extras.dbStats import TransactionalDatabase as tds
obj = tds.TransactionalDatabase('td_freq_db_sci_med.csv')
obj.run()
obj.printStats()
obj.plotGraphs()


# In[147]:


from PAMI.extras.dbStats import TransactionalDatabase as tds
obj = tds.TransactionalDatabase('td_freq_db_soc_religion_christian.csv')
obj.run()
obj.printStats()
obj.plotGraphs()


# In[148]:


from PAMI.extras.dbStats import TransactionalDatabase as tds
obj = tds.TransactionalDatabase('td_freq_db_alt_atheism.csv')
obj.run()
obj.printStats()
obj.plotGraphs()


# Now that we have reviewed the stats of our databases, there are some things to notice from them, the total number of transactions refer to the amount of documents per category, the number of items refer to the amount of unique words encountered in each category, the transaction size refers to the amount of words per document that it can be found, and we can see that our databases are very sparse, this is the result of having many zeros in the first place when making the document-term matrix. 

# Why are these stats important? It is because we are going to use the FPGrowth algorithm from PAMI, and for that we need to determine the *minimum support* (frequency) that our algorithm will use to mine for patterns in our transactions. 
# 
# When we set a minimum support threshold (minSup) for finding frequent patterns, we are looking for a good balance. We want to capture important patterns that show real connections in the data, but we also want to avoid too many unimportant patterns. For this dataset, we've chosen a minSup of 9. We have done this after observing the following:
# 
# - **Item Frequency**: The first graph shows that most items don't appear very often in transactions. There's a sharp drop in how frequently items appear, which means our data has many items that aren't used much.
# 
# - **Transaction Length**: The second graph shows that most transactions involve a small number of items. The most common transaction sizes are small, which matches our finding that the dataset does not group many items together often.
# 
# By setting minSup at 9, we focus on combinations of items that show up in these smaller, more common transactions. This level is low enough to include items that show up more than just a few times, but it's high enough to leave out patterns that don't appear often enough to be meaningful. This helps us keep our results clear and makes sure the patterns we find are useful and represent what's really happening in the dataset. 
# 
# **This value works for all categories**. Now let's get into mining those patterns. For more information you can visit the FPGrowth example in PAMI for transactional data: [FPGrowth Example](https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/basic/FPGrowth.ipynb#scrollTo=pLV84IYcDHe3).

# In[149]:


from PAMI.frequentPattern.basic import FPGrowth as alg
minSup=9
obj1 = alg.FPGrowth(iFile='td_freq_db_sci_med.csv', minSup=minSup)
obj1.mine()
frequentPatternsDF_sci_med= obj1.getPatternsAsDataFrame()
print('Total No of patterns: ' + str(len(frequentPatternsDF_sci_med))) #print the total number of patterns
print('Runtime: ' + str(obj1.getRuntime())) #measure the runtime


# In[150]:


obj1.save('freq_patterns_sci_med_minSup9.txt') #save the patterns
frequentPatternsDF_sci_med


# In[151]:


from PAMI.frequentPattern.basic import FPGrowth as alg
minSup=9
obj2 = alg.FPGrowth(iFile='td_freq_db_alt_atheism.csv', minSup=minSup)
obj2.mine()
frequentPatternsDF_alt_atheism= obj2.getPatternsAsDataFrame()
print('Total No of patterns: ' + str(len(frequentPatternsDF_alt_atheism))) #print the total number of patterns
print('Runtime: ' + str(obj2.getRuntime())) #measure the runtime


# In[152]:


obj2.save('freq_patterns_alt_atheism_minSup9.txt') #save the patterns
frequentPatternsDF_alt_atheism


# In[153]:


from PAMI.frequentPattern.basic import FPGrowth as alg
minSup=9
obj3 = alg.FPGrowth(iFile='td_freq_db_comp_graphics.csv', minSup=minSup)
obj3.mine()
frequentPatternsDF_comp_graphics= obj3.getPatternsAsDataFrame()
print('Total No of patterns: ' + str(len(frequentPatternsDF_comp_graphics))) #print the total number of patterns
print('Runtime: ' + str(obj3.getRuntime())) #measure the runtime


# In[154]:


obj3.save('freq_patterns_comp_graphics_minSup9.txt') #save the patterns
frequentPatternsDF_comp_graphics


# In[155]:


from PAMI.frequentPattern.basic import FPGrowth as alg
minSup=9
obj4 = alg.FPGrowth(iFile='td_freq_db_soc_religion_christian.csv', minSup=minSup)
obj4.mine()
frequentPatternsDF_soc_religion_christian= obj4.getPatternsAsDataFrame()
print('Total No of patterns: ' + str(len(frequentPatternsDF_soc_religion_christian))) #print the total number of patterns
print('Runtime: ' + str(obj4.getRuntime())) #measure the runtime


# In[156]:


obj4.save('freq_patterns_soc_religion_minSup9.txt') #save the patterns
frequentPatternsDF_soc_religion_christian


# Now that we've extracted the transactional patterns from our databases, the next step is to integrate them effectively with our initial data for further analysis. One effective method is to identify and use only the unique patterns that are specific to each category. This involves filtering out any patterns that are common across multiple categories.
# 
# The reason for focusing on **unique patterns** is that they can **significantly improve the classification process**. When a document contains these distinctive patterns, it provides clear, category-specific signals that help our model more accurately determine the document's category. This approach ensures that the patterns we use enhance the model's ability to distinguish between different types of content.

# In[157]:


import pandas as pd

#We group together all of the dataframes related to our found patterns
dfs = [frequentPatternsDF_sci_med, frequentPatternsDF_soc_religion_christian, frequentPatternsDF_comp_graphics, frequentPatternsDF_alt_atheism]


# Identify patterns that appear in more than one category
# Count how many times each pattern appears across all dataframes
pattern_counts = {}
for df in dfs:
    for pattern in df['Patterns']:
        if pattern not in pattern_counts:
            pattern_counts[pattern] = 1
        else:
            pattern_counts[pattern] += 1

# Filter out patterns that appear in more than one dataframe
unique_patterns = {pattern for pattern, count in pattern_counts.items() if count == 1}
# Calculate the total number of patterns across all categories
total_patterns_count = sum(len(df) for df in dfs)
# Calculate how many patterns were discarded
discarded_patterns_count = total_patterns_count - len(unique_patterns)

# For each category, filter the patterns to keep only the unique ones
filtered_dfs = []
for df in dfs:
    filtered_df = df[df['Patterns'].isin(unique_patterns)]
    filtered_dfs.append(filtered_df)

# Merge the filtered dataframes into a final dataframe
final_pattern_df = pd.concat(filtered_dfs, ignore_index=True)

# Sort by support
final_pattern_df = final_pattern_df.sort_values(by='Support', ascending=False)

# Display the final result
print(final_pattern_df)
# Print the number of discarded patterns
print(f"Number of patterns discarded: {discarded_patterns_count}")


# We observed a significant number of patterns that were common across different categories, which is why we chose to discard them. The next step is to integrate these now category-specific patterns into our data. How will we do this? By converting the patterns into binary data within the columns of our document-term matrix. Specifically, we will check each document for the presence of each pattern. If a pattern is found in the document, we'll mark it with a '1'; if it's not present, we'll mark it with a '0'. This binary encoding allows us to effectively augment our data, enhancing its utility for subsequent classification tasks.

# In[158]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Convert 'text' column into term-document matrix using CountVectorizer
count_vect = CountVectorizer()
X_tdm = count_vect.fit_transform(X['text'])  # X['text'] contains your text data
terms = count_vect.get_feature_names_out()  # Original terms in the vocabulary

# Tokenize the sentences into sets of unique words
X['tokenized_text'] = X['text'].str.split().apply(set)

# Initialize the pattern matrix
pattern_matrix = pd.DataFrame(0, index=X.index, columns=final_pattern_df['Patterns'])

# Iterate over each pattern and check if all words in the pattern are present in the tokenized sentence
for pattern in final_pattern_df['Patterns']:
    pattern_words = set(pattern.split())  # Tokenize pattern into words
    pattern_matrix[pattern] = X['tokenized_text'].apply(lambda x: 1 if pattern_words.issubset(x) else 0)

# Convert the term-document matrix to a DataFrame for easy merging
tdm_df = pd.DataFrame(X_tdm.toarray(), columns=terms, index=X.index)

# Concatenate the original TDM and the pattern matrix to augment the features
augmented_df = pd.concat([tdm_df, pattern_matrix], axis=1)

augmented_df


# ### >>> **Exercise 17 (take home):** 
# Implement the FAE Top-K and MaxFPGrowth algorithms from the PAMI library to analyze the 'comp.graphics' category in our processed database. **Only implement the mining part of the algorithm and display the resulting patterns**, like we did with the FPGrowth algorithm after creating the new databases. For the FAE Top-K, run trials with k values of 500, 1000, and 1500, recording the runtime for each. For MaxFPGrowth, test minimum support thresholds of 3, 6, and 9, noting the runtime for these settings as well. Compare the patterns these algorithms extract with those from the previously implemented FPGrowth algorithm. Document your findings, focusing on differences and similarities in the outputs and performance. For this you can find the following google collabs for reference provided by their github repository here: [FAE Top-K](https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/topk/FAE.ipynb) and [MaxFPGrowth](https://colab.research.google.com/github/UdayLab/PAMI/blob/main/notebooks/frequentPattern/maximal/MaxFPGrowth.ipynb)

# In[ ]:


"""
Frequent patterns were generated successfully using frequentPatternGrowth algorithm
Total No of patterns: 3
Runtime: 0.007008790969848633
	Patterns	Support
0	words	    9
1	26	        9
2	needs	    9

"""


# In[ ]:


# FAE Top-K Implementation
from PAMI.frequentPattern.topk import FAE as fae_alg

K_values = [500, 1000, 1500]
for k in K_values:
    obj_fae = fae_alg.FAE(iFile='td_freq_db_comp_graphics.csv', k=k)
    obj_fae.mine()
    frequentPatternsDF_fae = obj_fae.getPatternsAsDataFrame()
    
    print(f'Total No of patterns for K={k}: ' + str(len(frequentPatternsDF_fae)))
    print(f'Runtime for K={k}: ' + str(obj_fae.getRuntime()))  # measure the runtime
    
    obj_fae.save(f'freq_patterns_comp_graphics_FAE_topK{k}.txt')


# In[168]:


# MaxFPGrowth Implementation
from PAMI.frequentPattern.maximal import MaxFPGrowth  as max_fp_alg

minSup_values = [3, 6, 9]
for minSup in minSup_values:
    obj_maxfp = max_fp_alg.MaxFPGrowth(iFile='td_freq_db_comp_graphics.csv', minSup=minSup)
    obj_maxfp.mine()
    frequentPatternsDF_maxfp = obj_maxfp.getPatternsAsDataFrame()
    
    print(f'Total No of patterns for minSup={minSup}: ' + str(len(frequentPatternsDF_maxfp)))
    print(f'Runtime for minSup={minSup}: ' + str(obj_maxfp.getRuntime()))  # measure the runtime
    
    obj_maxfp.save(f'freq_patterns_comp_graphics_MaxFP_minSup{minSup}.txt')
    frequentPatternsDF_comp_graphics


# ---

# ### 5.5 Dimensionality Reduction
# Dimensionality reduction is a powerful technique for tackling the "curse of dimensionality," which commonly arises due to data sparsity. This technique is not only beneficial for visualizing data more effectively but also simplifies the data by reducing the number of dimensions without losing significant information. For a deeper understanding, please refer to the additional notes provided.
# 
# We will start with **Principal Component Analysis (PCA)**, which is focused on finding a projection that captures the largest amount of variation in the data. PCA is excellent for linear dimensionality reduction and works well when dealing with Gaussian distributed data. However, its effectiveness diminishes with non-linear data structures.
# 
# Additionally, we will explore two advanced techniques suited for non-linear dimensionality reductions:
# 
# - **t-Distributed Stochastic Neighbor Embedding (t-SNE)**:
#     - Pros:
#         - Effective at revealing local data structures at many scales.
#         - Great for identifying clusters in data.
#     - Cons:
#         - Computationally intensive, especially with large datasets.
#         - Sensitive to parameter settings and might require tuning (e.g., perplexity).
#         
# - **Uniform Manifold Approximation and Projection (UMAP)**:
#     - Pros:
#         - Often faster than t-SNE and can handle larger datasets.
#         - Less sensitive to the choice of parameters compared to t-SNE.
#         - Preserves more of the global data structure while also revealing local structure.
#     - Cons:
#         - Results can still vary based on parameter settings and random seed.
#         - May require some experimentation to find the optimal settings for specific datasets.
#         
# These methods will be applied to visualize our data more effectively, each offering unique strengths to mitigate the issue of sparsity and allowing us to observe underlying patterns in our dataset.

# [PCA Algorithm](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
# [t-SNE Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
# [UMAP Algorithm](https://umap-learn.readthedocs.io/en/latest/basic_usage.html)
# 
# **Input:** Raw term-vector matrix
# 
# **Output:** Projections 

# So, let's experiment with something interesting, from our previous work we have our data with only the document-term frequency data and also the one with both the document-term frequency and the pattern derived data, let's try to create a 2D plot after applying these algorithms to our dataframes and see what comes out.

# In[169]:


#Applying dimensionality reduction with only the document-term frequency data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

#This might take a couple of minutes to execute
# Apply PCA, t-SNE, and UMAP to the data
X_pca_tdm = PCA(n_components=2).fit_transform(tdm_df.values)
X_tsne_tdm = TSNE(n_components=2).fit_transform(tdm_df.values)
X_umap_tdm = umap.UMAP(n_components=2).fit_transform(tdm_df.values)


# In[170]:


X_pca_tdm.shape


# In[171]:


X_tsne_tdm.shape


# In[172]:


X_umap_tdm.shape


# In[173]:


# Plot the results in subplots
col = ['coral', 'blue', 'black', 'orange']
categories = X['category_name'].unique() 

fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # Create 3 subplots for PCA, t-SNE, and UMAP
fig.suptitle('PCA, t-SNE, and UMAP Comparison')

# Define a function to create a scatter plot for each method
def plot_scatter(ax, X_reduced, title):
    for c, category in zip(col, categories):
        xs = X_reduced[X['category_name'] == category].T[0]
        ys = X_reduced[X['category_name'] == category].T[1]
        ax.scatter(xs, ys, c=c, marker='o', label=category)
    
    ax.grid(color='gray', linestyle=':', linewidth=2, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')

# Step 4: Create scatter plots for PCA, t-SNE, and UMAP
plot_scatter(axes[0], X_pca_tdm, 'PCA')
plot_scatter(axes[1], X_tsne_tdm, 't-SNE')
plot_scatter(axes[2], X_umap_tdm, 'UMAP')

plt.show()


# From the 2D PCA visualization above, we can see a slight "hint of separation in the data"; i.e., they might have some special grouping by category, but it is not immediately clear. In the t-SNE graph we observe a more scattered distribution, but still intermixing with all the categories. And with the UMAP graph, the limits for the data seem pretty well defined, two categories seem to have some points well differentiated from the other classes, but most of them remain intermixed. The algorithms were applied to the raw frequencies and this is considered a very naive approach as some words are not really unique to a document. Only categorizing by word frequency is considered a "bag of words" approach. Later on in the course you will learn about different approaches on how to create better features from the term-vector matrix, such as term-frequency inverse document frequency so-called TF-IDF.

# Now let's try in tandem with our pattern augmented data:

# In[174]:


#This might take a couple of minutes to execute
#Applying dimensionality reduction with both the document-term frequency data and the pattern derived data
# Apply PCA, t-SNE, and UMAP to the data
X_pca_aug = PCA(n_components=2).fit_transform(augmented_df.values)
X_tsne_aug = TSNE(n_components=2).fit_transform(augmented_df.values)
X_umap_aug = umap.UMAP(n_components=2).fit_transform(augmented_df.values)


# In[175]:


# Plot the results in subplots
col = ['coral', 'blue', 'black', 'orange']
categories = X['category_name'].unique() 

fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # Create 3 subplots for PCA, t-SNE, and UMAP
fig.suptitle('PCA, t-SNE, and UMAP Comparison')

# Define a function to create a scatter plot for each method
def plot_scatter(ax, X_reduced, title):
    for c, category in zip(col, categories):
        xs = X_reduced[X['category_name'] == category].T[0]
        ys = X_reduced[X['category_name'] == category].T[1]
        ax.scatter(xs, ys, c=c, marker='o', label=category)
    
    ax.grid(color='gray', linestyle=':', linewidth=2, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')

# Create scatter plots for PCA, t-SNE, and UMAP
plot_scatter(axes[0], X_pca_aug, 'PCA')
plot_scatter(axes[1], X_tsne_aug, 't-SNE')
plot_scatter(axes[2], X_umap_aug, 'UMAP')

plt.show()


# We can see that our PCA visualization hasn't changed much from the previous version. This is likely because the original document-term matrix still dominates what the algorithm captures, overshadowing the new binary pattern data we added.
# 
# Looking at the t-SNE graph, it might seem different at first glance. However, upon closer inspection, it's almost the same but mirrored along the y-axis, with only slight changes in how the data points are placed. This similarity might be due to the stability of the t-SNE algorithm. Even small changes in the data can result in embeddings that look different but are structurally similar, indicating that the binary patterns may not have significantly altered the relationships among the data points in high-dimensional space.
# 
# The UMAP visualization shows the most noticeable changes—it appears more compact. This compactness could be because UMAP uses a more complex distance metric, which might be making it easier to see differences between closer and further points. The binary patterns could also be helping to reduce noise within categories, resulting in clearer, more coherent groups. However, the categories still appear quite mixed together.
# 
# Remember, just because you can't see clear groups in these visualizations doesn’t mean that a machine learning model won’t be able to classify the data correctly. These techniques are mainly used to help us see and understand complex data in a simpler two or three-dimensional space. However, they have their limits and might not show everything a computer model can find in the data. So, while these tools are great for getting a first look at your data, always use more methods and analyses to get the full picture.

# ### >>> Exercise 18 (take home):
# Please try to reduce the dimension to 3, and plot the result use 3-D plot. Use at least 3 different angle (camera position) to check your result and describe what you found.
# 
# $Hint$: you can refer to Axes3D in the documentation.

# In[ ]:


# Answer here
"""
1. PCA
    Structure: Clear separation of blue and coral clusters, with orange points on the periphery and scattered black points.
    Insights: Strong linear relationships, distinct group structures, with some outliers in the orange category.
2. t-SNE
    Structure: Dense, well-defined clusters, with a core blue cluster and black/coral clusters showing local organization.
    Insights: Preserves local structure and neighborhood relationships, highlighting category separation.
3. Umap
    Structure: Complex manifold with hierarchical relationships between clusters and continuous transitions.
    Insights: Maintains local/global relationships, revealing subtle group transitions and sub-clusters.

Comparative Analysis:
    PCA: Best for global variance and linear relationships.
    t-SNE: Highlights local clusters effectively.
    UMAP: Preserves both local and global topology, revealing more complex relationships.
Each method confirms distinct categories, with UMAP showing finer relationships and t-SNE emphasizing local distinctness.
"""


# In[178]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pca_angles = [(15, 45), (30, 135), (40, 210)]
tsne_angles = [(25, 45), (30, 120), (40, 210)]
umap_angles = [(15, 60), (30, 160), (50, 240)]

fig = plt.figure(figsize=(24, 18))

# PCA
for i, (elev, azim) in enumerate(pca_angles):
    ax = fig.add_subplot(3, 3, i+1, projection='3d')
    plot_3d_scatter(ax, X_pca_tdm_3d, f'PCA (Angle {i+1})', elev, azim)
    
# t-SNE
for i, (elev, azim) in enumerate(tsne_angles):
    ax = fig.add_subplot(3, 3, i+4, projection='3d')
    plot_3d_scatter(ax, X_tsne_tdm_3d, f't-SNE (Angle {i+1})', elev, azim)
    
# UMAP 
for i, (elev, azim) in enumerate(umap_angles):
    ax = fig.add_subplot(3, 3, i+7, projection='3d')
    plot_3d_scatter(ax, X_umap_tdm_3d, f'UMAP (Angle {i+1})', elev, azim)

plt.tight_layout(pad=3.0)
fig.suptitle('3D Visualizations of Dimensionality Reduction Results', 
             fontsize=16, y=0.95)

plt.show()


# ---

# ### 5.6 Discretization and Binarization
# In this section we are going to discuss a very important pre-preprocessing technique used to transform the data, specifically categorical values, into a format that satisfies certain criteria required by particular algorithms. Given our current original dataset, we would like to transform one of the attributes, `category_name`, into four binary attributes. In other words, we are taking the category name and replacing it with a `n` asymmetric binary attributes. The logic behind this transformation is discussed in detail in the recommended Data Mining text book (please refer to it on page 58). People from the machine learning community also refer to this transformation as one-hot encoding, but as you may become aware later in the course, these concepts are all the same, we just have different prefrence on how we refer to the concepts. Let us take a look at what we want to achieve in code. 

# In[179]:


from sklearn import preprocessing, metrics, decomposition, pipeline, dummy


# In[180]:


mlb = preprocessing.LabelBinarizer()


# In[181]:


mlb.fit(X.category)


# In[182]:


X['bin_category'] = mlb.transform(X['category']).tolist()


# In[183]:


X[0:9]


# Take a look at the new attribute we have added to the `X` table. You can see that the new attribute, which is called `bin_category`, contains an array of 0's and 1's. The `1` is basically to indicate the position of the label or category we binarized. If you look at the first two records, the one is places in slot 2 in the array; this helps to indicate to any of the algorithms which we are feeding this data to, that the record belong to that specific category. 
# 
# Attributes with **continuous values** also have strategies to tranform the data; this is usually called **Discretization** (please refer to the text book for more inforamation).

# ---

# ### >>> **Exercise 19 (take home):**
# Try to generate the binarization using the `category_name` column instead. Does it work?

# In[185]:


# Answer here
"""
It works.
"""
mlb = preprocessing.LabelBinarizer()
mlb.fit(X.category_name)  # Fitting on the 'category_name' column
X['bin_category_name'] = mlb.transform(X['category_name']).tolist()  # Transform and assign
X[0:9]


# ---

# # 6. Data Exploration

# Sometimes you need to take a peek at your data to understand the relationships in your dataset. Here, we will focus in a similarity example. Let's take 3 documents and compare them.

# In[143]:


# We retrieve 3 sentences for a random record
document_to_transform_1 = []
random_record_1 = X.iloc[50]
random_record_1 = random_record_1['text']
document_to_transform_1.append(random_record_1)

document_to_transform_2 = []
random_record_2 = X.iloc[100]
random_record_2 = random_record_2['text']
document_to_transform_2.append(random_record_2)

document_to_transform_3 = []
random_record_3 = X.iloc[150]
random_record_3 = random_record_3['text']
document_to_transform_3.append(random_record_3)


# Let's look at our emails.

# In[144]:


print(document_to_transform_1)
print(document_to_transform_2)
print(document_to_transform_3)


# In[145]:


from sklearn.preprocessing import binarize

# Transform sentence with Vectorizers
document_vector_count_1 = count_vect.transform(document_to_transform_1)
document_vector_count_2 = count_vect.transform(document_to_transform_2)
document_vector_count_3 = count_vect.transform(document_to_transform_3)

# Binarize vectors to simplify: 0 for abscence, 1 for prescence
document_vector_count_1_bin = binarize(document_vector_count_1)
document_vector_count_2_bin = binarize(document_vector_count_2)
document_vector_count_3_bin = binarize(document_vector_count_3)

# print vectors
print("Let's take a look at the count vectors:")
print(document_vector_count_1.todense())
print(document_vector_count_2.todense())
print(document_vector_count_3.todense())


# In[146]:


from sklearn.metrics.pairwise import cosine_similarity

# Calculate Cosine Similarity
cos_sim_count_1_2 = cosine_similarity(document_vector_count_1, document_vector_count_2, dense_output=True)
cos_sim_count_1_3 = cosine_similarity(document_vector_count_1, document_vector_count_3, dense_output=True)
cos_sim_count_2_3 = cosine_similarity(document_vector_count_2, document_vector_count_3, dense_output=True)

cos_sim_count_1_1 = cosine_similarity(document_vector_count_1, document_vector_count_1, dense_output=True)
cos_sim_count_2_2 = cosine_similarity(document_vector_count_2, document_vector_count_2, dense_output=True)
cos_sim_count_3_3 = cosine_similarity(document_vector_count_3, document_vector_count_3, dense_output=True)

# Print 
print("Cosine Similarity using count bw 1 and 2: %(x)f" %{"x":cos_sim_count_1_2})
print("Cosine Similarity using count bw 1 and 3: %(x)f" %{"x":cos_sim_count_1_3})
print("Cosine Similarity using count bw 2 and 3: %(x)f" %{"x":cos_sim_count_2_3})

print("Cosine Similarity using count bw 1 and 1: %(x)f" %{"x":cos_sim_count_1_1})
print("Cosine Similarity using count bw 2 and 2: %(x)f" %{"x":cos_sim_count_2_2})
print("Cosine Similarity using count bw 3 and 3: %(x)f" %{"x":cos_sim_count_3_3})


# As expected, cosine similarity between a sentence and itself is 1. Between 2 entirely different sentences, it will be 0. 
# 
# We can assume that we have the more common features in the documents 1 and 3 than in documents 1 and 2. This reflects indeed in a higher similarity than that of sentences 1 and 3. 
# 

# ---

# # 7. Data Classification
# Data classification is one of the most critical steps in the final stages of the data mining process. After uncovering patterns, trends, or insights from raw data, classification helps organize and label the data into predefined categories. This step is crucial in making the mined data actionable, as it allows for accurate predictions and decision-making. For example, in text mining, classification can be used to categorize documents based on their content, like classifying news articles into categories such as sports, politics, or technology.
# Among various classification techniques, the **Naive Bayes classifier** is a simple yet powerful algorithm commonly used for text classification tasks. Specifically, the Multinomial Naive Bayes classifier is particularly suited for datasets where features are represented by term frequencies, such as a document-term matrix, like the one we have.
# 
# - **Multinomial Naive Bayes:**
#     The Multinomial Naive Bayes classifier works by assuming that the features (words or terms in text data) follow a multinomial distribution. In simple terms, it calculates the probability of a document belonging to a particular category based on the frequency of words in that document, assuming independence between words (the "naive" part of Naive Bayes). Despite this assumption, it often performs remarkably well for text data, especially when working with word count features. Now, when incorporating the binary matrix of patterns we have, it remains compatible because the binary values can be seen as a count of pattern occurrences (1 for present, 0 for absent). Although binary features are not true "counts," the Multinomial Naive Bayes classifier can still handle them without issue. For more information you can go to: [NB Classifier](https://hub.packtpub.com/implementing-3-naive-bayes-classifiers-in-scikit-learn/)
#     
# We will implement a Multinomial Naive Bayes, for that we first choose how to split our data, in this case we will follow a typical **70/30 split for the training and test set**. Let's see a comparison of what we obtain when classifying our data without patterns vs our data with the patterns.

# In[147]:


#Model with only the document-term frequency data
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Create a mapping from numerical labels to category names
category_mapping = dict(X[['category', 'category_name']].drop_duplicates().values)

# Convert the numerical category labels to text labels
target_names = [category_mapping[label] for label in sorted(category_mapping.keys())]

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(tdm_df, X['category'], test_size=0.3, random_state=42)


# In[148]:


X_train


# In[149]:


X_test


# In[150]:


# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names, digits=4))


# In[151]:


#Model with the augmented data
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Create a mapping from numerical labels to category names
category_mapping = dict(X[['category', 'category_name']].drop_duplicates().values)

# Convert the numerical category labels to text labels
target_names = [category_mapping[label] for label in sorted(category_mapping.keys())]

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(augmented_df, X['category'], test_size=0.3, random_state=42)


# In[152]:


X_train


# In[153]:


X_test


# In[154]:


# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names, digits=4))


# As you might have seen from the first model, the document-term matrix data already allows the model to classify it with great accuracy, but if we add the additional information provided by the patterns then we see a slightly better result to an already high score. While the document-term matrix captures individual word frequencies, the pattern matrix adds valuable information about co-occurrences and higher-level word combinations, providing complementary insights. This enhanced feature set allows the classifier to better differentiate between categories, particularly in cases where word frequencies alone might not be enough. 
# 
# So, now you know the importance of feature creation and pattern mining, it can give you an edge at the time of data classification.

# -----

# ## 8. Concluding Remarks

# Wow! We have come a long way! We can now call ourselves experts of Data Preprocessing. You should feel excited and proud because the process of Data Mining usually involves 70% preprocessing and 30% training learning models. You will learn this as you progress in the Data Mining course. I really feel that if you go through the exercises and challenge yourself, you are on your way to becoming a super Data Scientist. 
# 
# From here the possibilities for you are endless. You now know how to use almost every common technique for preprocessing with state-of-the-art tools, such as Pandas, Scikit-learn, UMAP and PAMI. You are now with the trend! 
# 
# After completing this notebook you can do a lot with the results we have generated. You can train algorithms and models that are able to classify articles into certain categories and much more. You can also try to experiment with different datasets, or venture further into text analytics by using new deep learning techniques such as word2vec. All of this will be presented in the next lab session. Until then, go teach machines how to be intelligent to make the world a better place. 

# ----

# ## 9. References

# - Pandas cook book ([Recommended for starters](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html))
# - [Pang-Ning Tan, Michael Steinbach, Vipin Kumar, Introduction to Data Mining, Addison Wesley](https://dl.acm.org/citation.cfm?id=1095618)
