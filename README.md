# Getting started with basics of Machine Learning in Python
Machine learning is a set of computational techniques and algorithms for detecting and exploiting structure and patterns in data in order to make interesting predictions or provide useful insights. It is used by businesses for a variety of purposes, including converting text from one language to another, detecting hazards in front of self-driving vehicles, and detecting fraudulent credit card purchases. Data science's influence stems from a thorough knowledge of mathematics and algorithms, as well as programming and hacking capabilities and communication abilities. Data Science, however, is about applying these three skill sets in a rigorous and disciplined manner.

## 1. The Data Science Process
When a non-technical boss wants you to solve a data puzzle, the job summary can be vague at first. As a data scientist, it is your responsibility to transform the challenge into a tangible dilemma, work out how to fix it, and bring the solution to all of the stakeholders. To do so, follow the steps in this workflow, known as the *"Data Science Process"*.  
   1. Define the issue: Who is your customer? What exactly is the problem that the customer wants you to solve? How do you turn their hazy request into a specific, well-defined issue?
   2. Gather the data you'll need to solve the problem: Is this information still on hand? If that's the case, what pieces of the data are most useful? If not, what additional information do you require? What resources (time, money, infrastructure) would be required to collect this information in a usable format?
   3. Data wrangling (processing): True, raw data is rarely available straight out of the box. You'll have to do with data processing bugs, corrupt documents, lost meanings, and a slew of other issues. To translate the data into a format that can be further analysed, you'll need to clean it first.
   4. Investigate the data: Once you've cleaned the files, you'll need to have a thorough understanding of the details it contains. What are the obvious trends or correlations in the data that you see? What are the top-level characteristics, and are there some that are more important than others?
   5. In-depth analysis (machine learning, statistical models, algorithms): This is the meat of the project, where you use all of the cutting-edge data analysis machinery to unearth high-value observations and forecasts.  

PS: Check out the link to know about the history of machine learning :) (https://bit.ly/3hjI7vr)  

## 2. Learn Python
Python 3 is a powerful programming language that is popular among web developers, data scientists, and software engineers alike. There are some compelling explanations for this! Your production level and competitiveness will skyrocket until you get the hang of it!
Python is open-source and has a large support group, as well as robust support repositories and user-friendly data structures.
Download and install the latest version of Python and Anaconda, using the following guides: 
https://realpython.com/installing-python/  
https://www.python.org/downloads/  
https://www.youtube.com/watch?v=YJC6ldI3hWk  
Learn about the python basics from main Data Types, Flow Control, Functions, Exception Handling, Lists, Dictionaries and Structuring Data
sets, Object oriented programming etc (https://www.youtube.com/embed/N4mEzFDjqtA?showinfo=0&rel=0&controls=1&autoplay=1)  

## 3. Data Wrangling
You'll almost often have to deal with sloppy or unfinished results while working on a data science project. The raw data we receive from various data sources is often unusable at first. Data wrangling or data munging refers to all of the work you perform on raw data to get it *"clean"* enough to feed into the analytical algorithm.  
PS: You should expect to do a lot of data wrangling if you want to build an effective ETL pipeline (extract, convert, and load) or make stunning data visualisations.
### 3.1 Data Wrangling With Pandas
Pandas is a widely used Python library for data manipulation. Read Julia Evans' Pandas Cookbook (which includes explanations of how to use Pandas to solve data problems) and learn the fundamentals of Python to get started with Pandas. (https://github.com/cellatwork-24x7/pandas-cookbook)   
**NOTE**: 1. Finding missed data points and dropping them from the dataset so that bias does not impair our study is an important aspect of data wrangling and data cleaning.  
 2. Filtering Data: You might be interested in dealing with a subset of data or filtering out pieces of data using certain parameters during the exploratory data analysis process.  
 3. Grouping Data: The DataFrameGroupBy object returned by Pandas groupby has a number of methods. value_count() returns the count of values for each of the column's unique values.  
 4. Time series data handling: When dealing with financial data, weather data, and other time dependent datasets, to see temperature trends or the variations between financial growth and recession, you'll want to go back in time. Thus create time series data to understand timeseries info, using Python's Pandas. Depending on the results of the study, you might need to adjust the timezone or resample.  
 5. Exporting data: When you're finished with your data collection, think about ways to express your findings in the most effective way possible. Consider the various data types that your coworkers and clients use. Ascertain the data is delivered in the most effective manner.   
PS: Refer the hands on tutorial in which Krishna Sankar demonstrates how to manipulate data with Pandas, the most common Python data manipulation library. (https://www.youtube.com/embed/jMrmYP7PcPM?showinfo=0&rel=0&controls=1&autoplay=1) (https://github.com/cellatwork-24x7/cautious-octo-waffle)
## 4.Supervised machine learning
One of the most popular types of machine learning algorithms is supervised learning. The data includes labelled samples of the idea you want the algorithm to learn in this type of learning. If you want an algorithm to forecast fraud, for example, you'll give it examples of fraud and examples of non-fraud. The algorithm learns to differentiate between the two classes. The term "classification" refers to the process of separating divisions. Another kind of supervised learning is regression, in which an algorithm learns to forecast values on a continuous scale.

### 4.1 Linear Regression
Linear regression is a supervised learning machine learning algorithm. It carries out a regression mission. Centered on independent variables, regression models a desired prediction value. It is mostly used in predicting and determining the relationship between variables. Different regression models vary in terms of the kind of interaction that exists between the dependent and independent variables, as well as how well they work together. To sum it up, Linear regression is a technique for predicting a value (like the sale price of a house). First, attempt to match a line to the given data.The cost feature determines the quality of your rows i.e. tells you how good your line is. To find the best line, use gradient descent.

### 4.2 Logistic Regression
A percentage is returned by the logistic function. The performance is constrained to a range of 0 to 1 using the sigmoid equation. The expense is estimated on a log scale, with the higher the number of errors, the higher the penalty. A judgement boundary is a line that you draw to divide the data into two groups. Using one-vs-all division, if you have different grades. Choose neural networks or support vector machines if you want non-linear classification.

### 4.3 Non-Linear Predictions
Though linear models are always adequate, real-world problems are often much more complicated. To work with complex, non-linear data sets, we have other machine learning algorithms and techniques at our hands. Two of these techniques will be discussed in this section.
#### 4.3.1 Random Forest
Decision Tree Logic: At each node, it will ask — *What feature will allow me to split the observations at hand in a way that the resulting groups are as different from each other as possible (and the members of each resulting subgroup are as similar to each other as possible)?*
Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction.
For hands on tutorial on random forest check out the following repositories:  
1. https://github.com/cellatwork-24x7/Conference-Info  
2. https://www.youtube.com/embed/mtIePLVqVhA?showinfo=0&rel=0&controls=1&autoplay=1  
3. https://github.com/cellatwork-24x7/coin_flip_game  
#### 4.3.2 Neural Networks
A Neural Network is a cognitive model that is closely modelled on the functional cerebral cortex of a person in order to mimic the same thought and perception type. Layers of interconnected nodes make up Neural Networks, each of which has an activation mechanism that computes the network's performance.
## 5. Unsupervised Learning
Unsupervised learning, unlike supervised learning, does not include labelled results. The aim of unsupervised learning is to discover structure from scratch in a new data set. This can be incredibly helpful, for example, in identifying customer groups for a consumer product.
#### 5.1 Clustering
Clustering is a set of methods for dividing data into sets, or clusters. Clusters are loosely described as collections of data objects that are more similar to each other than to data objects from other clusters.In practise, clustering aids in the identification of two types of data:   
1. Authenticity  
2. Applicability

Selecting an appropriate clustering algorithm for your dataset is often difficult due to the number of choices available. Some important factors that affect this decision include the characteristics of the clusters, the features of the dataset, the number of outliers, and the number of data objects. There are three popular categories of clustering algorithms:  
1. Partitional clustering  
2. Hierarchical clustering  
3. Density-based clustering  


