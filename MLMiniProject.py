
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import sklearn.preprocessing

import numpy as np
import pandas as pd
import sys
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from pandas import DataFrame, read_csv
from sklearn import preprocessing


# In[2]:


car_data=pd.read_csv("C:/Users/Madhuri_Reddy/Desktop/Miniprojectdataset.csv")
car_data.head()


# In[3]:


car_data.shape


# In[4]:


from sklearn.preprocessing import LabelEncoder

list_of_feaatures_to_encode = ['make','fueltype','aspiration','numberofdoors','bodystyle',
                               'drivewheels','engineloc','enginetype','nocyl','fuelsys']
le = LabelEncoder()

for i in list_of_feaatures_to_encode:
    enc = le.fit(np.unique(car_data[i].values))
    print(enc.classes_)
    car_data[i] = le.fit_transform(car_data[i])


# In[5]:


for i in list_of_feaatures_to_encode:
    enc = le.fit(np.unique(car_data[i].values))
    print(enc.classes_)
    car_data[i] = le.fit_transform(car_data[i])


# In[6]:


(car_data.isnull().sum().sort_values(ascending=False))


# In[7]:


car_data.dtypes


# In[8]:


car_data[car_data['normalizedlosses']=='?'].count() # Normalized-losses has 41 '?'s


# In[9]:


# converting the ? with the mean values in the 'normalize-losses' column
nl =car_data[car_data['normalizedlosses']!='?'].count()
nmean=nl.astype(int).mean()
car_data['normalizedlosses']=car_data['normalizedlosses'].replace('?',nmean).astype(int)
car_data.head(7)


# In[10]:


n2=car_data[car_data['price']!='?'].count()
meanprice=n2.astype(int).mean()
car_data['price']=car_data['price'].replace('?',meanprice).astype(int)
car_data.head()


# In[11]:


n3=car_data[car_data['horsepower']!='?'].count()
meanhp=n3.astype(int).mean()
car_data['horsepower']=car_data['horsepower'].replace('?',meanhp).astype(int)
car_data.head()


# In[12]:


#Cleaning the peak rpm data

# Convert the non-numeric data to null and convert the datatype
car_data['peakrpm'] = pd.to_numeric(car_data['peakrpm'],errors='coerce')


# In[13]:


# cleaning the bore

# Find out the number of invalid value
car_data['boreratio'].loc[car_data['boreratio'] == '?']

# Replace the non-numeric value to null and conver the datatype
car_data['boreratio'] = pd.to_numeric(car_data['boreratio'],errors='coerce') # converting from Object
# to numeric and use of 'coerce' -> invalid parsing will be set as NaN


# In[14]:


#cleaning stoke the similar way
car_data['stroke'] = pd.to_numeric(car_data['stroke'],errors='coerce')
car_data.dtypes


# In[15]:


car_data.isnull().sum() # Now we have 3 columns with NaN's.


# In[16]:


car_data.shape


# In[17]:


pd.set_option('display.max_columns', None) # This will display all the columns.
car_data.head()


# In[18]:


#now let us check the descriptive statistics that summarize the central tendency, 
#dispersion and shape of a dataset’s distribution.
car_data.describe()


# In[19]:


cardatanm=car_data.dropna()


# In[20]:


cardatanm.isnull().sum()


# In[21]:


#Vehicle make frequency diagram

cardatanm.make.value_counts().nlargest(23).plot(kind='bar',figsize=(24,6)) # here there are 
#23 unique values under column -'make'. ]
# So we display all 23 of them. nomrally 10 will be a good number for visualization
plt.title("No. of Vehicles in terms of Make")
plt.xlabel("Make")
plt.ylabel("No. of Vehicles")


# In[22]:


# From above plot, Toyota has most number of vehicles with more than 40% than the 2nd highest Nissan

# Insurance risk ratings Histogram. 

cardatanm.symboling.hist(bins=6)
plt.title("Insurance risk ratings of vehicles")
plt.ylabel('Number of vehicles')
plt.xlabel('Risk rating');


# In[23]:


# Symboling or the insurance risk rating have the ratings between -3 and 3,
# however for our dataset it starts from -2. There are more cars in the range of 0 and 1

#Normalized losses histogram

cardatanm['normalizedlosses'].hist(bins=6,color='Green',grid=False) # grid removes the lines
plt.title("Normalized losses of vehicles")
plt.ylabel('Number of vehicles')
plt.xlabel('Normalized losses')


# In[24]:


# Normalized losses which is the average loss payment
#per insured vehicle has more number of cars in the range between 65 and 175.

#Fuel Type - Bar Graph

cardatanm['fueltype'].value_counts().plot(kind='bar',color='grey')
plt.title("Fuel type frequence diagram")
plt.ylabel('No. of Vehicles')
plt.xlabel('Fule type');


# In[25]:


# You can see there are more number of gas type vehicles than diesel.

# Fuel type pie diagram
cardatanm['aspiration'].value_counts().plot.pie(figsize=(6, 6), autopct='%.2f')
plt.title("Fuel type pie diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Fuel type');


# In[26]:


# Most preferred fuel type for the customer is standard vs turbo having more than 80% of the share.

# Horse power vs No. of Vehicles histogram :

cardatanm.horsepower[np.abs(cardatanm.horsepower-cardatanm.horsepower.mean())<=(3*cardatanm.horsepower.std())].hist(bins=5,color='red');
plt.title("Horse power histogram")
plt.ylabel('Number of vehicles')
plt.xlabel('Horse power')
# Here we are taking standard deviation of horse power as there are outliers present.


# In[27]:


#Curb weight histogram

cardatanm['curbweight'].hist(bins=6,color='blue') 
plt.title("curb weight histogram")
plt.ylabel('Number of vehicles')
plt.xlabel('Curb weight');


# In[28]:


#Curb weight of the cars are distributed between 1500 and 4000 approximately.

# Drive wheels bar chart

cardatanm['drivewheels'].value_counts().plot(kind='bar',color='purple')
plt.title("Drive wheels diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Drive wheels')


# In[29]:


#From above plot, front wheel drive has most number of cars followed by rear wheel and four wheel.
# Four wheel has very less number of cars.

#Number of doors bar chart

cardatanm['numberofdoors'].value_counts().plot(kind='bar',color='black')
plt.title("Number of doors frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Number of doors');


# In[30]:


#Findings
#We have taken some key features of the automobile dataset for this analysis and below are our findings.
# Toyota is the make of the car which has most number of vehicles with more than 40% than the 2nd highest Nissan
#Most preferred fuel type for the customer is standard vs trubo having more than 80% of the choice
#For drive wheels, front wheel drive has most number of cars followed by rear wheel and four wheel. 
   #There are very less number of cars for four wheel drive.
# Curb weight of the cars are distributed between 1500 and 4000 approximately
#Symboling or the insurance risk rating have the ratings between -3 and 3 however for our dataset it starts 
      #from -2. There are more cars in the range of 0 and 1.
#Normalized losses which is the average loss payment per insured vehicle year is has more number of cars
      #in the range between 65 and 150.


# In[31]:


# no of cylinders Bar Graph

cardatanm['nocyl'].value_counts().plot(kind='bar',color='green')
plt.title("Number of cylinders frequency diagram")
plt.ylabel('Number of Vehicles')
plt.xlabel('Number of Cylinders');


# In[32]:


#Almost 160 cars have 4 cylinders which is more than 75%.

#Body Style Bar Graph

cardatanm['bodystyle'].value_counts().plot(kind='bar',color='brown')
plt.title("Body Style frequency diagram")
plt.ylabel('Number of Vehicles')
plt.xlabel('Body Style');


# In[33]:


#Sedan and Hutchback constitues to more than 60% in the body style feature.

#Now lets see the correlation between different columns using heat map. The highly correlated columns
# will have same or similar charecteristics and one of them can be dropped or not considered (in general).

import seaborn as sns
corr=cardatanm.corr()
sns.set_context("notebook",font_scale=1.0,rc={"lines.linewidth":3})
plt.figure(figsize=(13,7))
a=sns.heatmap(corr,annot=True,fmt ='.2f')


# In[34]:


#Correlation Analysis

# Findings: There are some good inferences we can take it from the correlation heat map.
# Price is more correlated with engine size and curb weight of the car 
# Curb weight is mostly correlated with engine size, length, width and wheel based which is
      #expected as these adds up the weight of the car
#Wheel base is highly correlated with length and width of the car
#Symboling and normalized car are correlated than the other fields


# In[35]:



import seaborn as sns
corr = cardatanm.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
a = sns.heatmap(corr, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)


# In[36]:


#Bivariate Analysis

#Boxplot of Price and make
#Findings: Below are our findings on the make and price of the car
   # ○ The most expensive car is manufacture by Mercedes benz and the least expensive is Chevrolet
   # ○ The premium cars costing more than 20000 are BMW, Jaquar, Mercedes benz and Porsche
   # ○ Less expensive cars costing less than 10000 are Chevrolet, Dodge, Honda, Mitsubishi, Plymoth and Subaru
   # ○ Rest of the cars are in the midrange between 10000 and 20000 which has the highest number of cars


# In[37]:


# Create a Boxplot of Price and make

plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="make", y="price", data=cardatanm)


# In[38]:


# Scatter plot of price and engine size:

sns.lmplot('price',"engsiz",data=cardatanm);


# In[39]:


#We can see that bigger the engine size higher is the price of car.

#Scatter plot of normalized losses and symboling

sns.lmplot('normalizedlosses',"symboling",data=cardatanm)


# In[40]:


#From the above plot, we can say that if a car has got negative rating, then the normalized losses 
#are less or lesser the rating, lesser the normalized loss

# Scatter plot of engine size and city MPH

plt.scatter(cardatanm['engsiz'],cardatanm['citympg'])
plt.xlabel('Engine size')
plt.ylabel('City MPG')


# In[41]:


#From above plot, we can observe an inverse relationship i.e is more than engine size - less the MPG it gives.

#Scatter plot of City MPG, Highway MPG and Curb weight based on Make of the car.
#Note that as city and highway mpg are highly corellated, the plots will be almost the same.


# In[42]:


g = sns.lmplot('citympg',"curbweight", cardatanm, hue="make", fit_reg=False);


# In[43]:


h = sns.lmplot('highwayrpm',"curbweight", cardatanm, hue="make", fit_reg=False);


# In[44]:



# Drive wheels and City MPG bar chart

cardatanm.groupby('drivewheels')['citympg'].mean().plot(kind='bar',color='blue')
plt.title("Drive wheels City MPG")
plt.ylabel('City MPG')
plt.xlabel('Drive wheels');


# In[45]:


#Above plot gives the city mpg for different drive wheels

#Boxplot of Drive wheels and Price :
#Findings: It's very evident that the Rear wheel drive cars are most expensive and front wheel is least expensive cars. 
    #Four wheel drive cars are little higher than the front wheel drive cars.
    #There is very less number of four wheel drive cars in our dataset so this picture might not be very accurate.


# In[46]:


plt.rcParams['figure.figsize']=(10,7)
sns.boxplot(x='drivewheels',y='price',data=cardatanm);


# In[47]:


# Normalized losses based on no. of doors and body style

pd.pivot_table(cardatanm, index=['numberofdoors','bodystyle'],values='normalizedlosses').plot(kind='bar',color='green')
plt.title("Normalized losses based on body style and no. of doors")
plt.ylabel('Normalized losses')
plt.xlabel('Body style and No. of doors');


# In[48]:


#Normalized losses based on body style and no. of doors
#Findings: As we understand the normalized loss which is the average loss payment per insured vehicle is 
    #calculated with many features of the cars which includes body style and no. of doors. Normalized losses are 
    #distributed across different body style but the two door cars has more number of losses than the four door cars.


# In[49]:


#Conclusion
#Analysis of the data set provides
#How the data set are distributed
#Correlation between different fields and how they are related
#Normalized loss of the manufacturer 
#Symboling : Cars are initially assigned a risk factor symbol associated with its price
#Mileage : Mileage based on City and Highway driving for various make and attributes
#Price : Factors affecting Price of the Automobile.
#Importance of drive wheels and curb weight


# In[50]:



feature_cols = ['normalizedlosses','numberofdoors','wheelbase','nocyl','engsiz','horsepower','peakrpm']
X = cardatanm[feature_cols]
print(type(X))
print(X.shape)

Y = cardatanm['price']
print(type(Y))
print(Y.shape)


# In[51]:


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[80]:


X_test.head()


# In[73]:


reg = LinearRegression()
reg.fit(X_train,Y_train)


# In[74]:


reg.score(X_train,Y_train)


# In[75]:


print(reg.intercept_)
print(reg.coef_)


# In[76]:


list(zip(feature_cols, reg.coef_))


# In[77]:


#making predictions
Y_pred = reg.predict(X_test)
Y_pred


# In[63]:


from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))


# In[64]:


feature_cols = ['horsepower','make','wheelbase','nocyl','fueltype','aspiration','drivewheels','engineloc','enginetype','length','width']
#'wheel_base','number_of_cylinders','fuel_type','aspiration','drive_wheels','engine_location','engine_type','length','width'
# use the list to select a subset of the original DataFrame
X = car_data[feature_cols]

# select a Series from the DataFrame
Y = car_data.price

# split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
print(reg.intercept_)
print(reg.coef_)

# fit the model to the training data (learn the coefficients)
reg.fit(X_train, Y_train)

# make predictions on the testing set
Y_pred = reg.predict(X_test)

# compute the RMSE of our predictions

print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))


# In[66]:


reg.predict([[111,0,88.6,0,0,0,0,0,0,168.8,64.0]])


# In[79]:


reg.predict([[111,0,88.6,3,1,0,2,0,0,168.8,88.6]])

