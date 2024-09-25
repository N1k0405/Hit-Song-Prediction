import pandas as pd
import numpy as np

# for the Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# Data path containing song metadata.
data_path = './spotify_data_urls.csv'

# reads the csv file and assigns it to a pandas DataFrame 
data = pd.read_csv(data_path)

#Features of each song
X = data[['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']]

# Label is either 1 or 0 indicating whether the song is a "hit" or not
y = data[['Label']]

#Splits the data into training and test data, in this case using 80% of the data for training and 20% for testing
X_train ,X_test ,y_train , y_test = train_test_split(X, y , test_size = 0.2)

#Loading logistic regression model from sklearn
model = LogisticRegression()

#Fitting the model and calculating accuracy against the test data
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(accuracy)

# print coefficients to see which features show more correlation to the song being a "hit"
coef = model.coef_
print(pd.DataFrame(list(zip(X, coef[0])), columns=['feature', 'coef']))


#Testing the removal of certain features to see the impact on accuracy
#Removing seemingly arbitrary features seems to still have slight impacts on the overall accuracy of the model 
X2 = data[['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']]
y2 = data[['Label']]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2, test_size=0.2)


model.fit(X2_train,y2_train)
accuracy = model.score(X2_test,y2_test)
print(accuracy)



