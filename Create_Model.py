import pandas as pd
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential

# Load training data set from CSV file
training_data_df = pd.read_csv("sales_data_training.csv")

# Load testing data set from CSV file
test_data_df = pd.read_csv("sales_data_test.csv")

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)

# Print out the adjustment that the scaler applied to the total_earnings column of data
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

# Create new pandas DataFrame objects from the scaled data
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)

scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)

#read the scaled training dataset
training_data_df1=pd.read_csv('sales_data_training_scaled.csv')

X = training_data_df1.drop('total_earnings', axis=1).values
Y = training_data_df1[['total_earnings']].values

#model creation
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1,activation='linear'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error',optimizer="adam")

#train the model
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

#Testing the model on the test dataset
#read the test dataset
test_data_df1 = pd.read_csv('sales_data_testing_scaled.csv')

X_test = test_data_df1.drop('total_earnings',axis=1).values
Y_test =  test_data_df1[['total_earnings']].values

#printing the MSE for the test dataset
test_error_rate = model.evaluate(X_test,Y_test,verbose=0)
print('The MSE for the test data set is : {}'.format(test_error_rate))

#Now Use this model to predict the new data
#Load the new data set to make predictions
prediction_data = pd.read_csv('proposed_new_product.csv').values

#Make a prediction for the neural network
prediction = model.predict(prediction_data)
prediction = prediction[0][0]

#As the values of the dataset were scaled between 0 1 for the neural network. The predicted output should be scaled back to the original value
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

#print out the predicted result
print(' The toat earnings predicted for the game is : ${}'.format(prediction))
