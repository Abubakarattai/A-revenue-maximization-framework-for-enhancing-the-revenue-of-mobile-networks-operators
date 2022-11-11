import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import np_utils

#hyper parameters
LEARNING_RATE = 0.0009

dataset1 = pd.read_csv('C:\\Users\\2309848A\\Dropbox\\Figs\\VFA\\RevMax_12Scs_ES_hyb.csv', header = None)
dataset2 = pd.read_csv('C:\\Users\\2309848A\\Dropbox\\Figs\\VFA\\RevMax_12Scs_ES.csv', header = None)


# creating input features and target variables
#training dataset
X_train = dataset1.iloc[0:50400,0:25]
Y_train = dataset1.iloc[0:50400,37]
y_train = np_utils.to_categorical(Y_train)

#test dataset
X_test = dataset2.iloc[0:1008,0:25]
Y_test = dataset2.iloc[0:1008,37]
y_test = np_utils.to_categorical(Y_test)

#%%
#creating object class
class solver:
    def __init__(self, features_space, y_train):
        
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(features_space,), activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(4096, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=1000, batch_size=50, verbose=1)
    
    def pred(self, X_test):
        y_pred = self.model.predict_classes(X_test)
        return y_pred
    
    def eva(self, X_test, y_test):
       score = self.model.evaluate(X_test, y_test, verbose=1)
       return score

learn = solver(25, y_train)
predicted_y1 = learn.pred(X_test)
y_pred1 = pd.DataFrame(predicted_y1)
score1 = learn.eva(X_test, y_test)
print(score1)

# create containing to save all predicted values
#%%
X_new = pd.DataFrame(X_test)
X_new = X_new.reset_index(drop=True)
y_pred_total = pd.concat([X_new, y_pred1], axis=1)
#switching_data = pd.concat([X_test, y_pred_total], axis = 1)

#y_pred_total.to_csv('C:\\Users\\2309848A\\Dropbox\\Figs\\VFA\\Rev_max_ANN.csv', header = None, index = None)
y_pred_total.to_csv('C:\\Users\\2309848A\\Dropbox\\Figs\\VFA\\Rev_max12sc_ANN_new.csv', header = None, index = None)
















