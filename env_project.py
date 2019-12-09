# Code for Template of the  environment adopted from Programming Assignment 2
# Things to consider
# Muting / Unmuting certain features
# Would abstracting as much information from the 
# Environment code help?
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

    

class EnvSpec(object):
    def __init__(self,nS,nA,gamma):
        self._nS = nS
        self._nA = nA
        self._gamma = gamma

    @property
    def nS(self) -> int:
        """ # possible states """
        return self._nS

    @property
    def nA(self) -> int:
        """ # possible actions """
        return self._nA

    @property
    def gamma(self) -> float:
        """ discounting factor of the environment """
        return self._gamma

class Env(object):
    def __init__(self,env_spec):
        self._env_spec = env_spec

    @property
    def spec(self) -> EnvSpec:
        return self._env_spec

    def reset(self) -> int:
        """
        reset the environment. It should be called when you want to generate a new episode
        return:
            initial state
        """
        raise NotImplementedError()

   

    def step(self,action:int) -> (int, int, bool):
        """
        proceed one step.
        return:
            next state, reward, done (whether it reached to a terminal state)
        """
        raise NotImplementedError()


class diabetes(Env):
    def __init__(self):
        """
     
        unit_increment = np.array of size equal to size of x_columns
        value of the increment in each dimension
        """
        path =  "diabetes.csv"
        num_bins = 6.0
        x_columns  = ['Pregnancies','Glucose', 'BloodPressure',
                    'SkinThickness', 'Insulin',
                    'BMI',
                    'DiabetesPedigreeFunction', 'Age']
        y_column = 'Outcome'
        # An  action for each dimension to be incremented or decremented by a unit amount
        self.num_independent_variables = len(x_columns)
        self.state_dimensions = len(x_columns)
        self.num_actions =  2 *  (self.num_independent_variables)
        
        self.gamma = 1.0
        self.x_columns = x_columns
        self.y_column = y_column
        df = pd.read_csv(path)
        self.data =  df.loc[:, self.x_columns].to_numpy()
        self.path = path
        self.num_bins = num_bins
        print(self.data.shape)
        self.state_low = np.amin(self.data, axis = 0)
        self.state_high = np.amax(self.data, axis = 0)
        # print(self.state_low, "\n", self.state_high)
        self.tile_width =  (self.state_high - self.state_low)/self.num_bins
        # print(self.tile_width)
        self.unit_step = self.tile_width
        
        # X = StateActionFeatureVectorWithTile(
        #                 self.state_low ,
        #                 self.state_high,
        #                 self.num_actions,
        #                 num_tilings=1,
        #                 tile_width=self.tile_width)
        # print(X(self.data[1], False, 0)) 

        # env_spec = EnvSpec(X.state_vector_len(), self.num_actions, self.gamma)
        # super().__init__(env_spec) 
        self.model = self.train_model(path, x_columns, y_column)
        print("Optimal Threshold for classifier is ", self.optimal_threshold)
        # self.initial_state = self.reset()
        # self.initial_prediction =  self.predict_model(self.initial_state.T)
        # print(self.initial_state, self.initial_prediction)
           


        # data['pred'] = model.predict(xgb_full)
        
    def predict_model(self,point):
        # print("  In predict_model ")
        a = int(self.model.predict(xgboost.DMatrix(point, feature_names= self.x_columns))[0] > self.optimal_threshold)
        # print("predictions are ", a > self.optimal_threshold)
        return a
        # return  int(self.model.predict(xgboost.DMatrix(point, feature_names= self.x_columns)[0] > self.optimal_threshold))
      
    def reset(self):
    # Random initialze location for each episode run
        point = np.array([np.random.uniform(low, high, size =(1,)) for  low,high in zip(self.state_low, self.state_high)])
        # print(self.state.shape)
        prediction = self.predict_model(point.T)
        if prediction == 0:
            self.state = point
            self.initial_prediction = prediction

        else:
            # If we can't find a initial point with 0 
            self.reset()

        # print("Initial prediction is" , self.initial_prediction)
        return self.state

      



    def step(self, action):
        assert action in range(self.num_actions), "Invalid Action"
        # Number of actions is number of independent features * 2
        # Each independent features has two actions for itself.
        # (feature_id * 2 + 0) -> action to decrease value of feature a by unit_step[feature_id]
        # (feature_id * 2 + 1) -> action to increase value of a feature by unit_step[feature_id]

        # assert self.state not in self.terminal_state, "Episode has ended!"
        feature_id, action_feature = divmod(action, 2)
        next_state = np.copy(self.state)
        r_wall =  -10
        r_goal = 100
        r_step = -1
        if action_feature==0:
            #decrease feature by unit_step[feature_id]
            
            next_state[feature_id] = self.state[feature_id] - self.unit_step[feature_id]
            if next_state[feature_id] < self.state_low[feature_id]:
                # print("Wall hit low for feature ", feature_id, " for action ", action )
                return self.state, r_wall, False
        else: 
            # increase value by unit_step[feature_id]
            next_state[feature_id] = self.state[feature_id] +  self.unit_step[feature_id]
            if next_state[feature_id] > self.state_high[feature_id]:
                # print("Wall hit High for feature ", feature_id, " for action ", action )
                return self.state, r_wall, False
        self.state = np.copy(next_state)

        new_prediction = self.predict_model(self.state.T)
        # print("action ", action, "new_prediction ", new_prediction, self.state )
        if new_prediction != self.initial_prediction:
            # If we have found a point with prediction != initial prediction
            return self.state, r_goal, True
        else:
            # If we havent found a point with different than initial prediction
            return self.state, r_step, False


    def train_model(self,path , x_columns, y_column):

        data = pd.read_csv(path)
        
        X =  data.loc[:, x_columns]
        y =  data[y_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =  0.1, random_state =  4)

        xgb_full = xgboost.DMatrix(X, label=y)
        xgb_train =  xgboost.DMatrix(X_train, label = y_train)
        xgb_test = xgboost.DMatrix(X_test, label = y_test)

        params = {
        "eta": 0.002,
        "max_depth": 2,
        "objective": 'binary:logistic',
        "eval_metric":"auc",
        "subsample": 0.5,
        "silent":1
        }
        model = xgboost.train(params, xgb_train, 10000, evals = [(xgb_test, "test")], verbose_eval= 1000)
        xgboost.cv(params,xgb_full, nfold = 3, metrics="auc" , num_boost_round=10)
        data['pred'] = model.predict(xgb_full)
        fpr, tpr , thresholds = roc_curve(data['Outcome'], data['pred'])
        self.optimal_threshold = thresholds[np.argmax(tpr - fpr)]
        return model

    def get_num_actions(self):
        return len(self.num_actions)
    
    def get_prediction(self, x):
        return self.model.predict(x)


class toy(Env):
    def __init__(self, path):
        """
     
        unit_increment = np.array of size equal to size of x_columns
        value of the increment in each dimension
        """
        num_bins = 10.0
        y_column = 'Outcome'
        df = pd.read_csv(path)
        x_columns = list(df.columns)
        x_columns.remove('Outcome')
        print(x_columns)
        self.state_dimensions = len(x_columns)
        # An  action for each dimension to be incremented or decremented by a unit amount
        self.num_independent_variables = len(x_columns)
        self.num_actions =  2 *  (self.num_independent_variables)
        
        self.gamma = 1.0
        self.x_columns = x_columns
        self.y_column = y_column
        
        self.data =  df.loc[:, self.x_columns].to_numpy()
        self.data_0 = df[df['Outcome']== 0].loc[:, self.x_columns].to_numpy()
        self.data_1 = df[df['Outcome']== 1].loc[:, self.x_columns].to_numpy()
        print(self.data_0.shape, self.data_1.shape)
        self.path = path
        self.num_bins = num_bins
        # print(self.data.shape)
        self.state_low =  np.array([0.0,0.0])
        self.state_high =  np.array([1.0,1.0])
        # print(self.state_low, "\n", self.state_high)
        self.tile_width =  (self.state_high - self.state_low)/self.num_bins
        # print(self.tile_width)
        self.unit_step = self.tile_width
        

        # env_spec = EnvSpec(X.state_vector_len(), self.num_actions, self.gamma)
        # super().__init__(env_spec) 
        self.model_type = 'xgboost'
        self.model = self.train_model(path, x_columns, y_column, self.model_type)
        # print("Optimal Threshold for classifier is ", self.optimal_threshold)
        # self.initial_state = self.reset()
        # self.initial_prediction =  self.predict_model(self.initial_state.T)
        # print(self.initial_state, self.initial_prediction)

        
    def predict_model(self,point):
        # print("  In predict_model ")
        if self.model_type == 'xgboost':
            a = int(self.model.predict(xgboost.DMatrix(point, feature_names= self.x_columns))[0] > self.optimal_threshold)
        else:
            # print("In LR predcition model ", self.model)
            a = self.model.predict(point)
        # print("predictions are ", a > self.optimal_threshold)
        return a
        # return  int(self.model.predict(xgboost.DMatrix(point, feature_names= self.x_columns)[0] > self.optimal_threshold))
      
    def reset(self, s = None):
    # Random initialze point from input data set for each episode run
        
        # point = np.array([np.random.uniform(low, high, size =(1,)) for  low,high in zip(self.state_low, self.state_high)])
        # print(point)
        if s == None:
            idx = np.random.choice(self.data_0.shape[0])

            point = self.data_0[idx,:].reshape(( self.num_independent_variables, 1))
            # print(point)
            

            # print("Initial prediction is" , self.initial_prediction)
            
        if s is not None:
            point = np.array(s).reshape(( self.num_independent_variables, 1))
        
        if self.model_type == 'logistic':
            prediction = self.predict_model(point.T)[0]
        else:
            prediction = self.predict_model(point.T)
        # print(" Prediction " , prediction)
        if prediction == 0:
            self.state = point
            self.initial_prediction = prediction
        else:
            "If we can't find a initial point with 0 "
            self.reset()
        return self.state


    def step(self, action):
        assert action in range(self.num_actions), "Invalid Action"
        # Number of actions is number of independent features * 2
        # Each independent features has two actions for itself.
        # (feature_id * 2 + 0) -> action to decrease value of feature a by unit_step[feature_id]
        # (feature_id * 2 + 1) -> action to increase value of a feature by unit_step[feature_id]

        # assert self.state not in self.terminal_state, "Episode has ended!"
        feature_id, action_feature = divmod(action, 2)
        next_state = np.copy(self.state)
        r_wall =  -10
        r_goal = 100
        r_step = -1
        if action_feature==0:
            #decrease feature by unit_step[feature_id]
            
            next_state[feature_id] = self.state[feature_id] - self.unit_step[feature_id]
            if next_state[feature_id] < self.state_low[feature_id]:
                # print("Wall hit low for feature ", feature_id, " for action ", action )
                return self.state, r_wall, False
        else: 
            # increase value by unit_step[feature_id]
            next_state[feature_id] = self.state[feature_id] +  self.unit_step[feature_id]
            if next_state[feature_id] > self.state_high[feature_id]:
                # print("Wall hit High for feature ", feature_id, " for action ", action )
                return self.state, r_wall, False
        self.state = np.copy(next_state)

        new_prediction = self.predict_model(self.state.T)
        # print("action ", action, "new_prediction ", new_prediction, self.state )
        if new_prediction != self.initial_prediction:
            # If we have found a point with prediction != initial prediction
            return self.state, r_goal, True
        else:
            # If we havent found a point with different than initial prediction
            return self.state, r_step, False


    def train_model(self,path , x_columns, y_column, type = 'xgboost'):

        data = pd.read_csv(path)
        
        X =  data.loc[:, x_columns]
        y =  data[y_column]

        if type == 'xgboost':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =  0.1, random_state =  4)

            xgb_full = xgboost.DMatrix(X, label=y)
            xgb_train =  xgboost.DMatrix(X_train, label = y_train)
            xgb_test = xgboost.DMatrix(X_test, label = y_test)

            params = {
            "eta": 0.002,
            "max_depth": 2,
            "objective": 'binary:logistic',
            "eval_metric":"auc",
            "subsample": 0.5,
            "silent":1
            }
            model = xgboost.train(params, xgb_train, 2000, evals = [(xgb_test, "test")], verbose_eval= 1000)
            xgboost.cv(params,xgb_full, nfold = 3, metrics="auc" , num_boost_round=10)
            data['pred'] = model.predict(xgb_full)
            fpr, tpr , thresholds = roc_curve(data['Outcome'], data['pred'])
            self.optimal_threshold = thresholds[np.argmax(tpr - fpr)]
            self.pred = data['pred'].values
            return model
        else:
            y = y.values
            data = X.values
            model = LogisticRegression()
            model.fit(data, y)
            
            self.pred = model.predict(data)
            # self.model = model
            print("Accuracy of logistic regression model ", accuracy_score(y, self.pred))
            print(" Parameters Coeff, intercept", model.coef_ , model.intercept_)
            return model
            # print(self.model)
            
    def get_num_actions(self):
        return len(self.num_actions)
    
    def get_prediction(self, x):
        return self.model.predict(x)

