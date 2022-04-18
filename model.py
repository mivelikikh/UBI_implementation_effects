import numpy as np

from sklearn.model_selection import ParameterGrid


class Regressor(object):
    def __init__(self, model, scorer):
        self.model = model
        self.scorer = scorer
        
        self.best_score = None
        self.best_parameters = None
    
    def reset(self):
        self.best_score = None
        self.best_parameters = None
    
    def fit(self, features, target):
        self.model.fit(X=features, y=target)
    
    def predict(self, features):
        return self.model.predict(X=features)
        
    def score(self, features, target):
        prediction = self.predict(features)
        return self.scorer(y_true=target, y_pred=prediction)
    
    def grid_search(self, parameters_grid, features, target):
        self.reset()
        
        for parameters in ParameterGrid(parameters_grid):
            self.model.set_params(**parameters)
            
            self.fit(features, target)
            prediction = self.predict(features)
            
            current_score = self.score(features, target)
            if self.best_score is None or current_score > self.best_score:
                self.best_score = current_score
                self.best_parameters = parameters
        
        self.model.set_params(**self.best_parameters)


class Predictor(object):
    def __init__(self, regressor):
        self.regressor = regressor
    
    def _form_dummy_table(self, features_table, payment_in_minimum_wage):
        dummy_table = features_table.copy()
        
        for feature_name in self.regressor.model.feature_names_in_:
            if feature_name not in features_table.columns:
                if feature_name == 'payment_in_minimum_wage':
                    dummy_table[feature_name] = len(dummy_table) * [payment_in_minimum_wage]
                    continue

                features = feature_name.split(' ')
                if len(features) == 1:
                    if feature_name[-2] == '^':
                        features = int(feature_name[-1]) * [feature_name[:-2]]
                    else:
                        features = [feature_name]

                dummy_table[feature_name] = np.prod([dummy_table[feature] for feature in features], axis=0)

        return dummy_table[self.regressor.model.feature_names_in_]
    
    def predict(self, features_table, payment_in_minimum_wage):
        dummy_table = self._form_dummy_table(features_table, payment_in_minimum_wage)
        
        return self.regressor.model.predict(X=dummy_table)
