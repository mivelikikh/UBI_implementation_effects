import os
import pandas as pd


class Dataset(object):
    def __init__(self, data_path, features_transform=None):
        self.data_path = data_path
        self.features_transform = features_transform
    
    @staticmethod
    def feature_tables_intersection(feature_names, feature_tables):
        for feature_name, feature_table in zip(feature_names, feature_tables):
            feature_table.columns = [feature_table.columns[0], feature_name]

        features_intersection_table = feature_tables[0]

        for i in range(1, len(feature_tables)):
            features_intersection_table = pd.merge(features_intersection_table, feature_tables[i],
                                                   how='inner', on=[feature_tables[i].columns[0]])

        return features_intersection_table


    def get_data(self, region, feature_names, target_name):
        feature_names.append(target_name)
        feature_tables = [pd.read_csv(os.path.join(self.data_path, region, feature_name + '.csv')).dropna()
                          for feature_name in feature_names]

        features_table = self.feature_tables_intersection(feature_names, feature_tables)
        features_table.drop(columns=[features_table.columns[0]], inplace=True)

        target_table = features_table[target_name].to_frame()
        features_table = features_table.drop(columns=[target_name])

        if self.features_transform is not None:
            features_table = self.features_transform.fit_transform(features_table)
            features_table = pd.DataFrame(data=features_table,
                                          columns=self.features_transform.get_feature_names_out())
        
        return features_table, target_table
