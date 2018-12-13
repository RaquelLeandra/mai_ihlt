from sklearn.ensemble import RandomForestRegressor
import numpy as np


class RFR(RandomForestRegressor):

    def __init__(self, n_jobs=-1, n_estimators=100):
        super().__init__(n_jobs=n_jobs, n_estimators=n_estimators)

    def print_feature_importance(self, trn):
        importance = self.feature_importances_
        indices = np.argsort(importance)[::-1]
        try:
            feat_labels = trn.columns
            for f in range(10):
                print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))
        except:
            pass
