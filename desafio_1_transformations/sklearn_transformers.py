import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, LabelBinarizer, OneHotEncoder



class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        for col in self.columns:
            if col in data.columns:
                data=data.drop(col, axis='columns')
        return data


class FeatureColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        return data[self.columns]


class DropNa(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        for datacol in self.columns:
            data = data.dropna(axis='index', how='any', subset=[datacol])
        return data
    

class ImputerZero(BaseEstimator, TransformerMixin):
    def __init__(self, columns,createColumns=None):
        self.columns = columns
        self.createColumns = createColumns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        if(self.createColumns == "global"):
            data["imputedZero"]=0
        for datacol in self.columns:
            if(self.createColumns == "each"):
                data["imputedZero"+datacol]=0
                data.loc[data[datacol].isnull(),"imputedZero"+datacol]=1
            if(self.createColumns == "global"):
                data.loc[data[datacol].isnull(),"imputedZero"]=1
            data.loc[data[datacol].isnull(),datacol] = 0
        return data


    
class ImputerMedian(BaseEstimator, TransformerMixin):
    def __init__(self, columns,createColumns=None):
        self.columns = columns
        self.createColumns = createColumns
        self.imputers = {}
        for datacol in self.columns:
            self.imputers[datacol]=SimpleImputer(missing_values=np.nan,strategy='median',verbose=0,copy=True)

    def fit(self, X, y=None):
        for datacol in self.columns:
            self.imputers[datacol].fit(X.drop(X.drop(datacol ,axis=1).columns ,axis=1))
        return self
    
    def transform(self, X):
        data = X.copy()
        if(self.createColumns == "global"):
            data["imputedMedian"]=0
        for datacol in self.columns:
            if(self.createColumns == "each"):
                data["imputedMedian"+datacol]=0
                data.loc[data[datacol].isnull(),"imputedMedian"+datacol]=1
            if(self.createColumns == "global"):
                data.loc[data[datacol].isnull(),"imputedMedian"]=1
            data[datacol]=self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1))
        return data

class ImputerMean(BaseEstimator, TransformerMixin):
    def __init__(self, columns,createColumns=None):
        self.columns = columns
        self.createColumns = createColumns
        self.imputers = {}
        for datacol in self.columns:
            self.imputers[datacol]=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0,copy=True)

    def fit(self, X, y=None):
        for datacol in self.columns:
            self.imputers[datacol].fit(X.drop(X.drop(datacol ,axis=1).columns ,axis=1))
        return self
    
    def transform(self, X):
        data = X.copy()
        if(self.createColumns == "global"):
            data["imputedMean"]=0
        for datacol in self.columns:
            if(self.createColumns == "each"):
                data["imputedMean"+datacol]=0
                data.loc[data[datacol].isnull(),"imputedMean"+datacol]=1
            if(self.createColumns == "global"):
                data.loc[data[datacol].isnull(),"imputedMean"]=1
            data[datacol]=self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1))
        return data
    
class ImputerFrequent(BaseEstimator, TransformerMixin):
    def __init__(self, columns,createColumns=None):
        self.columns = columns
        self.createColumns = createColumns
        self.imputers = {}
        for datacol in self.columns:
            self.imputers[datacol]=SimpleImputer(missing_values=np.nan,strategy='most_frequent',verbose=0,copy=True)

    def fit(self, X, y=None):
        for datacol in self.columns:
            self.imputers[datacol].fit(X.drop(X.drop(datacol ,axis=1).columns ,axis=1))
        return self
    
    def transform(self, X):
        data = X.copy()
        if(self.createColumns == "global"):
            data["imputedFrequent"]=0
        for datacol in self.columns:
            if(self.createColumns == "each"):
                data["imputedFrequent"+datacol]=0
                data.loc[data[datacol].isnull(),"imputedFrequent"+datacol]=1
            if(self.createColumns == "global"):
                data.loc[data[datacol].isnull(),"imputedFrequent"]=1
            data[datacol]=self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1))
        return data


class ImputerBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        for datacol in self.columns:
            self.imputers[datacol]=LabelBinarizer()

    def fit(self, X, y=None):
        for datacol in self.columns:
            self.imputers[datacol].fit(X.drop(X.drop(datacol ,axis=1).columns ,axis=1))
        return self
    
    def transform(self, X):
        data = X.copy()
        for datacol in self.columns:
            if(len(self.imputers[datacol].classes_)==2):
                imputed=pd.DataFrame(self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1)),columns=[datacol+"Dummies"],index=data.index)
            else:
                classes = [datacol + "_" + str(x) for x in self.imputers[datacol].classes_]
                imputed=pd.DataFrame(self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1)),columns=classes,index=data.index)
            data.drop(datacol,axis=1,inplace=True)
            data=pd.merge(
                data, imputed, how='inner',
                on=None, left_index=True, right_index=True, sort=False,
                suffixes=('_x', '_y'), copy=True, indicator=False,
                validate=None
            )
        return data
    
    
    
class ScalerStandard(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputers=None
        self.columns=None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        if(self.imputers==None):
            self.columns=[col for col in data.columns if len(set(data[col]))>2]
            self.imputers={}
            for col in self.columns:
                self.imputers[col] = StandardScaler()
                self.imputers[col].fit(np.array(data[col]).reshape((-1,1)))
        
        for col in self.columns:
            data[col]=self.imputers[col].transform(np.array(data[col]).reshape((-1,1)))
        return data

class PreprocessCustom(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        
        data["CHECKING_BALANCE_UNK"] = data["CHECKING_BALANCE"] == "NO_CHECKING"
        data.loc[data["CHECKING_BALANCE"] == "NO_CHECKING","CHECKING_BALANCE"] = np.nan
        data.loc[data["CHECKING_BALANCE"].isna(),"CHECKING_BALANCE"] = 0
        data["CHECKING_BALANCE"]=data["CHECKING_BALANCE"].astype(float).astype(int)
        
        data["EXISTING_SAVINGS_UNK"] = data["EXISTING_SAVINGS"] == "UNKNOWN"
        data.loc[data["EXISTING_SAVINGS"] == "UNKNOWN","EXISTING_SAVINGS"] = np.nan
        data.loc[data["EXISTING_SAVINGS"].isna(),"EXISTING_SAVINGS"] = 0
        data["EXISTING_SAVINGS"]= data["EXISTING_SAVINGS"].astype(float).astype(int)
        
        data["AGE"] = pd.Series(data.AGE/10).astype(int)
        
        data["PAYMENT_TERM"] = pd.Series(data.PAYMENT_TERM/31).astype(int)
        

        #data
        return data


class ImputerPolinomical(BaseEstimator, TransformerMixin):
    def __init__(self, subset,grade):
        self.grade = grade
        self.subset = subset
        self.columnNames = {}
        self.polinomical = PolynomialFeatures(self.grade)
        
    def fit(self, X, y=None):
        self.polinomical.fit(X[self.subset])
        self.columnNames = self.polinomical.get_feature_names(self.subset)
        return self
    
    def transform(self, X):
        data = X[self.subset].copy()
        data = self.polinomical.transform(data)
        return pd.concat([X[ [col for col in X.columns if col not in self.subset] ],pd.DataFrame(data,columns=self.columnNames,index=X.index)],axis=1)
    
    

class ImputerOneHot(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        self.classes = {}
        for datacol in self.columns:
            self.imputers[datacol]=OneHotEncoder(drop="first",handle_unknown='ignore',)
#            a=OneHotEncoder(drop="first",handle_unknown='ignore')
#            a.

    def transform(self, X):
        data = X.copy()
        for datacol in self.columns:
            
            print(self.classes[datacol])
            print(self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1)))
            imputed = pd.DataFrame(self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1)),columns=self.classes[datacol],index=data.index)
            
            data.drop(datacol,axis=1,inplace=True)
            
            data=pd.concat([data, imputed],axis=1)
        return data
    
    def fit(self, X, y=None, **fitparams):
        for datacol in self.columns:
            self.imputers[datacol].fit(X.drop(X.drop(datacol ,axis=1).columns ,axis=1))
            self.classes[datacol] = self.imputers[datacol].get_feature_names_out()         
            
        return self
    
class GetDummiesAuto(TransformerMixin):
    """Fast one-hot-encoder that makes use of pandas.get_dummies() safely
    on train/test splits.
    """
    def __init__(self, dtypes=None):
        self.input_columns = None
        self.final_columns = None
        if dtypes is None:
            dtypes = [object, 'category']
        self.dtypes = dtypes

    def fit(self, X, y=None, **kwargs):
        self.input_columns = list(X.select_dtypes(self.dtypes).columns)
        X = pd.get_dummies(X, columns=self.input_columns)
        self.final_columns = X.columns
        return self
        
    def transform(self, X, y=None, **kwargs):
        X = pd.get_dummies(X, columns=self.input_columns)
        X_columns = X.columns
        # if columns in X had values not in the data set used during
        # fit add them and set to 0
        missing = set(self.final_columns) - set(X_columns)
        for c in missing:
            X[c] = 0
        # remove any new columns that may have resulted from values in
        # X that were not in the data set when fit
        return X[self.final_columns]
    
    def get_feature_names(self):
        return tuple(self.final_columns)

class GetDummies(TransformerMixin):
    def __init__(self, dummyColumns=None, drop_first = False):
        self.input_columns = list(dummyColumns)
        self.final_columns = None
        self.drop_first = drop_first
        
        
    def fit(self, X, y=None, **kwargs):
        X = pd.get_dummies(X, columns=self.input_columns,drop_first = self.drop_first)
        self.final_columns = X.columns
        return self
        
    def transform(self, X, y=None, **kwargs):
        X = pd.get_dummies(X, columns=self.input_columns)
        X_columns = X.columns
        # if columns in X had values not in the data set used during
        # fit add them and set to 0
        missing = set(self.final_columns) - set(X_columns)
        for c in missing:
            X[c] = 0
        # remove any new columns that may have resulted from values in
        # X that were not in the data set when fit
        return X[self.final_columns]
    
    def get_feature_names(self):
        return tuple(self.final_columns)
