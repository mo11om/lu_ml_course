# %%
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# %%
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
 

from xgboost import XGBRegressor

# %%
pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',1000)

# %%
train=pd.read_csv('./X_train.csv', index_col=0)
test=pd.read_csv('./X_test.csv', index_col=0)
price_res=pd.read_csv('./y_train.csv', index_col=0)
print(f"train shape: {train.shape}")
print(f"test shape: {test.shape}")
print(f"price shape: {price_res.shape}")

# %% [markdown]
# # pipeline

# %%
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


# %%

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['建築完成年月'] = pd.to_datetime(df['建築完成年月'])
        today = datetime.today()
        df['date_str'] = df['交易年'].astype(str) + '-' + df['交易月'].astype(str).str.zfill(2) + '-' + df['交易日'].astype(str).str.zfill(2)
        df['交易日期'] = pd.to_datetime(df['date_str'])
        df['建築年紀'] = df['交易日期'].dt.year - df['建築完成年月'].dt.year
        df = df.drop(['建築完成年月', 'date_str','交易日期'], axis=1)
        
        return df
class FloorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['所在樓層'] = df['移轉層次']/df['總樓層數']
       
         
        
        return df

class AddShoppingPlaceFeature(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None): 
        return self 
    def transform(self, X): 
        X = X.copy() 
        X['購物場所'] = X[['大賣場', '超市', '百貨公司']].mean(axis=1) 
        return X 
class AddSchoolFeature(BaseEstimator, TransformerMixin): 
        def fit(self, X, y=None):
            return self 
        def transform(self, X): 
            X = X.copy() 
            X['學校'] = X[['托兒所', '國中', '高中職', '大學']].mean(axis=1) 
            return X
class AddGovernmentFeature(BaseEstimator, TransformerMixin): 
        def fit(self, X, y=None):
            return self 
        def transform(self, X): 
            X = X.copy() 
            X['政府'] = X[['警察局', '消防局']].mean(axis=1) 
            return X

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        df = X.copy()
        for col, le in self.label_encoders.items():
            df[col] = le.transform(df[col])
        return df

class OneHotEncoderTransformer(BaseEstimator,  
 TransformerMixin):
    def __init__(self, handle_unknown='ignore'):
        """
        Initializes the OneHotEncoderTransformer.

        Args:
            handle_unknown (str, default='ignore'): Specifies how to handle
                unknown categories during transformation. Possible values are:
                - 'error': Raise an error for unknown categories.
                - 'ignore': Ignore unknown categories and treat them as
                  previously unseen categories.
        """
        self.encoders = {}
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """
        Fits the transformer to the data X.

        Args:
            X (pd.DataFrame): The data to fit the transformer on.
            y (None, optional): Not used in this context.

        Returns:
            OneHotEncoderTransformer: The fitted transformer.
        """
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in X.select_dtypes(include=['object']).columns:
            #if col not in self.numerical_cols:
                encoder = OneHotEncoder(handle_unknown=self.handle_unknown)
                encoder.fit(X[[col]])  # Reshape to 2D for OneHotEncoder
                self.encoders[col] = encoder
        return self

    def transform(self, X):
        """
        Transforms the data X using one-hot encoding.

        Args:
            X (pd.DataFrame): The data to transform.

        Returns:
            pd.DataFrame: The transformed data with one-hot encoded columns 
                         and original numerical columns.
        """
        encoded_df = X[self.numerical_cols].copy()
        for col, encoder in self.encoders.items():
            encoded_features = encoder.transform(X[[col]]).toarray()
            encoded_df = pd.concat([encoded_df, pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([col]))], axis=1)
        return encoded_df
class GaussianBasisFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ngrid=3):
        self.ngrid = ngrid

    def fit(self, X, y=None):
        coord_x = X['橫坐標'].values
        coord_y = X['縱坐標'].values

        xmin, xmax = min(coord_x), max(coord_x) 
        ymin, ymax = min(coord_y), max(coord_y) 
        self.xgrids = np.linspace(xmin, xmax, self.ngrid + 1)
        self.ygrids = np.linspace(ymin, ymax, self.ngrid + 1)

        self.mu_x_all, self.mu_y_all, self.std_x_all, self.std_y_all, self.count_all = [], [], [], [], []

        for i in range(self.ngrid):
            for j in range(self.ngrid):
                x1, x2 = self.xgrids[i], self.xgrids[i + 1]
                y1, y2 = self.ygrids[j], self.ygrids[j + 1]
                tmpindx = (x1 <= coord_x) * (coord_x < x2)
                tmpindy = (y1 <= coord_y) * (coord_y < y2)
                tmpind = tmpindx * tmpindy
                npoints = np.sum(tmpind)

                if npoints >= 20:
                    mu_x = np.mean(coord_x[tmpind])
                    mu_y = np.mean(coord_y[tmpind])
                    std_x = np.std(coord_x[tmpind])
                    std_y = np.std(coord_y[tmpind])

                    self.mu_x_all.append(mu_x)
                    self.mu_y_all.append(mu_y)
                    self.std_x_all.append(std_x)
                    self.std_y_all.append(std_y)
                    self.count_all.append(npoints)

        return self

    def transform(self, X):
        coord_x = X['橫坐標'].values
        coord_y = X['縱坐標'].values

        ngf = len(self.mu_x_all)
        gf_all = np.zeros((coord_x.shape[0], ngf))

        for ii in range(ngf):
            mu_x = self.mu_x_all[ii]
            mu_y = self.mu_y_all[ii]
            std_x = self.std_x_all[ii]
            std_y = self.std_y_all[ii]

            tmpgf = np.exp(-(coord_x - mu_x) ** 2 / (2 * std_x ** 2) - (coord_y - mu_y) ** 2 / (2 * std_y ** 2))
            gf_all[:, ii] = tmpgf

        gf_df = pd.DataFrame(gf_all, columns=[f"gf_{i}" for i in range(gf_all.shape[1])])
        
        return pd.concat([X, gf_df], axis=1)

# Create the pipeline
pipeline = Pipeline([
    ('date_transformer', DateTransformer()),
    ('floor_transformer',FloorTransformer()),
   #('add_shopping_place', AddShoppingPlaceFeature()), 
  # ('add_school', AddSchoolFeature()),
  #('add_GovernmentFeature', AddGovernmentFeature()),
 ('drop_unwanted_columns', DropColumns(columns=['托兒所','國小', '國中', '高中職', '大學', '大賣場', '超市', '百貨公司'])),
# ('label_encoder', LabelEncoderTransformer()),
 #('onehot encoder',OneHotEncoderTransformer()),
   ('gaussian_basis', GaussianBasisFunctionTransformer(ngrid=3)),
  


])




# %%
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


# %%

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['建築完成年月'] = pd.to_datetime(df['建築完成年月'])
        today = datetime.today()
        df['date_str'] = df['交易年'].astype(str) + '-' + df['交易月'].astype(str).str.zfill(2) + '-' + df['交易日'].astype(str).str.zfill(2)
        df['交易日期'] = pd.to_datetime(df['date_str'])
        df['建築年紀'] = df['交易日期'].dt.year - df['建築完成年月'].dt.year
        df = df.drop(['建築完成年月', 'date_str','交易日期'], axis=1)
        
        return df
class FloorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['所在樓層'] = df['移轉層次']/df['總樓層數']
       
         
        
        return df

class AddShoppingPlaceFeature(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None): 
        return self 
    def transform(self, X): 
        X = X.copy() 
        X['購物場所'] = X[['大賣場', '超市', '百貨公司']].mean(axis=1) 
        return X 
class AddSchoolFeature(BaseEstimator, TransformerMixin): 
        def fit(self, X, y=None):
            return self 
        def transform(self, X): 
            X = X.copy() 
            X['學校'] = X[['托兒所', '國中', '高中職', '大學']].mean(axis=1) 
            return X
class AddGovernmentFeature(BaseEstimator, TransformerMixin): 
        def fit(self, X, y=None):
            return self 
        def transform(self, X): 
            X = X.copy() 
            X['政府'] = X[['警察局', '消防局']].mean(axis=1) 
            return X

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        df = X.copy()
        for col, le in self.label_encoders.items():
            df[col] = le.transform(df[col])
        return df

class OneHotEncoderTransformer(BaseEstimator,  
 TransformerMixin):
    def __init__(self, handle_unknown='ignore'):
        """
        Initializes the OneHotEncoderTransformer.

        Args:
            handle_unknown (str, default='ignore'): Specifies how to handle
                unknown categories during transformation. Possible values are:
                - 'error': Raise an error for unknown categories.
                - 'ignore': Ignore unknown categories and treat them as
                  previously unseen categories.
        """
        self.encoders = {}
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """
        Fits the transformer to the data X.

        Args:
            X (pd.DataFrame): The data to fit the transformer on.
            y (None, optional): Not used in this context.

        Returns:
            OneHotEncoderTransformer: The fitted transformer.
        """
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in X.select_dtypes(include=['object']).columns:
            #if col not in self.numerical_cols:
                encoder = OneHotEncoder(handle_unknown=self.handle_unknown)
                encoder.fit(X[[col]])  # Reshape to 2D for OneHotEncoder
                self.encoders[col] = encoder
        return self

    def transform(self, X):
        """
        Transforms the data X using one-hot encoding.

        Args:
            X (pd.DataFrame): The data to transform.

        Returns:
            pd.DataFrame: The transformed data with one-hot encoded columns 
                         and original numerical columns.
        """
        encoded_df = X[self.numerical_cols].copy()
        for col, encoder in self.encoders.items():
            encoded_features = encoder.transform(X[[col]]).toarray()
            encoded_df = pd.concat([encoded_df, pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([col]))], axis=1)
        return encoded_df
class GaussianBasisFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ngrid=3):
        self.ngrid = ngrid

    def fit(self, X, y=None):
        coord_x = X['橫坐標'].values
        coord_y = X['縱坐標'].values

        xmin, xmax = min(coord_x), max(coord_x) 
        ymin, ymax = min(coord_y), max(coord_y) 
        self.xgrids = np.linspace(xmin, xmax, self.ngrid + 1)
        self.ygrids = np.linspace(ymin, ymax, self.ngrid + 1)

        self.mu_x_all, self.mu_y_all, self.std_x_all, self.std_y_all, self.count_all = [], [], [], [], []

        for i in range(self.ngrid):
            for j in range(self.ngrid):
                x1, x2 = self.xgrids[i], self.xgrids[i + 1]
                y1, y2 = self.ygrids[j], self.ygrids[j + 1]
                tmpindx = (x1 <= coord_x) * (coord_x < x2)
                tmpindy = (y1 <= coord_y) * (coord_y < y2)
                tmpind = tmpindx * tmpindy
                npoints = np.sum(tmpind)

                if npoints >= 20:
                    mu_x = np.mean(coord_x[tmpind])
                    mu_y = np.mean(coord_y[tmpind])
                    std_x = np.std(coord_x[tmpind])
                    std_y = np.std(coord_y[tmpind])

                    self.mu_x_all.append(mu_x)
                    self.mu_y_all.append(mu_y)
                    self.std_x_all.append(std_x)
                    self.std_y_all.append(std_y)
                    self.count_all.append(npoints)

        return self

    def transform(self, X):
        coord_x = X['橫坐標'].values
        coord_y = X['縱坐標'].values

        ngf = len(self.mu_x_all)
        gf_all = np.zeros((coord_x.shape[0], ngf))

        for ii in range(ngf):
            mu_x = self.mu_x_all[ii]
            mu_y = self.mu_y_all[ii]
            std_x = self.std_x_all[ii]
            std_y = self.std_y_all[ii]

            tmpgf = np.exp(-(coord_x - mu_x) ** 2 / (2 * std_x ** 2) - (coord_y - mu_y) ** 2 / (2 * std_y ** 2))
            gf_all[:, ii] = tmpgf

        gf_df = pd.DataFrame(gf_all, columns=[f"gf_{i}" for i in range(gf_all.shape[1])])
        
        return pd.concat([X, gf_df], axis=1)

# Create the pipeline
pipeline = Pipeline([
    ('date_transformer', DateTransformer()),
    ('floor_transformer',FloorTransformer()),
   #('add_shopping_place', AddShoppingPlaceFeature()), 
  # ('add_school', AddSchoolFeature()),
  #('add_GovernmentFeature', AddGovernmentFeature()),
 ('drop_unwanted_columns', DropColumns(columns=['托兒所','國小', '國中', '高中職', '大學', '大賣場', '超市', '百貨公司'])),
# ('label_encoder', LabelEncoderTransformer()),
 #('onehot encoder',OneHotEncoderTransformer()),
   ('gaussian_basis', GaussianBasisFunctionTransformer(ngrid=3)),
  


])




# %%
full=train.copy()
full_transformed=pipeline.fit_transform(full)
# Apply the pipeline to your data
y_test = test.copy()
test_transformed = pipeline.fit_transform(y_test)


# %%
# 填補類別型特徵的缺失值
categorical_cols = full_transformed.select_dtypes(include=['object']).columns

# 對所有類別型特徵進行獨熱編碼
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
full_encoded = onehot_encoder.fit_transform(full_transformed[categorical_cols])
test_encoded = onehot_encoder.transform(test_transformed[categorical_cols])
# 將編碼後的結果轉為 DataFrame 並確保列名稱為字符串
full_encoded_df = pd.DataFrame(full_encoded, columns=onehot_encoder.get_feature_names_out(categorical_cols).astype(str))
test_encoded_df = pd.DataFrame(test_encoded, columns=onehot_encoder.get_feature_names_out(categorical_cols).astype(str))
print(test_encoded_df.shape,full_encoded_df.shape)
# 刪除原始類別特徵，並添加編碼後的數據
full_transformed = full_transformed.drop(categorical_cols, axis=1).reset_index(drop=True)
test_transformed = test_transformed.drop(categorical_cols, axis=1).reset_index(drop=True)

full_transformed = pd.concat([full_transformed, full_encoded_df], axis=1)
test_transformed = pd.concat([test_transformed, test_encoded_df], axis=1)

# %%
full_transformed.shape,test_transformed.shape

# %%
# 創建交互特徵
full_transformed['面積_房交互'] = full_transformed['土地移轉總面積平方公尺'] * full_transformed['建物現況格局-房']
test_transformed['面積_房交互'] = test_transformed['土地移轉總面積平方公尺'] * test_transformed['建物現況格局-房']
full_transformed['面積_地鐵站交互'] = full_transformed['土地移轉總面積平方公尺'] * full_transformed['地鐵站']
test_transformed['面積_地鐵站交互'] = test_transformed['土地移轉總面積平方公尺'] * test_transformed['地鐵站']
full_transformed['面積_廳交互'] = full_transformed['土地移轉總面積平方公尺'] * full_transformed['建物現況格局-廳']
test_transformed['面積_廳交互'] = test_transformed['土地移轉總面積平方公尺'] * test_transformed['建物現況格局-廳']
full_transformed['廳_房交互'] = full_transformed['建物現況格局-廳'] * full_transformed['建物現況格局-房']
test_transformed['廳_房交互'] = test_transformed['建物現況格局-廳'] * test_transformed['建物現況格局-房']
full_transformed['面積_橫坐標交互'] = full_transformed['土地移轉總面積平方公尺'] * full_transformed['橫坐標']
test_transformed['面積_橫坐標交互'] = test_transformed['土地移轉總面積平方公尺'] * test_transformed['橫坐標']


# %%
full_transformed.shape,test_transformed.shape

# %% [markdown]
# # model & Evaluate

# %%
from sklearn .model_selection import KFold

# %% [markdown]
# ## score

# %%
# define cross validation strategy
def rmse_cv(model,X,y):
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfold))
    return rmse


# %%
class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])


# %% [markdown]
# # gpu support

# %%

# Create the models with GPU support
xgb = XGBRegressor(learning_rate=0.05,max_depth=8, min_child_weight=5, subsample=0.8,n_estimators=1000, tree_method='gpu_hist')




# %%
score =rmse_cv(xgb,full_transformed,price_res['單價元平方公尺'])
score.mean()

# %%
xgb.fit(full_transformed,price_res['單價元平方公尺'])

# %%
res=xgb.predict(test_transformed)


 
# 建立 DataFrame
df = pd.DataFrame({'單價元平方公尺': res})

# 新增 Id 欄位
# Create a DataFrame, prioritizing 'Id'
df = pd.DataFrame({'Id': df.index, '單價元平方公尺': res})

# Specify the output directory
output_dir = './output'  # Replace with your desired path
output_file = output_dir + '/output.csv'



# # 將 DataFrame 輸出為 CSV 檔案
df.to_csv(output_file, index=False)


