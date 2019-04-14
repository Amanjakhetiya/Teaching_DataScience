#
# My Template Soln

import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import  metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import explained_variance_score
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.svm import SVR
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from scipy.stats import skew
from scipy.stats.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def prepData(df):
    # Have a quick look of how data looks like
    # print(df.head())

    # Have a look at the summary of each column
    # Total - count will give you missing values
    # print(df.describe())

    # Check for missing values
    # print(pd.isnull(df).any())

    # Count missing values in training data set
    # print(pd.isnull(train).sum())
    # Let's fill in the "holes" with the means on numerical attributes
    df.fillna(df.mean(), inplace=True)

    # log transform skewed numeric features:
    # https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
    # numeric_feats = df.dtypes[df.dtypes != "object"].index
    #
    # skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    # skewed_feats = skewed_feats[skewed_feats > 0.75]
    # skewed_feats = skewed_feats.index
    #
    # df[skewed_feats] = np.log1p(df[skewed_feats])
    # df = pd.get_dummies(df)
    # # Find correlation for between all columns
    # pearson = df.corr(method='pearson')
    # #print(pearson)
    #
    # # Since the target attr is the last, remove corr with itself
    # corr_with_target = pearson.ix[-1][:-1]
    # corr_with_target_dict = corr_with_target.to_dict()
    #
    # # List the attributes sorted from the most predictive by their correlation with Sale Price
    # print("FEATURE \tCORRELATION")
    # for attr in sorted(corr_with_target_dict.items(), key=lambda x: -abs(x[1])):
    #     print("{0}: \t{1}".format(*attr))
    # print(corr_with_target[abs(corr_with_target).argsort()[::1]])
    #
    # attrs = pearson.iloc[:-1, :-1]  # all except target
    # # only important correlations and not auto-correlations
    # threshold = 0.5
    # # {(YearBuilt, YearRemodAdd): 0.592855, (1stFlrSF, GrLivArea): 0.566024, ...
    # important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
    #     .unstack().dropna().to_dict()
    # #     attribute pair                   correlation
    # # 0     (OverallQual, TotalBsmtSF)     0.537808
    # # 1     (GarageArea, GarageCars)	   0.882475
    # # ...
    # unique_important_corrs = pd.DataFrame(
    #     list(set([(tuple(sorted(key)), important_corrs[key]) \
    #               for key in important_corrs])), columns=['Attribute Pair', 'Correlation'])
    # # sorted by absolute value
    # unique_important_corrs = unique_important_corrs.ix[
    #     abs(unique_important_corrs['Correlation']).argsort()[::-1]]
    #
    # print(unique_important_corrs)

    # # Generate a mask for the upper triangle
    # mask = np.zeros_like(pearson, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    #
    # # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(12, 12))
    #
    # # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    #
    # # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(pearson, mask=mask, cmap=cmap, vmax=.3,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


    # target = df['SalePrice']
    # sns.distplot(target)
    #

    # # Scatter Plot
    # x, y = df['YearBuilt'], df['SalePrice']
    # plt.scatter(x, y, alpha=0.5)
    #
    # # or via jointplot (with histograms aside):
    # sns.jointplot(x, y, kind='scatter', joint_kws={'alpha': 0.5})
    #
    # # Hexagonal 2-D plot
    # sns.jointplot(x, y, kind='hex')
    #sns.jointplot(x, y, kind='kde')
    # plt.figure(1)
    # f, axarr = plt.subplots(3, 2, figsize=(10, 9))
    # y = target.values
    # axarr[0, 0].scatter(train['OverallQual'].values, y)
    # axarr[0, 0].set_title('OverallQual')
    # axarr[0, 1].scatter(train['TotRmsAbvGrd'].values, y)
    # axarr[0, 1].set_title('TotRmsAbvGrd')
    # axarr[1, 0].scatter(train['GarageCars'].values, y)
    # axarr[1, 0].set_title('GarageCars')
    # axarr[1, 1].scatter(train['GarageArea'].values, y)
    # axarr[1, 1].set_title('GarageArea')
    # axarr[2, 0].scatter(train['TotalBsmtSF'].values, y)
    # axarr[2, 0].set_title('TotalBsmtSF')
    # axarr[2, 1].scatter(train['1stFlrSF'].values, y)
    # axarr[2, 1].set_title('1stFlrSF')
    # f.text(-0.01, 0.5, 'Sale Price', va='center', rotation='vertical', fontsize=12)
    # plt.tight_layout()
    # plt.show()

    # print(df.dtypes)
    # For  the  non-numerical  values look  at frequency distribution
    # print(df['Property_Area'].value_counts())

    # Study distributions, study outliers
    # df['ApplicantIncome'].hist(bins=50) # or by df.boxplot(column='ApplicantIncome')
    # df.boxplot(column='ApplicantIncome', by = 'Education') # Segregate by education
    # plt.show()

    # Cross Tabulation or Pivot table
    # table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)

    # temp1 = df['Credit_History'].value_counts(ascending=True)
    # temp2 = df.pivot_table(values="Loan_Status", index=['Credit_History'], aggfunc = lambda x:x.map({"Y":1,"N":0}).mean())
    #
    # temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
    # temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

    # temp4 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
    # temp4.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
    # plt.show()

    # Check missing values of all the columns
    # mt = df.apply(lambda x: sum(x.isnull()), axis=0)
    #
    # # fill missing values
    # df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    # #
    # # # Outliers treatment
    # df['LoanAmount_log'] = np.log(df['LoanAmount'])
    # # df['LoanAmount_log'].hist(bins=20)
    # # plt.show()
    #
    # df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    # df['TotalIncome_log'] = np.log(df['TotalIncome'])
    #
    # Building a Predictive Model in Python
    # sklearn requires all inputs to be numeric,
    # convert all our categorical variables into numeric by encoding the categories.
    # var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
    nonnumeric_columns = df.loc[:, df.dtypes == object]
    le = preprocessing.LabelEncoder()
    for i in nonnumeric_columns:
        df[i] = le.fit_transform(df[i])

    # NEW FEATURES
    # df['total_sq_footage'] = df['GrLivArea'] + df['TotalBsmtSF']
    # df['total_baths'] = df['BsmtFullBath'] + df['FullBath'] + (0.5 * (df['BsmtHalfBath'] + df['HalfBath']))

    # One more call for cleanup
    # df['total_sq_footage'].fillna(df['total_sq_footage'].mode(),inplace=True)
    # df['total_baths'].fillna(df['total_baths'].mode(), inplace=True)
    return df

def prepModels(models, regression = True):
    if regression == True:

        # alpha: 0.01,0.1,1,10,100
        # fit_intercept: True/False
        # normalize: True/False
        model = Ridge(alpha=0.1,fit_intercept=True,normalize=True)
        models.append(('Ridge', model))

        model = KernelRidge()
        models.append(('KernelRidge', model))

        # # alphas: 0.01,0.1,1,10,100
        # # normalize: True/False
        # model = LassoCV(alphas=[0.1])
        # models.append(model)

        # model = SVR()
        # models.append(('SVR',model))

        # fit_intercept: True/False
        # normalize: True/False
        model = LinearRegression(fit_intercept=True,normalize=True)
        models.append(('LinearRegression', model))

        model = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=120, random_state=10)
        models.append(('BaggingClassifier',model))

        model = RandomForestRegressor(n_estimators=120)
        models.append(('RandomForestRegressor',model))

        model = ExtraTreesRegressor(n_estimators=120)
        models.append(('ExtraTreesRegressor',model))

        model = AdaBoostRegressor(n_estimators=120, random_state=10)
        models.append(('AdaBoostRegressor',model))

        model = GradientBoostingRegressor(n_estimators=120, random_state=10)
        models.append(('GradientBoostingRegressor',model))

        model = xgb.XGBRegressor (gamma=0.5,max_depth=3,min_child_weight=1,subsample=0.6,colsample_bytree=0.6)
        models.append(('XGBRegressor ', model))


    else:
        # penalty='l1','l2'
        # C=0.001 to 100 in multiples of 10
        model = LogisticRegression(penalty='l1',C=0.001)
        models.append(('LogisticRegression', model))

        model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=7)
        models.append(('BaggingClassifier',model))

        # n_estimators=120,300,500,800,1200
        # max_depth=5,8,15,25,30,None
        # min_samples_split=1,2,5,10,15,100
        # min_samples_leaf=1,2,5,10
        # max_features='log2','sqrt
        model = RandomForestClassifier(n_estimators=120,max_depth=5,min_samples_split=1, min_samples_leaf=1,max_features='log2')
        models.append(('RandomForestClassifier',model))

        model = ExtraTreesClassifier(n_estimators=100)
        models.append(('ExtraTreesClassifier',model))

        model = AdaBoostClassifier(n_estimators=30, random_state=7)
        models.append(('AdaBoostClassifier',model))

        model = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,
                                                               max_depth=8,max_features='sqrt',subsample=0.8,random_state=10,
                                                                n_estimators=60)
        models.append(('GradientBoostingClassifier',model))

        model = GaussianNB()
        models.append(('GaussianNB',model))

        #class_weight='balanced'/'none'
        # gamma = 'auto'
        model = SVC(gamma='auto', class_weight='balanced')
        models.append(('SVC',model))

        model = KNeighborsClassifier()
        models.append(('KNeighborsClassifier', model))

        # model = xgb.XGBClassifier (base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
        #    learning_rate=0.1, max_delta_step=0, max_depth=2,
        #    min_child_weight=1, missing=None, n_estimators=360, nthread=-1,
        #    objective='reg:linear', reg_alpha=0, reg_lambda=1,
        #    scale_pos_weight=1, seed=0, silent=True, subsample=1)
        # gamma = 0.5,0.7,0.9
        # max_depth = 3,5,7,9
        # min_child_weight = 1,3,5,7
        # subsample = 0.6,0.7...1
        # colsample_bytree = 0.6,0.7...1,
        # alpha = 0.1,0.5,1
        model = xgb.XGBClassifier (gamma=0.5,max_depth=3,min_child_weight=1,subsample=0.6,colsample_bytree=0.6)
        models.append(('XGBClassifier ', model))

        # estimators = []
        # estimators.append(('linear', LinearRegression()))
        # estimators.append(('XGBRegressor', XGBRegressor()))
        # ensemble = VotingClassifier(estimators)
        # models.append(('VotingClassifier',ensemble))

    return models

def classification_model(model, data, predictors, outcome):
    kf = KFold(data.shape[0], n_folds=5)
    scores = []
    for train_rows, test_rows in kf:
        train_predictors = (data[predictors].iloc[train_rows, :])
        train_target = data[outcome].iloc[train_rows]

        model.fit(train_predictors, train_target)

        test_predictors = (data[predictors].iloc[test_rows, :])
        test_target = data[outcome].iloc[test_rows]

        #  Evaluation: In case of a skewed binary classification problem we generally choose area under the receiver
        # operating characteristic curve (ROC AUC or simply AUC). In case of multi-label or multi-class classification
        # problems, we generally choose categorical cross-entropy or multiclass log loss and mean squared error in case
        # of regression problems.

        # # Print accuracy
        # predictions = model.predict(test_predictors)
        # accuracy = metrics.accuracy_score(predictions, test_target)
        # print("Accuracy : %s" % "{0:.3%}".format(accuracy))

        score = model.score(test_predictors,test_target)
        scores.append(score)

    mean_score = np.mean(scores)
    print("Accuracy : {0:4.2f}".format(mean_score))
    return mean_score


train = pd.read_csv('data/cropdamageTrain.csv')
test = pd.read_csv('data/cropdamageTest.csv')
train = prepData(train)
test = prepData(test)

outcome_variable = 'Crop_Damage'
id_variable = 'ID'

# predictor_all_variables = [col for col in test.columns if col != id_variable]

# # Feature selection by RandomForrest
# model = RandomForestClassifier(n_estimators=100)
# classification_model(model,train,predictor_all_variables,outcome_variable)
# featimp = pd.Series(model.feature_importances_,
#                     index=predictor_all_variables).sort_values(ascending=False)
# print(featimp)
# least_imp_60_variables = list(featimp.index.values)
# drop_columns = [id_variable] + least_imp_60_variables[-60:]
#drop_columns = [id_variable,'MSSubClass','OverallCond','YrSold','MoSold','MiscVal','PoolArea','X3SsnPorch','BsmtHalfBath','HalfBath','GrLivArea','BsmtFullBath','FullBath','TotalBsmtSF']
#predictor_variables = [col for col in test.columns if col not in drop_columns]
# https://www.kaggle.com/tilii7/house-prices-advanced-regression-techniques/feature-importance-four-different-ways/output
#predictor_variables =['total_sq_footage','total_baths','YearBuilt','YearRemodAdd','GarageArea','GrLivArea','LotArea','BsmtFinSF1','TotalBsmtSF','OverallQual','1stFlrSF','OverallCond','2ndFlrSF']

# # Feature selection by SelectBest
# skb = SelectKBest(chi2,k=10)
# indices = skb.fit(train[predictor_all_variables], train[outcome_variable])
# print(test.columns[[skb.get_support(indices)]])

drop_columns = [id_variable,outcome_variable,'Number_Weeks_Used']
predictor_variables = [col for col in train.columns if col not in drop_columns]
# # Finding parameters for boosting
# param_grid = {
#             'max_leaf_nodes ': [1,7,15] ## not possible in our example (only 1 fx)
#               }
# est = GradientBoostingRegressor(n_estimators=120,min_samples_split=200,loss='huber',learning_rate=0.05, max_depth=4, min_samples_leaf=3,subsample=0.9,random_state=10,max_features=len(predictor_variables))
# gs_cv = GridSearchCV(est, param_grid, n_jobs=1).fit(train[predictor_variables], train[outcome_variable])
# print(gs_cv.best_params_)
# indices = est.feature_importances_
# print(test.columns[indices])

models = []
models = prepModels(models, False)
max_score = -1
best_model = None
best_model_name = None
for name, model in models:
    score = classification_model(model,train,predictor_variables,outcome_variable)
    if score > max_score:
        max_score = score
        best_model = model
        best_model_name = name

print("Best Model {0} with Score {1:4.2f}".format(best_model_name, max_score))
best_model.fit(train[predictor_variables], train[outcome_variable])
test[outcome_variable] = best_model.predict(test[predictor_variables])
test.to_csv('data/cropdamageSubmission.csv',columns=[id_variable,outcome_variable],index=False)


