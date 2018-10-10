import pandas as pd
import numpy as np
from sklearn.externals import joblib

from utils.preprocess import  feature_expand
from utils.preprocess import Preprocessor

if __name__=="__main__":

	preprocessor=Preprocessor()

	#扩展特征
	print("开始扩展特征...")
	feature_expand()

	# 扩展标签
	train = pd.read_csv('data/train_expanded.csv')
	train = train[train['pay_price'] > 0]
	train['new_pay_label']=train['prediction_pay_price']!=train['pay_price']
	train['new_pay_label']=train['new_pay_label'].map({True:1,False:0})
	train['new_pay_price']=train['prediction_pay_price']-train['pay_price']
	train['new_pay_rate']=(train['prediction_pay_price']-train['pay_price'])/train['pay_price']

	# train['new_pay_label']=round(train['new_pay_rate'])
	train.set_value(train[train['new_pay_rate']>0].index,'new_pay_label',1)

	classify_train=train.iloc[:,1:-4]
	classify_train=preprocessor.time_spliter(classify_train)

	test=pd.read_csv('data/test_expanded.csv')

	test_for_pre=test[test['pay_price']>0]

	classify_test=test_for_pre.iloc[:,1:]
	classify_test=preprocessor.time_spliter(classify_test)

	# 训练分类模型
	print("训练分类模型...")
	cl_t = classify_train.iloc[:, :-3].drop_duplicates()

	from sklearn.model_selection import GridSearchCV
	from xgboost import XGBClassifier

	model = GridSearchCV(
		estimator=XGBClassifier(tree_method='gpu_hist', max_bin=128),
		param_grid={
			'n_estimators': [1000],
			'learning_rate': [0.1],
			'max_depth': [2],
			'subsample': [1],
			'colsample_bytree': [0.8],
			'scale_pos_weight': [2.5, ],
			'min_child_weight': [2, ]
		},
		scoring='f1',
		cv=3,
		n_jobs=1,
		verbose=1)

	model.fit(cl_t, train.loc[cl_t.index,'new_pay_label'])
	joblib.dump(model,'model_save/xgb_clf.model')
	print(model.best_score_, model.best_params_)

	# 训练回归模型
	print("训练回归模型...")
	from xgboost import XGBRegressor

	model_reg = GridSearchCV(
		estimator=XGBRegressor(tree_method='gpu_hist', max_bin=128),
		#     estimator=LinearRegression(),
		param_grid={
			'n_estimators': [80],
			'learning_rate': [0.05],
			'max_depth': [2],
			'subsample': [1],
			'colsample_bytree': [0.5],
			'reg_alpha': [13.4],
		},
		scoring='neg_mean_squared_error',
		cv=3,
		n_jobs=1,
		verbose=1)
	model_reg.fit(classify_train.iloc[:, [105, 126, 121, 132, 150]][train['new_pay_label'] == 1],
	              train.iloc[:, -1][train['new_pay_label'] == 1])
	# model_reg.fit(cl_t.iloc[:,[105,126,121,132,150]][train.loc[cl_t.index,'new_pay_label']==1],train.iloc[cl_t.index,-1][train.loc[cl_t.index,'new_pay_label']==1])
	print(np.sqrt(-model_reg.best_score_), model_reg.best_params_)
	joblib.dump(model_reg,'model_save/xgb_reg.model')

	# 准备测试集
	classify_test = test_for_pre.iloc[:, 1:]
	classify_test = preprocessor.time_spliter(classify_test)

	# 测试集分类
	print("开始预测...")
	y_label = model.predict(classify_test.iloc[:, :-3])

	# 测试集回归
	test_for_reg = test_for_pre[y_label == 1]

	y_pre = model_reg.predict(test_for_reg.iloc[:, 2:].iloc[:, [105, 126, 121, 132, 150]])
	y_pre = test_for_reg['pay_price'] * y_pre
	y_pre = pd.DataFrame({'prediction_pay_price': y_pre})
	y_pre = y_pre.set_index(test_for_reg.index)

	# 合并测试集结果
	label_one = pd.concat([test_for_reg.iloc[:, 0], y_pre], axis=1)

	test_not_for_reg = test_for_pre[y_label == 0]
	label_zero = pd.DataFrame(
		{'user_id': test_not_for_reg['user_id'], 'prediction_pay_price': test_not_for_reg['pay_price']})

	pay_zero = pd.DataFrame({'user_id': test[test['pay_price'] == 0].iloc[:, 0],
	                         'prediction_pay_price': np.zeros(len(test[test['pay_price'] == 0]))})

	result = pd.concat([label_one, label_zero, pay_zero], axis=0)
	result=result.sort_values(by='user_id')

	# 保存结果
	print("保存结果")
	result.to_csv('result.csv',index=False)