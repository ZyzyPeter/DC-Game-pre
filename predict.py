import pandas as pd
import numpy as np
import sys
from sklearn.externals import joblib

from utils.preprocess import feature_expand_path
from utils.preprocess import Preprocessor

if __name__ == "__main__":

	preprocessor = Preprocessor()

	# 扩展特征
	if len(sys.argv) == 3:
		feature_expand_path(sys.argv[1], sys.argv[2])
	else:
		print('请指定需要预测的文件的路径以及特征扩展文件的路径')

	test = pd.read_csv(sys.argv[2])

	test_for_pre = test[test['pay_price'] > 0]

	classify_test = test_for_pre.iloc[:, 1:]
	classify_test = preprocessor.time_spliter(classify_test)

	model = joblib.load('model_save/xgb_clf.model')

	model_reg = joblib.load('model_save/xgb_reg.model')

	# 准备测试集
	classify_test = test_for_pre.iloc[:, 1:]
	classify_test = preprocessor.time_spliter(classify_test)

	# 测试集分类
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
	result = result.sort_values(by='user_id')

	# 保存结果
	result.to_csv('result.csv', index=False)
