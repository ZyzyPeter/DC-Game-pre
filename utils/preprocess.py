import pandas as pd
import numpy as np

class Preprocessor(object):

	def feature_expand(self, train_data):
		def rate_feature(train_data, feature_num, feature_den, new_feature_name):
			'''
			生成分数特征
			:param data:训练数据（不含标签）
			:param feature_sub: 分子
			:param feature_mom: 分母
			:param new_feature_name:新特征名字 
			'''
			train_data[new_feature_name] = \
				train_data[feature_num] / (train_data[feature_den] + 1)

		# 1为平滑系数，避免分母为0
		column_count = 0
		column_name = []
		for column in train_data.iloc[:, :32].columns:
			column_name.append(column)
			column_count += 1
			if column_count == 2:
				column_count = 0
				rate_feature(
					train_data=train_data,
					feature_num=column_name[0],
					feature_den=column_name[1],
					new_feature_name=column_name[0].replace('value', 'rate'))
				column_name = []
		rate_feature(
			train_data=train_data,
			feature_den='pvp_lanch_count',
			feature_num='pvp_battle_count',
			new_feature_name='pvp_lanch_battle_rate'
		)  # 主动发起pvp对总pvp占比
		rate_feature(
			train_data=train_data,
			feature_den='pvp_win_count',
			feature_num='pvp_battle_count',
			new_feature_name='pvp_win_battle_rate'
		)  # pvp胜利对总pvp占比
		rate_feature(
			train_data=train_data,
			feature_den='pvp_win_count',
			feature_num='pvp_lanch_count',
			new_feature_name='pvp_win_lanch_rate'
		)  # 主动发起pvp对胜利pvp占比
		rate_feature(
			train_data=train_data,
			feature_den='pve_win_count',
			feature_num='pvp_battle_count',
			new_feature_name='pve_win_battle_rate'
		)  # pve胜利对总pve占比
		rate_feature(
			train_data=train_data,
			feature_den='pay_price',
			feature_num='pay_count',
			new_feature_name='pay_price_rate'
		)  # 平均充值量
		rate_feature(
			train_data=train_data,
			feature_den='pay_price',
			feature_num='avg_online_minutes',
			new_feature_name='price_online_minutes_rate'
		)  # 充值与在线时间比
		# 求资源获取均值
		train_data['resource_add_mean'] = \
			train_data.iloc[:, [0, 2, 4, 6, 8]].apply(lambda x: x.mean(), axis=1)
		# 求资源获取中位数
		train_data['resource_add_med'] = \
			train_data.iloc[:, [0, 2, 4, 6, 8]].apply(lambda x: x.median(), axis=1)
		# 求资源消耗均值
		train_data['resource_reduce_mean'] = \
			train_data.iloc[:, [1, 3, 5, 7, 9]].apply(lambda x: x.mean(), axis=1)
		# 求资源消耗均值
		train_data['resource_reduce_med'] = \
			train_data.iloc[:, [1, 3, 5, 7, 9]].apply(lambda x: x.median(), axis=1)
		# 资源增耗均值比
		rate_feature(
			train_data=train_data,
			feature_den='resource_add_mean',
			feature_num='resource_reduce_mean',
			new_feature_name='resource_mean_rate'
		)
		# 资源增耗中值比
		rate_feature(
			train_data=train_data,
			feature_den='resource_add_med',
			feature_num='resource_reduce_med',
			new_feature_name='resource_med_rate'
		)
		# 部队招募均值
		train_data['army_add_mean'] = \
			train_data.iloc[:, [10, 12, 14]].apply(lambda x: x.mean(), axis=1)
		# 部队招募中位数
		train_data['army_add_med'] = \
			train_data.iloc[:, [10, 12, 14]].apply(lambda x: x.median(), axis=1)
		# 部队消耗均值
		train_data['army_reduce_mean'] = \
			train_data.iloc[:, [11, 13, 15]].apply(lambda x: x.mean(), axis=1)
		# 部队招募中位数
		train_data['army_reduce_med'] = \
			train_data.iloc[:, [11, 13, 15]].apply(lambda x: x.median(), axis=1)
		# 部队增损均值比
		rate_feature(
			train_data=train_data,
			feature_den='army_add_mean',
			feature_num='army_reduce_mean',
			new_feature_name='army_mean_rate'
		)
		# 部队增损中值比
		rate_feature(
			train_data=train_data,
			feature_den='army_add_med',
			feature_num='army_reduce_med',
			new_feature_name='army_med_rate'
		)
		# 伤兵产生均值
		train_data['wound_add_mean'] = \
			train_data.iloc[:, [16, 18, 20]].apply(lambda x: x.mean(), axis=1)
		# 伤兵产生中位数
		train_data['wound_add_med'] = \
			train_data.iloc[:, [16, 18, 20]].apply(lambda x: x.median(), axis=1)
		# 伤兵回复均值
		train_data['wound_reduce_mean'] = \
			train_data.iloc[:, [17, 19, 21]].apply(lambda x: x.mean(), axis=1)
		# 伤兵回复中位数
		train_data['wound_reduce_med'] = \
			train_data.iloc[:, [17, 19, 21]].apply(lambda x: x.median(), axis=1)
		# 伤兵产回均值比
		rate_feature(
			train_data=train_data,
			feature_den='wound_add_mean',
			feature_num='wound_reduce_mean',
			new_feature_name='wound_mean_rate'
		)
		# 伤兵产回中值比
		rate_feature(
			train_data=train_data,
			feature_den='wound_add_med',
			feature_num='wound_reduce_med',
			new_feature_name='wound_med_rate'
		)
		# 加速产生均值
		train_data['accelerate_add_mean'] = \
			train_data.iloc[:, [22, 24, 26, 28, 30]].apply(lambda x: x.mean(), axis=1)
		# 加速产生中位数
		train_data['accelerate_add_med'] = \
			train_data.iloc[:, [22, 24, 26, 28, 30]].apply(lambda x: x.median(), axis=1)
		# 加速消耗均值
		train_data['accelerate_reduce_mean'] = \
			train_data.iloc[:, [23, 25, 27, 29, 31]].apply(lambda x: x.mean(), axis=1)
		# 加速消耗中位数
		train_data['accelerate_reduce_med'] = \
			train_data.iloc[:, [23, 25, 27, 29, 31]].apply(lambda x: x.median(), axis=1)
		# 加速增耗均值比
		rate_feature(
			train_data=train_data,
			feature_den='accelerate_add_mean',
			feature_num='accelerate_reduce_mean',
			new_feature_name='accelerate_mean_rate'
		)
		# 加速增耗中值比
		rate_feature(
			train_data=train_data,
			feature_den='accelerate_add_med',
			feature_num='accelerate_reduce_med',
			new_feature_name='accelerate_med_rate'
		)
		print('扩展数据集的特征')
		return train_data

	def time_spliter(self,data):
		'''
		将数据按照时间分类
		:param data: 
		:return: 
		'''
		data['register_time_month'] = data.register_time.str[5:7]
		data['register_time_day'] = data.register_time.str[8:10]
		data = data.drop(['register_time'], axis=1)

		# object转换float
		data[['register_time_month', 'register_time_day']] = data[['register_time_month', 'register_time_day']].apply(
			pd.to_numeric)
		# data=pd.DataFrame(data,dtype=np.float)

		# 对于注册时间，拆分开月和日之后，再合并一个数值，更好反馈时间的前后
		data['register_time_count'] = data['register_time_month'] * 31 + data['register_time_day']
		return data

	def __init__(self):
		pass

def feature_expand():
	train=pd.read_csv('../data/tap_fun_train.csv')
	test=pd.read_csv('../data/tap_fun_test.csv')
	preprocessor=Preprocessor()

	# 特征扩展
	train_expanded = preprocessor.feature_expand(train.iloc[:, 2:-1])
	test_expanded = preprocessor.feature_expand(test.iloc[:, 2:])

	train_processed = pd.concat([train.iloc[:, :2], train_expanded, train.iloc[:, -1]], axis=1)
	test_processed = pd.concat([test.iloc[:, :2], test_expanded], axis=1)
	train_processed.to_csv('../data/train_expanded.csv', index=False)
	test_processed.to_csv('../data/test_expanded.csv', index=False)

def feature_expand_path(path,expand_path):

	test = pd.read_csv(path)
	preprocessor = Preprocessor()

	# 特征扩展
	test_expanded = preprocessor.feature_expand(test.iloc[:, 2:])

	test_processed = pd.concat([test.iloc[:, :2], test_expanded], axis=1)
	test_processed.to_csv(expand_path, index=False)
