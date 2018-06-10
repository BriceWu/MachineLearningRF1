from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# 鸢尾花识别
iris = load_iris()
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[0])
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(len(df))
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
# print(df['is_train'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]
# 取0到3列
features = df.columns[:4]
# 随机森林分类器
clf = RandomForestClassifier(n_jobs=2)
# print(train['species'])
y, _ = pd.factorize(train['species'])
# print(y)
# print(_[2])
clf.fit(train[features], y)
test_predict = clf.predict(test[features])
predict = iris.target_names[test_predict]

# pd.crosstab见https://blog.csdn.net/alanguoo/article/details/52330404，相当于excel数据透视表中的count
p1 = pd.crosstab(test['species'], predict, rownames=['actual'], colnames=['preds'])
print(p1)
