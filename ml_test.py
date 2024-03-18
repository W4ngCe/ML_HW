import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

# 加载数据集
df = pd.read_csv('Machine Learning-002-dataset.csv')

# Assuming the dataset includes columns: 'Years', 'Number_of_Doors', 'Color', 'Price'

# Preprocessing
# Split the data into features and target
X = df[['Years', 'Number_of_Doors', 'Color']]
y = df['Price']

# Split the dataset into training and test sets (adjust the test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
random_forest_model = RandomForestRegressor(
    n_estimators=300,  # 增加树的数量
    max_depth=35,       # 设置最大深度
    min_samples_split=3,  # 设置内部节点再划分所需最小样本数
    min_samples_leaf=1,    # 设置叶节点所需最小样本数
    max_features='sqrt',   # 最大特征数设置为特征总数的平方根
    random_state=42        # 确保结果的可重现性
)
random_forest_model.fit(X_train, y_train)

# Evaluate the model (Optional)
predictions = random_forest_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# 假设 y_test 是真实的价格值，predictions 是模型预测的价格值
percentage_errors = np.abs((y_test - predictions) / y_test) * 100

# 计算平均百分比误差
mean_percentage_error = np.mean(percentage_errors)

# 计算误差率，这里我们定义为超过一个阈值（例如10%）的预测的比例
error_rate = np.mean(percentage_errors > 10)  # 可以根据需求调整阈值

print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
print(f"Error Rate (predictions with >10% error): {error_rate:.2f}")



# 加载新的预测数据集
new_data = pd.read_csv('C:/Users/fengz/Desktop/Q2.csv')

# 确保列名正确，这里不应该包含 'Price'
expected_features = ['Years', 'Number_of_Doors', 'Color']

# 检查列名是否匹配
if not all(column in new_data.columns for column in expected_features):
    raise ValueError(f"The new data must only contain the following columns: {expected_features}")


# 使用训练好的模型进行预测
# 假设 random_forest_model 是已经训练好的模型
try:
    predicted_prices = random_forest_model.predict(new_data)
except Exception as e:
    print(f"Error during prediction: {e}")

# 输出预测的价格
print(predicted_prices)


print("Black =", 0, "Blue =", 1, "Red =", 2, "White =", 3)
