import pandas as pd

# 读取数据
train_df = pd.read_csv('训练集结果.csv')
test_df = pd.read_csv('测试集结果.csv')

# 查看is_fraud列的前20行
print("=== 训练集is_fraud列前20行 ===")
print(train_df['is_fraud'].head(20))

print("\n=== 测试集is_fraud列前20行 ===")
print(test_df['is_fraud'].head(20))

# 查看is_fraud列的数据类型
print("\n=== 训练集is_fraud列数据类型 ===")
print(train_df['is_fraud'].dtype)

print("\n=== 测试集is_fraud列数据类型 ===")
print(test_df['is_fraud'].dtype)

# 查看is_fraud列的唯一值
print("\n=== 训练集is_fraud列唯一值 ===")
print(train_df['is_fraud'].unique())

print("\n=== 测试集is_fraud列唯一值 ===")
print(test_df['is_fraud'].unique())

# 检查NaN值数量
print("\n=== 训练集is_fraud列NaN值数量 ===")
print(train_df['is_fraud'].isna().sum())

print("\n=== 测试集is_fraud列NaN值数量 ===")
print(test_df['is_fraud'].isna().sum())
