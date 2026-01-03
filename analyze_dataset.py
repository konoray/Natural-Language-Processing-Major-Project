import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练集和测试集
train_df = pd.read_csv('训练集结果.csv')
test_df = pd.read_csv('测试集结果.csv')

# 基本信息
print("=== 训练集基本信息 ===")
print(f"总行数: {len(train_df)}")
print(f"欺诈样本数: {train_df['is_fraud'].sum()}")
print(f"欺诈样本比例: {train_df['is_fraud'].mean():.2%}")

print("\n=== 测试集基本信息 ===")
print(f"总行数: {len(test_df)}")
print(f"欺诈样本数: {test_df['is_fraud'].sum()}")
print(f"欺诈样本比例: {test_df['is_fraud'].mean():.2%}")

# 交互策略分布
print("\n=== 训练集交互策略分布 ===")
print(train_df['interaction_strategy'].value_counts())

print("\n=== 测试集交互策略分布 ===")
print(test_df['interaction_strategy'].value_counts())

# 呼叫类型分布
print("\n=== 训练集呼叫类型分布 ===")
print(train_df['call_type'].value_counts())

print("\n=== 测试集呼叫类型分布 ===")
print(test_df['call_type'].value_counts())

# 欺诈类型分布
print("\n=== 训练集欺诈类型分布 ===")
print(train_df['fraud_type'].value_counts())

print("\n=== 测试集欺诈类型分布 ===")
print(test_df['fraud_type'].value_counts())

# 对话长度分析
train_df['dialogue_length'] = train_df['specific_dialogue_content'].apply(lambda x: len(x) if isinstance(x, str) else 0)
test_df['dialogue_length'] = test_df['specific_dialogue_content'].apply(lambda x: len(x) if isinstance(x, str) else 0)

print("\n=== 训练集对话长度统计 ===")
print(train_df['dialogue_length'].describe())

print("\n=== 测试集对话长度统计 ===")
print(test_df['dialogue_length'].describe())

# 可视化
def plot_distribution(df, column, title, filename):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# 绘制欺诈分布
plot_distribution(train_df, 'is_fraud', '训练集欺诈样本分布', 'train_fraud_dist.png')
plot_distribution(test_df, 'is_fraud', '测试集欺诈样本分布', 'test_fraud_dist.png')

# 绘制交互策略分布
plot_distribution(train_df, 'interaction_strategy', '训练集交互策略分布', 'train_strategy_dist.png')
plot_distribution(test_df, 'interaction_strategy', '测试集交互策略分布', 'test_strategy_dist.png')

# 绘制呼叫类型分布
plot_distribution(train_df, 'call_type', '训练集呼叫类型分布', 'train_calltype_dist.png')
plot_distribution(test_df, 'call_type', '测试集呼叫类型分布', 'test_calltype_dist.png')

# 绘制欺诈类型分布
plot_distribution(train_df, 'fraud_type', '训练集欺诈类型分布', 'train_fraudtype_dist.png')
plot_distribution(test_df, 'fraud_type', '测试集欺诈类型分布', 'test_fraudtype_dist.png')

print("\n数据集分析完成，可视化图表已保存。")
