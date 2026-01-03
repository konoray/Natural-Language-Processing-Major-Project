import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
train_df = pd.read_csv('训练集结果.csv')
test_df = pd.read_csv('测试集结果.csv')

# 准备数据 - 处理NaN值和布尔值转换
train_df_clean = train_df.dropna(subset=['is_fraud'])
test_df_clean = test_df.dropna(subset=['is_fraud'])

X_train = train_df_clean['specific_dialogue_content'].astype(str)
y_train = train_df_clean['is_fraud'].astype(int)

X_test = test_df_clean['specific_dialogue_content'].astype(str)
y_test = test_df_clean['is_fraud'].astype(int)

print(f"训练集清洗后样本数: {len(train_df_clean)} (移除了{len(train_df) - len(train_df_clean)}个NaN值)")
print(f"测试集清洗后样本数: {len(test_df_clean)} (移除了{len(test_df) - len(test_df_clean)}个NaN值)")

# 使用TF-IDF进行特征提取
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words=['的', '了', '和', '是', '在', '有', '为', '与', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']
)

# 拟合并转换训练集
X_train_tfidf = vectorizer.fit_transform(X_train)
# 转换测试集
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF特征维度: {X_train_tfidf.shape[1]}")

# 训练SVM模型
print("\n=== 训练SVM模型 ===")
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# 训练随机森林模型
print("\n=== 训练随机森林模型 ===")
rfc_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rfc_model.fit(X_train_tfidf, y_train)

# 评估SVM模型
print("\n=== SVM模型评估 ===")
y_pred_svm = svm_model.predict(X_test_tfidf)
print(f"准确率: {accuracy_score(y_test, y_pred_svm):.2%}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_svm))

# 评估随机森林模型
print("\n=== 随机森林模型评估 ===")
y_pred_rfc = rfc_model.predict(X_test_tfidf)
print(f"准确率: {accuracy_score(y_test, y_pred_rfc):.2%}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_rfc))

# 绘制混淆矩阵

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非欺诈', '欺诈'],
                yticklabels=['非欺诈', '欺诈'])
    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300)
    plt.close()

plot_confusion_matrix(y_test, y_pred_svm, 'SVM')
plot_confusion_matrix(y_test, y_pred_rfc, '随机森林')

# 保存模型和向量器
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(rfc_model, 'random_forest_model.pkl')

print("\n模型和向量器已保存成功！")
print("保存的文件:")
print("- tfidf_vectorizer.pkl: TF-IDF向量器")
print("- svm_model.pkl: SVM分类器")
print("- random_forest_model.pkl: 随机森林分类器")
