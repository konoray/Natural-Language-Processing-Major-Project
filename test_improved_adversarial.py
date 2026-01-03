import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载训练好的模型和向量器
print("=== 加载模型和向量器 ===")
vectorizer = joblib.load('tfidf_vectorizer.pkl')
svm_model = joblib.load('svm_model.pkl')
rfc_model = joblib.load('random_forest_model.pkl')

# 加载改进后的对抗样本数据集
print("\n=== 加载改进后的对抗样本数据集 ===")
improved_df = pd.read_csv('测试集_对抗样本_改进版.csv')
print(f"改进后的对抗样本数据集大小: {len(improved_df)}")

# 准备数据
y_test = improved_df['is_fraud'].astype(int)

# 定义测试方法
def test_model_performance(model_name, model, X_test_text, y_test):
    # 使用向量器转换文本
    X_test_tfidf = vectorizer.transform(X_test_text)
    # 预测
    y_pred = model.predict(X_test_tfidf)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    # 生成分类报告
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

# 测试集类型
test_types = {
    '原始测试集': 'specific_dialogue_content',
    '组合方法': 'specific_dialogue_content_combined',
    '改进方法': 'specific_dialogue_content_improved'
}

# 存储结果
results = {
    'SVM': {},
    '随机森林': {}
}

# 测试所有数据集类型
for test_name, text_column in test_types.items():
    print(f"\n=== 测试{test_name} ===")
    
    # 准备文本数据
    X_test_text = improved_df[text_column].astype(str)
    
    # 测试SVM模型
    svm_acc, svm_report = test_model_performance('SVM', svm_model, X_test_text, y_test)
    results['SVM'][test_name] = {
        'accuracy': svm_acc,
        'report': svm_report
    }
    print(f"SVM准确率: {svm_acc:.2%}")
    
    # 测试随机森林模型
    rfc_acc, rfc_report = test_model_performance('随机森林', rfc_model, X_test_text, y_test)
    results['随机森林'][test_name] = {
        'accuracy': rfc_acc,
        'report': rfc_report
    }
    print(f"随机森林准确率: {rfc_acc:.2%}")

# 可视化结果
def plot_improved_accuracy_comparison(results):
    plt.figure(figsize=(12, 6))
    
    # 提取数据
    test_names = list(test_types.keys())
    svm_accs = [results['SVM'][name]['accuracy'] for name in test_names]
    rfc_accs = [results['随机森林'][name]['accuracy'] for name in test_names]
    
    # 绘制柱状图
    x = np.arange(len(test_names))
    width = 0.35
    
    plt.bar(x - width/2, svm_accs, width, label='SVM')
    plt.bar(x + width/2, rfc_accs, width, label='随机森林')
    
    # 添加标签和标题
    plt.xlabel('测试集类型')
    plt.ylabel('准确率')
    plt.title('改进前后对抗样本对模型准确率的影响对比')
    plt.xticks(x, test_names)
    plt.ylim(0.8, 1.0)
    
    # 在柱子上添加准确率数值
    for i, v in enumerate(svm_accs):
        plt.text(i - width/2, v + 0.005, f'{v:.2%}', ha='center', va='bottom')
    for i, v in enumerate(rfc_accs):
        plt.text(i + width/2, v + 0.005, f'{v:.2%}', ha='center', va='bottom')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('improved_accuracy_comparison.png', dpi=300)
    plt.close()
    print("\n改进前后准确率对比图已保存为 improved_accuracy_comparison.png")

# 生成详细报告
def generate_improved_report(results, filename='improved_adversarial_test_report.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== 改进后对抗样本测试详细报告 ===\n\n")
        
        for model_name in results:
            f.write(f"\n{model_name}模型测试结果\n")
            f.write("=" * 50 + "\n")
            
            for test_name in results[model_name]:
                f.write(f"\n{test_name}:\n")
                accuracy = results[model_name][test_name]['accuracy']
                f.write(f"准确率: {accuracy:.2%}\n")
                
                report = results[model_name][test_name]['report']
                f.write("分类报告:\n")
                f.write(f"{'类':<8} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}\n")
                f.write("-" * 50 + "\n")
                for class_name in ['0', '1']:
                    class_report = report[class_name]
                    f.write(f"{class_name:<8} {class_report['precision']:<10.2f} {class_report['recall']:<10.2f} {class_report['f1-score']:<10.2f} {class_report['support']:<10}\n")
                avg_report = report['weighted avg']
                f.write("-" * 50 + "\n")
                f.write(f"{'加权平均':<8} {avg_report['precision']:<10.2f} {avg_report['recall']:<10.2f} {avg_report['f1-score']:<10.2f} {avg_report['support']:<10}\n")
    
    print(f"\n改进后的详细报告已保存为 {filename}")

# 可视化结果
plot_improved_accuracy_comparison(results)

# 生成详细报告
generate_improved_report(results)

# 打印摘要结果
print("\n=== 测试结果摘要 ===")
print(f"{'测试集类型':<12} {'SVM准确率':<12} {'随机森林准确率':<12}")
print("-" * 40)
for test_name in test_types:
    svm_acc = results['SVM'][test_name]['accuracy']
    rfc_acc = results['随机森林'][test_name]['accuracy']
    print(f"{test_name:<12} {svm_acc:<12.2%} {rfc_acc:<12.2%}")

# 计算准确率下降情况
print("\n=== 准确率下降情况 ===")
print(f"{'测试集类型':<12} {'SVM下降率':<12} {'随机森林下降率':<12}")
print("-" * 40)
original_svm_acc = results['SVM']['原始测试集']['accuracy']
original_rfc_acc = results['随机森林']['原始测试集']['accuracy']

for test_name in list(test_types.keys())[1:]:  # 跳过原始测试集
    svm_acc = results['SVM'][test_name]['accuracy']
    rfc_acc = results['随机森林'][test_name]['accuracy']
    
    svm_drop = (original_svm_acc - svm_acc) / original_svm_acc
    rfc_drop = (original_rfc_acc - rfc_acc) / original_rfc_acc
    
    print(f"{test_name:<12} {svm_drop:<12.2%} {rfc_drop:<12.2%}")

# 分析改进方法的效果
print("\n=== 改进方法效果分析 ===")
print("1. 改进的对抗样本生成方法结合了以下技术：")
print("   - 扩展的同义词词典，覆盖更多高频关键词")
print("   - 更多的句式转换规则，特别是针对欺诈话术")
print("   - 词序调整和句子重组")
print("   - 更高的替换比例")
print("\n2. 改进前后效果对比：")
print(f"   - SVM模型改进前准确率: {results['SVM']['组合方法']['accuracy']:.2%}")
print(f"   - SVM模型改进后准确率: {results['SVM']['改进方法']['accuracy']:.2%}")
print(f"   - 准确率下降: {(results['SVM']['组合方法']['accuracy'] - results['SVM']['改进方法']['accuracy']):.2%}")
print(f"   - 随机森林模型改进前准确率: {results['随机森林']['组合方法']['accuracy']:.2%}")
print(f"   - 随机森林模型改进后准确率: {results['随机森林']['改进方法']['accuracy']:.2%}")
print(f"   - 准确率下降: {(results['随机森林']['组合方法']['accuracy'] - results['随机森林']['改进方法']['accuracy']):.2%}")
