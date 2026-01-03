import pandas as pd
import numpy as np
import random
import jieba
import time

# 同义词词典（与之前的脚本保持一致）
synonyms_dict = {
    '客服': ['客服中心', '客户服务', '售后'],
    '退款': ['返还', '退回', '退钱'],
    '链接': ['网址', '链接地址', '网页'],
    '商品': ['物品', '产品', '货品'],
    '购买': ['选购', '订购', '买入'],
    '权益': ['利益', '权利', '保障'],
    '验证': ['确认', '核实', '校验'],
    '订单': ['定单', '购买记录', '交易'],
    '操作': ['处理', '操作步骤', '执行'],
    '系统': ['平台', '服务器', '系统后台'],
    '问题': ['状况', '情况', '异常'],
    '解决方案': ['处理方式', '解决办法', '应对策略'],
    '短信': ['短消息', '手机短信', '信息'],
    '验证': ['确认', '核实', '校验'],
    '账户': ['账号', '账户信息', '用户账号'],
    '信息': ['资料', '数据', '内容'],
    '安全': ['保障', '安全保障', '防护'],
    '点击': ['点选', '点击进入', '访问'],
    '按照': ['依照', '根据', '遵循'],
    '提示': ['指引', '提示信息', '说明'],
    '完成': ['结束', '完成操作', '搞定'],
    '联系': ['联络', '取得联系', '联系我们'],
    '专人为您': ['专门人员为您', '专业人员为您', '专人负责'],
    '解答': ['解释', '解答问题', '回答'],
    '注意': ['留意', '注意事项', '关注'],
    '祝您生活愉快': ['祝您愉快', '祝您生活幸福', '祝您一切顺利'],
    '再见': ['拜拜', '再见了', '下次见']
}

# 基于规则的同义词替换方法
def synonym_replacement(text, replacement_ratio=0.3):
    # 使用jieba进行分词
    words = list(jieba.cut(text))
    
    # 记录需要替换的词的位置
    replaceable_words = []
    for i, word in enumerate(words):
        if word in synonyms_dict:
            replaceable_words.append(i)
    
    # 确定替换数量
    num_replace = max(1, int(len(replaceable_words) * replacement_ratio))
    
    # 随机选择替换的词
    if len(replaceable_words) > 0:
        replace_indices = random.sample(replaceable_words, min(num_replace, len(replaceable_words)))
        
        # 进行替换
        for idx in replace_indices:
            original_word = words[idx]
            synonyms = synonyms_dict[original_word]
            # 随机选择一个同义词
            new_word = random.choice(synonyms)
            words[idx] = new_word
    
    # 重新组合文本
    return ''.join(words)

# 句式转换方法（简单示例）
def sentence_rephrasing(text):
    # 定义一些句式转换规则
    patterns = [
        ('我们注意到您最近在我们平台购买的商品出现的一些问题', '我们发现您近期在本平台选购的产品存在某些状况'),
        ('为了保证您的权益，然后我们特地为您申请了一笔退款', '为保障您的利益，我们专门为您申请了退款'),
        ('请您点击我们发送的链接，按照提示操作即可完成退款流程', '请您访问我们发送的网址，根据指引操作就能完成退款'),
        ('如果您在操作过程中遇到任何问题，可以随时联系我们的客服的话', '若您在处理过程中遇到任何状况，可随时联络我们的客服'),
        ('我们会有专人为您解答', '我们将安排专业人员为您解释'),
        ('我们需要你点击一个链接来验证你的订单信息', '我们需要您访问一个网址来确认您的订单数据'),
        ('我可以发一个短信给你，里面有一个链接', '我会发送一条短消息给您，内附相关网址'),
        ('你点击后按照提示操作就可以解决这个问题了', '您访问后根据指引操作即可解决此问题')
    ]
    
    # 应用句式转换
    for old_pattern, new_pattern in patterns:
        if old_pattern in text:
            text = text.replace(old_pattern, new_pattern)
    
    return text

# 组合式对抗生成
def generate_adversarial_example(text, method='combined'):
    if method == 'synonym':
        return synonym_replacement(text)
    elif method == 'rephrase':
        return sentence_rephrasing(text)
    elif method == 'combined':
        # 先进行同义词替换，再进行句式转换
        text = synonym_replacement(text)
        text = sentence_rephrasing(text)
        return text
    else:
        return text

# 读取测试集数据
test_df = pd.read_csv('测试集结果.csv')

# 移除NaN值
test_df_clean = test_df.dropna(subset=['is_fraud'])
print(f"测试集清洗后样本数: {len(test_df_clean)}")

# 生成对抗样本
def generate_adversarial_dataset(df, methods=['synonym', 'rephrase', 'combined']):
    # 复制原始数据
    result_df = df.copy()
    
    for method in methods:
        print(f"\n=== 生成{method}方法的对抗样本 ===")
        start_time = time.time()
        
        # 生成对抗样本
        adversarial_texts = []
        for i, row in df.iterrows():
            text = row['specific_dialogue_content']
            if isinstance(text, str):
                adv_text = generate_adversarial_example(text, method)
                adversarial_texts.append(adv_text)
            else:
                adversarial_texts.append(text)
            
            # 每处理100个样本显示一次进度
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(df)} 个样本")
        
        # 添加到结果数据框
        result_df[f'specific_dialogue_content_{method}'] = adversarial_texts
        
        end_time = time.time()
        print(f"生成{method}方法的对抗样本耗时: {end_time - start_time:.2f}秒")
    
    return result_df

# 生成对抗样本数据集
adversarial_df = generate_adversarial_dataset(test_df_clean)

# 保存结果
save_path = '测试集_对抗样本.csv'
adversarial_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"\n对抗样本数据集已保存到: {save_path}")
print(f"原始列数: {len(test_df_clean.columns)}")
print(f"生成后列数: {len(adversarial_df.columns)}")

# 显示生成前后的对比示例
print("\n=== 生成前后对比示例 ===")
for i in range(3):
    print(f"\n示例 {i+1}:")
    print("原始文本:")
    print(adversarial_df['specific_dialogue_content'].iloc[i][:200] + "...")
    print("\n同义词替换:")
    print(adversarial_df['specific_dialogue_content_synonym'].iloc[i][:200] + "...")
    print("\n句式转换:")
    print(adversarial_df['specific_dialogue_content_rephrase'].iloc[i][:200] + "...")
    print("\n组合方法:")
    print(adversarial_df['specific_dialogue_content_combined'].iloc[i][:200] + "...")
