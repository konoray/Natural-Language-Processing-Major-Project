import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取测试结果报告
print("=== 读取测试结果报告 ===")
test_results = {
    'SVM': {
        '原始测试集': 98.94,
        '同义词替换': 98.78,
        '句式转换': 98.94,
        '组合方法': 98.86
    },
    '随机森林': {
        '原始测试集': 91.13,
        '同义词替换': 90.82,
        '句式转换': 91.13,
        '组合方法': 90.78
    }
}

# 1. 分析对抗样本效果有限的原因
print("\n=== 1. 对抗样本效果分析 ===")
print("当前对抗样本生成方法对模型准确率影响较小，可能的原因：")
print("1. 简单的同义词替换和句式转换难以改变模型的特征表示")
print("2. TF-IDF特征对这种类型的攻击不敏感")
print("3. 模型训练充分，对简单对抗攻击具有鲁棒性")
print("4. 对抗样本生成规则不够全面，覆盖的模式有限")

# 2. 分析数据集中的高频关键词
print("\n=== 2. 数据集高频关键词分析 ===")
# 读取数据
adversarial_df = pd.read_csv('测试集_对抗样本.csv')

# 分词并统计高频词
def get_top_keywords(text_series, top_n=20):
    # 合并所有文本
    all_text = ' '.join(text_series.astype(str))
    # 分词
    words = list(jieba.cut(all_text))
    # 过滤停用词和短词
    stop_words = ['的', '了', '和', '是', '在', '有', '为', '与', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '我', '是', '你', '他', '她', '它', '们', '来', '去', '啊', '哦', '嗯', '呢', '吧', '嗨', '喂', '嘿', '哎', '哎', '哦', '嗯', '呢', '吧', '嗨', '喂', '嘿', '哎', '\n', ' ', '\t', 'audio', '内容', 'left', 'right', '**']
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    # 统计词频
    word_counts = Counter(filtered_words)
    # 获取前N个高频词
    top_words = word_counts.most_common(top_n)
    return top_words

# 分析原始文本和对抗文本的关键词差异
original_texts = adversarial_df['specific_dialogue_content']
synonym_texts = adversarial_df['specific_dialogue_content_synonym']
combined_texts = adversarial_df['specific_dialogue_content_combined']

# 获取高频词
top_original = get_top_keywords(original_texts, 15)
top_synonym = get_top_keywords(synonym_texts, 15)
top_combined = get_top_keywords(combined_texts, 15)

print("\n原始文本高频词：")
for word, count in top_original:
    print(f"{word}: {count}")

print("\n同义词替换文本高频词：")
for word, count in top_synonym:
    print(f"{word}: {count}")

print("\n组合方法文本高频词：")
for word, count in top_combined:
    print(f"{word}: {count}")

# 3. 改进的对抗样本生成策略
print("\n=== 3. 改进的对抗样本生成策略 ===")
print("基于分析，提出以下改进策略：")
print("1. 扩展同义词词典，覆盖更多高频关键词")
print("2. 增加更多句式转换规则，尤其是针对高频欺诈话术")
print("3. 引入基于上下文的同义词替换，考虑词语在对话中的具体含义")
print("4. 增加词序调整，句子重组等更复杂的改写方法")
print("5. 针对TF-IDF特征的特性，重点替换权重高的关键词")

# 4. 实现改进的对抗样本生成
print("\n=== 4. 实现改进的对抗样本生成 ===")

# 扩展同义词词典
extended_synonyms_dict = {
    '客服': ['客服中心', '客户服务', '售后', '服务中心', '在线客服', '人工客服'],
    '退款': ['返还', '退回', '退钱', '返现', '退款申请', '退款处理'],
    '链接': ['网址', '链接地址', '网页', '链接入口', '访问地址', '跳转链接'],
    '商品': ['物品', '产品', '货品', '商品信息', '购买的商品', '所购产品'],
    '购买': ['选购', '订购', '买入', '下单', '购买了', '选购了'],
    '权益': ['利益', '权利', '保障', '权益保障', '合法权益', '应有权益'],
    '验证': ['确认', '核实', '校验', '验证信息', '身份验证', '信息核实'],
    '订单': ['定单', '购买记录', '交易', '订单信息', '交易记录', '订单详情'],
    '操作': ['处理', '操作步骤', '执行', '操作流程', '进行操作', '按照操作'],
    '系统': ['平台', '服务器', '系统后台', '后台系统', '平台系统', '系统平台'],
    '问题': ['状况', '情况', '异常', '问题出现', '出现问题', '存在问题'],
    '解决方案': ['处理方式', '解决办法', '应对策略', '解决措施', '解决方案', '应对方案'],
    '短信': ['短消息', '手机短信', '信息', '手机信息', '短信通知', '发送短信'],
    '账户': ['账号', '账户信息', '用户账号', '账号信息', '个人账户', '用户账户'],
    '信息': ['资料', '数据', '内容', '相关信息', '信息内容', '详细信息'],
    '安全': ['保障', '安全保障', '防护', '安全防护', '安全措施', '安全保障'],
    '点击': ['点选', '点击进入', '访问', '点击查看', '点击链接', '访问链接'],
    '按照': ['依照', '根据', '遵循', '按照要求', '根据提示', '依照指引'],
    '提示': ['指引', '提示信息', '说明', '提示内容', '操作指引', '操作提示'],
    '完成': ['结束', '完成操作', '搞定', '完成流程', '操作完成', '完成处理'],
    '联系': ['联络', '取得联系', '联系我们', '与我们联系', '联系客服', '联系工作人员'],
    '专人为您': ['专门人员为您', '专业人员为您', '专人负责', '专业人员负责', '专人处理', '专业人员处理'],
    '解答': ['解释', '解答问题', '回答', '解答疑问', '进行解答', '详细解答'],
    '注意': ['留意', '注意事项', '关注', '请注意', '注意查收', '注意查看'],
    '祝您生活愉快': ['祝您愉快', '祝您生活幸福', '祝您一切顺利', '祝您万事如意', '祝您生活美满', '祝您工作顺利'],
    '再见': ['拜拜', '再见了', '下次见', '拜拜了', '有缘再见', '后会有期'],
    # 新增高频词同义词
    '投资': ['理财', '投资项目', '金融投资', '投资机会', '投资产品', '投资计划'],
    '收益': ['利润', '回报', '收益率', '收益情况', '收益回报', '投资收益'],
    '数字货币': ['虚拟货币', '加密货币', '数字资产', '虚拟资产', '加密资产', '数字币'],
    '基金': ['理财产品', '投资基金', '基金产品', '金融产品', '投资产品', '理财产品'],
    '年化': ['年度', '年息', '年利率', '年化收益', '年化利率', '年收益率'],
    '安全': ['可靠', '安全可靠', '稳健', '风险低', '安全性高', '低风险'],
    '下载': ['安装', '下载安装', '获取', '下载应用', '安装应用', '下载软件'],
    '应用程序': ['APP', '应用', '软件', '程序', '应用软件', '手机应用'],
    '订单号': ['订单编号', '交易号', '订单号码', '交易编号', '订单ID', '交易ID'],
    '链接': ['网址', '网页地址', '网站链接', '页面链接', '访问链接', '跳转链接'],
    '短信': ['短消息', '手机短信', '信息', '手机信息', '短信通知', '发送短信'],
    '验证码': ['短信验证码', '验证代码', '校验码', '验证号码', '短信校验码', '安全验证码'],
    '银行卡': ['储蓄卡', '信用卡', '银行卡号', '卡号', '银行账户', '账户号码'],
    '密码': ['登录密码', '支付密码', '安全密码', '密码信息', '账户密码', '交易密码']
}

# 扩展句式转换规则
extended_patterns = [
    ('我们注意到您最近在我们平台购买的商品出现的一些问题', '我们发现您近期在本平台选购的产品存在某些状况'),
    ('为了保证您的权益，然后我们特地为您申请了一笔退款', '为保障您的利益，我们专门为您申请了退款'),
    ('请您点击我们发送的链接，按照提示操作即可完成退款流程', '请您访问我们发送的网址，根据指引操作就能完成退款'),
    ('如果您在操作过程中遇到任何问题，可以随时联系我们的客服的话', '若您在处理过程中遇到任何状况，可随时联络我们的客服'),
    ('我们会有专人为您解答', '我们将安排专业人员为您解释'),
    ('我们需要你点击一个链接来验证你的订单信息', '我们需要您访问一个网址来确认您的订单数据'),
    ('我可以发一个短信给你，里面有一个链接', '我会发送一条短消息给您，内附相关网址'),
    ('你点击后按照提示操作就可以解决这个问题了', '您访问后根据指引操作即可解决此问题'),
    # 新增句式转换规则
    ('最近我们有个特别好的投资项目，想了解一下您的意向', '近期我们推出了一个优质的投资机会，想询问您是否有兴趣'),
    ('这是一个新推出的数字货币基金，预计年化收益率可以达到百分之十五', '这是我们最新上线的加密货币理财产品，预计年收益率可达15%'),
    ('我们的风控措施非常严格，绝对安全', '我们的风险控制体系相当完善，安全性极高'),
    ('检测到你最近购买的一件商品出现了质量问题，我们要给你办理退款', '系统显示您近期所购商品存在质量异常，我们将为您办理退款手续'),
    ('为了确保你的退款能够快速到账，我们需要你下载一个官方的应用程序', '为保证您的退款及时到账，我们需要您获取并安装官方指定的应用程序'),
    ('你只需要点击我发送给你的链接，按照提示下载应用程序', '您只需访问我发给您的网址，根据操作指引获取并安装应用程序'),
    ('然后输入你的订单号和银行卡信息', '随后输入您的订单编号及银行账户信息'),
    ('我们的客服人员会在24小时内联系您', '我们的服务人员将在1个工作日内与您取得联系'),
    ('请您提供一下您的身份证号码和手机号码', '麻烦您提供您的身份证件号码以及联系电话'),
    ('这是我们公司的优惠活动，只有今天有效', '这是我们平台的限时特惠，仅限今日'),
    ('您只需要支付少量的手续费，就可以获得高额的回报', '您只需缴纳小额的服务费，即可获取丰厚的收益回报'),
    ('我们的产品已经通过了国家认证，绝对安全可靠', '我们的产品已获得官方认证，安全性有保障')
]

# 词序调整方法
def word_order_shuffle(text):
    # 简单的词序调整，仅针对逗号分隔的短句
    sentences = text.split('，')
    shuffled_sentences = []
    for sentence in sentences:
        # 仅对较长的句子进行词序调整
        if len(sentence) > 10 and 'left:' not in sentence and 'right:' not in sentence:
            # 分词
            words = list(jieba.cut(sentence))
            # 简单的词序调整，将一些修饰词移到后面
            if len(words) > 5:
                # 例如："我们公司推出了新产品" -> "我们公司新产品推出了"
                if '推出' in words and '了' in words:
                    推出_idx = words.index('推出')
                    if 推出_idx + 1 < len(words) and words[推出_idx + 1] == '了':
                        # 找到推出后面的名词短语
                        for i in range(推出_idx + 2, len(words)):
                            if words[i] in ['，', '。', '！', '？']:
                                break
                        # 调整词序
                        if i > 推出_idx + 2:
                            words = words[:推出_idx] + words[推出_idx + 2:i] + ['推出', '了'] + words[i:]
            shuffled_sentences.append(''.join(words))
        else:
            shuffled_sentences.append(sentence)
    return '，'.join(shuffled_sentences)

# 改进的对抗样本生成
def improved_adversarial_generation(text):
    # 1. 扩展的同义词替换
    words = list(jieba.cut(text))
    replaceable_words = []
    for i, word in enumerate(words):
        if word in extended_synonyms_dict:
            replaceable_words.append(i)
    
    # 提高替换比例
    replacement_ratio = 0.5
    num_replace = max(1, int(len(replaceable_words) * replacement_ratio))
    
    if len(replaceable_words) > 0:
        import random
        # 选择所有可替换的词
        replace_indices = random.sample(replaceable_words, min(num_replace, len(replaceable_words)))
        
        for idx in replace_indices:
            original_word = words[idx]
            synonyms = extended_synonyms_dict[original_word]
            new_word = random.choice(synonyms)
            words[idx] = new_word
    
    text = ''.join(words)
    
    # 2. 扩展的句式转换
    for old_pattern, new_pattern in extended_patterns:
        if old_pattern in text:
            text = text.replace(old_pattern, new_pattern)
    
    # 3. 词序调整
    text = word_order_shuffle(text)
    
    return text

# 测试改进的对抗样本生成
sample_text = adversarial_df['specific_dialogue_content'].iloc[0]
print(f"\n原始文本：")
print(sample_text[:300] + "...")

improved_text = improved_adversarial_generation(sample_text)
print(f"\n改进后的对抗样本：")
print(improved_text[:300] + "...")

# 4. 生成改进的对抗样本数据集
print("\n=== 5. 生成改进的对抗样本数据集 ===")
print("使用改进的对抗样本生成方法生成新的对抗样本...")

# 生成改进的对抗样本
improved_adversarial_texts = []
for i, row in adversarial_df.iterrows():
    text = row['specific_dialogue_content']
    if isinstance(text, str):
        adv_text = improved_adversarial_generation(text)
        improved_adversarial_texts.append(adv_text)
    else:
        improved_adversarial_texts.append(text)
    
    if (i + 1) % 500 == 0:
        print(f"已处理 {i + 1}/{len(adversarial_df)} 个样本")

# 添加到数据集
adversarial_df['specific_dialogue_content_improved'] = improved_adversarial_texts

# 保存改进后的对抗样本数据集
adversarial_df.to_csv('测试集_对抗样本_改进版.csv', index=False, encoding='utf-8-sig')
print(f"\n改进后的对抗样本数据集已保存到: 测试集_对抗样本_改进版.csv")

# 5. 可视化结果
print("\n=== 6. 结果可视化 ===")

# 绘制当前对抗样本效果对比图
plt.figure(figsize=(12, 6))
test_types = list(test_results['SVM'].keys())
svm_accs = [test_results['SVM'][name] for name in test_types]
rfc_accs = [test_results['随机森林'][name] for name in test_types]

x = np.arange(len(test_types))
width = 0.35

plt.bar(x - width/2, svm_accs, width, label='SVM')
plt.bar(x + width/2, rfc_accs, width, label='随机森林')

plt.xlabel('测试集类型')
plt.ylabel('准确率 (%)')
plt.title('当前对抗样本对模型准确率的影响')
plt.xticks(x, test_types)
plt.ylim(90, 100)

# 添加数值标签
for i, v in enumerate(svm_accs):
    plt.text(i - width/2, v + 0.1, f'{v:.2f}%', ha='center', va='bottom')
for i, v in enumerate(rfc_accs):
    plt.text(i + width/2, v + 0.1, f'{v:.2f}%', ha='center', va='bottom')

plt.legend()
plt.tight_layout()
plt.savefig('ablation_study_current.png', dpi=300)
print("\n当前对抗样本效果对比图已保存为 ablation_study_current.png")

# 6. 结论与建议
print("\n=== 7. 结论与建议 ===")
print("当前对抗样本生成方法效果有限，主要原因是：")
print("1. 简单的同义词替换和句式转换难以改变模型的核心特征")
print("2. 模型对这种类型的攻击具有一定的鲁棒性")
print("3. 对抗样本生成规则覆盖的模式有限")

print("\n改进建议：")
print("1. 采用更高级的对抗样本生成方法，如基于梯度的攻击或生成式模型")
print("2. 针对模型的核心特征进行攻击，重点替换权重高的关键词")
print("3. 结合多种攻击方法，提高对抗样本的多样性和复杂性")
print("4. 考虑使用大模型生成更自然、更具欺骗性的对抗样本")
print("5. 针对不同模型设计特定的攻击策略，提高攻击的针对性")

print("\n消融实验完成，改进后的对抗样本生成方法已实现，可进一步测试其效果。")
