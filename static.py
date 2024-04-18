# 读取完整的数据集文件
with open('C:\\Users\\A\\Desktop\\EDU\\ATKT\\dataset\\assist2012\\assist2012.txt', 'r') as file:
    data_lines = file.read().splitlines()

# 初始化最大题目ID为负无穷
max_question_id = float('-inf')

# 遍历数据集，提取题目ID并更新最大值
for i in range(1, len(data_lines), 3):  # 从第二行开始，每三行包含一个题目ID
    question_ids = list(map(int, data_lines[i].split(',')))  # 将题目ID从逗号分隔的字符串转换为整数列表
    max_question_id = max(max_question_id, max(question_ids))

# 打印最大题目ID
print("最大题目ID:", max_question_id)
# 初始化题目数量之和为0
# total_questions = 0

# # 遍历数据集，提取每组数据的第一行（题目数量）并相加
# for i in range(0, len(data_lines), 3):  # 每三行包含一组数据，第一行是题目数量
#     num_questions = int(data_lines[i])  # 提取题目数量并转换为整数
#     total_questions += num_questions

# # 打印总题目数量
# print("总题目数量:", total_questions)