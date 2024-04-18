###loss可视化
import json
import matplotlib.pyplot as plt

# 加载保存的训练损失数据
with open("2017train.json", "r") as f:
    train_loss = json.load(f)

epochs = list(range(1, len(train_loss) + 1))

# 绘制损失曲线图
plt.plot(epochs, train_loss, 'b', label='Loss')
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


###过滤前后的skill可视化
# import torch

# a=torch.load("a.pth")
# b=torch.load("b.pth")

# # print(a[0].size(),b[0].size())
# print(a[0][0],'\n',b[0][0])
# print(a[0][0].max(),'\n',b[0][0].max())
# print(a[0][0].min(),'\n',b[0][0].min())
# pre=a[0][0].unsqueeze(dim=0)
# post=b[0][0].unsqueeze(dim=0)
# # print(pre)
# #torch.Size([500, 352]) torch.Size([500, 352])

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# # 创建一个示例的嵌入向量
# batch_size,length,dim = pre.size()

# embeddings = np.random.rand(batch_size, length, dim)

# # 将嵌入向量展平为二维数组
# flattened_embeddings = embeddings.reshape(batch_size * length, dim)

# # 使用主成分分析(PCA)进行降维
# pca = PCA(n_components=2)
# pca_embeddings = pca.fit_transform(flattened_embeddings)

# # 使用t-SNE进行降维
# tsne = TSNE(n_components=2)
# tsne_embeddings = tsne.fit_transform(flattened_embeddings)

# # 可视化PCA降维结果
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1])
# plt.title('PCA')

# # 可视化t-SNE降维结果
# plt.subplot(1, 2, 2)
# plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
# plt.title('t-SNE')

# # 显示图形
# plt.tight_layout()
# plt.show()

# # 保存图片
# plt.savefig('pre.png')


# # 创建一个示例的嵌入向量
# batch_size,length,dim = post.size()

# embeddings = np.random.rand(batch_size, length, dim)

# # 将嵌入向量展平为二维数组
# flattened_embeddings = embeddings.reshape(batch_size * length, dim)

# # 使用主成分分析(PCA)进行降维
# pca = PCA(n_components=2)
# pca_embeddings = pca.fit_transform(flattened_embeddings)

# # 使用t-SNE进行降维
# tsne = TSNE(n_components=2)
# tsne_embeddings = tsne.fit_transform(flattened_embeddings)

# # 可视化PCA降维结果
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1])
# plt.title('PCA')

# # 可视化t-SNE降维结果
# plt.subplot(1, 2, 2)
# plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
# plt.title('t-SNE')

# # 显示图形
# plt.tight_layout()
# plt.show()

# # 保存图片
# plt.savefig('post.png')

# ##热力图
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.patches import Wedge


# def read_data(path,start,end):
#     # 加载数据
#     with open(path, 'rb') as file:
#         loaded_data = pickle.load(file)
#     # 从文件中检索张量
#     # 获取每个时间步对应的技能ID
#     skill_ids = loaded_data['tensor1'][start:end].numpy()  # skill

#     answer  = loaded_data['tensor2'][start:end].numpy()  # answer 
#     predicted = loaded_data['tensor3'][start:end,:].numpy() # pre 
    
#     return skill_ids,answer,predicted


# skill_ids,answer,predicted = read_data('saved_data.pkl',22,48)

# print("skill_ids: ",skill_ids)

# print("true_labels: ",answer)

# # skill到answer的映射
# skill_answer = [(key,value) for key, value in zip(skill_ids, answer)]
# print("skill_answer: ",skill_answer)

# # skill index
# skill_index = np.unique(skill_ids)
# print("skill_index: ",skill_index)

# #skill to name ,there will be a dict to show skill name  

# '''···'''
# #根据 skill index获取它们在每一个时间步的预测值 
# predicted_perstep = predicted[:,skill_index].T
# print("predicted_perstep: ",predicted_perstep)


# # 创建热力图
# plt.figure(figsize=(12, 6))  # 设置图形大小

# # 绘制热力图
# cmap = plt.get_cmap('viridis')
# ax = sns.heatmap(predicted_perstep, cmap=cmap, annot=False, square="equal", xticklabels=True, yticklabels=False, cbar=False)
# # 设置热力图的背景色为透明
# ax.set_facecolor('none')

# # 绘制颜色条
# cbar = plt.colorbar(ax.get_children()[0], ax=ax, fraction=0.04, pad=0.04, shrink=0.4)
# cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# # 创建垂直排列的文本标签
# label_text = "Predicted Probability"
# label_text = label_text.replace(" ", "\n")  # 将空格替换为换行符
# cbar.set_label(label_text, rotation=90, labelpad=15, va="center")  # 使用va参数设置垂直对齐为"center"

# # 设置颜色条刻度标签的大小
# cbar.ax.tick_params(labelsize=8)  # 根据需要调整刻度标签大小

# # 定义颜色列表
# colors = ['red', 'orange', 'yellow', 'green', 'blue']

# # 自定义 Y 轴标签为“技能索引: ●”，调整字体大小、实心圆大小和颜色
# for i, skill_idx in enumerate(skill_index):
#     color = colors[i % len(colors)]  # 根据索引选择颜色，循环使用颜色列表
#     plt.text(-0.7, i + 0.5, f'{skill_idx}:  ', fontsize=10, va='center', ha='right', color='black')
#     plt.text(-0.55, i + 0.5, '●', fontsize=20, va='center', ha='center', color=color)

# # 绘制空心圆环和内部小球
# for i, (skill_id, answer) in enumerate(skill_answer):
#     color = colors[np.where(skill_index == skill_id)[0][0] % len(colors)]
#     circle_radius = 0.3  # 调整圆的大小
#     center_x = i + 0.5
#     center_y = len(skill_index) + 1  # 将圆环和小球放置在热力图的上方
#     if answer == 1:
#         # 绘制扇形（扇形的起始角度为0度，结束角度为360度，表示一个完整的圆形）
#         circle = Wedge(center=(center_x, center_y), r=circle_radius, theta1=0, theta2=360, fill=False, color=color, linewidth=2, zorder=10)
#         ax.add_patch(circle)
#         # 添加小球，小球位于扇形的中心
#         inner_circle = plt.Circle((center_x, center_y), circle_radius * 0.5, fill=True, color=color, zorder=20)
#         ax.add_patch(inner_circle)
#     else:
#         # 空心圆环
#         circle = plt.Circle((i + 0.5, center_y), circle_radius, fill=False, color=color, linewidth=2, zorder=10)
#         ax.add_patch(circle)

# # 设置轴标签和标题
# plt.xlabel('Time step')
# plt.ylim(0, len(skill_index) + 2)  # 调整y轴的范围以显示上方的刻度

# # 显示热力图
# plt.show()