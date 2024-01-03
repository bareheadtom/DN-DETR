from pathlib import Path
from util.plot_utils import plot_logs, plot_precision_recall

import matplotlib.pyplot as plt

# # 示例 1：使用 plot_logs() 函数并保存结果为 JPG 文件
# log_dir_1 = Path("/root/autodl-tmp/outputs/DNDETR/20231228221409/")  # 替换为你的第一个日志目录路径
# log_dir_2 = Path("/root/autodl-tmp/outputs/DABDETR/20231227162854/")  # 替换为你的第二个日志目录路径

# logs = [log_dir_1]  # 包含日志路径的列表
# fields_to_plot = [ 'mAP']  # 要绘制的字段列表

# # 调用 plot_logs 函数来绘制日志文件中指定字段的结果
# fig, axs = plot_logs(logs)

# # 保存绘制结果为 JPG 图像文件
# plt.savefig('logs_visualization.jpg')  # 文件名可根据需要自定义
# plt.close()  # 关闭图表

# 示例 2：使用 plot_precision_recall() 函数并保存结果为 JPG 文件
model_file_1 = Path("/root/autodl-tmp/outputs/DNDETR/20231228221409/log.txt")  # 替换为你的第一个模型数据文件路径
model_file_2 = Path("/root/autodl-tmp/outputs/DABDETR/20231227162854/log.txt")  # 替换为你的第二个模型数据文件路径

model_files = [model_file_1, model_file_2]  # 包含模型文件路径的列表
# 调用 plot_precision_recall 函数来绘制模型的精度-召回曲线和得分
fig, axs = plot_precision_recall(model_files, naming_scheme='iter')

# 保存绘制结果为 JPG 图像文件
plt.savefig('precision_recall_visualization.jpg')  # 文件名可根据需要自定义
plt.close()  # 关闭图表
