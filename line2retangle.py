import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Polygon

def line_to_rectangle(line_start, line_end):
    line_length = np.linalg.norm(line_end - line_start)  # 计算线段的长度
    line_angle = np.arctan2(line_end[1] - line_start[1], line_end[0] - line_start[0])  # 计算线段的角度

    # 创建矩形框
    rectangle_width = line_length  # 宽度设置为线段的长度
    rectangle_height = 1 # 根据需求设置矩形框的高度
    rectangle_center = (line_start + line_end) / 2  # 位置设置为线段的中点
    rectangle_angle = np.degrees(line_angle)  # 将角度转换为度数
    points = cv2.boxPoints((tuple(rectangle_center), (rectangle_width, rectangle_height), rectangle_angle))
    print(points)
    # rectangle = plt.Rectangle(rectangle_center, rectangle_width, rectangle_height, angle=rectangle_angle, color='r')
    return points

# 示例线段的起点和终点坐标
line_start = np.array([0, 0])
line_end = np.array([5, 5])

# 将线段转换为矩形框
rectangle = line_to_rectangle(line_start, line_end)

fig, ax = plt.subplots()
# 绘制线段和矩形框
plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'b', linewidth=2)  # 绘制线段
# 创建多边形对象
polygon = Polygon(rectangle, closed=True, alpha=0.5)
ax.add_patch(polygon)
# plt.gca().add_patch(rectangle)  # 绘制矩形框
# plt.axis('equal')
plt.show()
