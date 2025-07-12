import csv

# 初始化存储数组
avg_head_times = []
avg_tail_times = []
avg_total_times = []
avg_head_powers = []
avg_tail_powers = []
avg_data_sizes = []

t = []

energy = [10.0266, 8.928740000000001, 8.23096, 7.3532399999999996, 7.62148, 7.787719999999999, 7.7426200000000005, 8.08278, 8.664000000000001, 8.30916, 7.511239999999999, 7.769500000000001, 7.60658, 7.92742, 8.00046, 8.05462, 8.59386, 8.74728, 9.470400000000001, 9.48572, 9.45816, 10.12352, 9.90312, 10.23676, 10.69184, 10.71354, 11.2897, 11.18164, 11.34244, 11.6097, 12.0404, 12.084159999999999, 12.54522, 12.52002, 12.619340000000001, 12.39894, 13.28642, 13.6512, 14.157240000000002, 14.02164, 13.89478, 15.0298, 15.0755, 15.270499999999998, 15.476420000000001, 15.64466, 14.587399999999999, 13.83934, 14.3911, 13.75262]

# 读取CSV文件
with open('final_results.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)  # 自动解析列名
    
    for row in csv_reader:
        t.append(float(row['avg_head_energy']) + float(row['avg_tail_energy']))
        # 提取目标字段并转换为浮点数（avg_data_size可能是整数）
        # avg_head_times.append(float(row['avg_head_time']))
        # avg_tail_times.append(float(row['avg_tail_time']))
        # avg_total_times.append(float(row['avg_total_time']))
        # avg_head_powers.append(float(row['avg_head_power']))
        # avg_tail_powers.append(float(row['avg_tail_power']))
        # avg_data_sizes.append(int(float(row['avg_data_size'])))  # 处理可能的浮点转整数

# 打印结果
print(t)
# print("avg_head_times:", avg_head_times)
# print("avg_tail_times:", avg_tail_times)
# print("avg_total_times:", avg_total_times)
# print("avg_head_powers:", avg_head_powers)
# print("avg_tail_powers:", avg_tail_powers)
# print("avg_data_sizes:", avg_data_sizes)