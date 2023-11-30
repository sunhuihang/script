import os,sys
ts_file = "ts_score.txt"
# 根据ts_sum进行排序
with open(ts_file, 'r') as f:
    lines = f.readlines()
lines = list(set(lines))
lines_with_ts_sum = [(float(line.split('ts_sum=')[1]), line) for line in lines]
#从大到小
lines_with_ts_sum.sort(key=lambda x: x[0], reverse=True)
#从小到大
#lines_with_ts_sum.sort(key=lambda x: x[0])
with open('ts_sorted.txt', 'w') as f:
    for line in lines_with_ts_sum:
        f.write(line[1])
