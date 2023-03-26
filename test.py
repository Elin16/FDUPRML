from collections import Counter

c = Counter({'a':1, 'b':3, 'c':3})

print(c.most_common(1)[0][0]) # 默认参数