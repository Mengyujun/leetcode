## 优先级队列

堆（heap），它是一种优先队列

实际上，Python没有独立的堆类型，而只有一个包含一些堆操作函数的模块。这个模块名为**heapq**（其中的q表示队列），它包含6个函数，其中前4个与堆操作直接相关。必须使用**列表** list来表示堆对象本身。

最大第k个元素的可以使用最小堆（一个只有k个元素的堆，堆顶是最小元素，即第k大的元素） 

4个堆操作：

1. heappush 压入堆
2. heappop 弹出最小元素
3. heapify 将列表变成合法的堆 
4. heapreplace 从堆中弹出最小的元素，再压入一个新元素 （结合push and pop）
5. nlargest(n, iter)                    返回iter中n个最大的元素
6.  nsmallest(n, iter)                  返回iter中n个最小的元素

最常使用的两种方法 heappush and headpop  

```
>>> from heapq import * 
>>> from random import shuffle 
>>> data = list(range(10)) 
>>> shuffle(data) 
>>> heap = [] 
>>> for n in data: 
... heappush(heap, n) 
... 
>>> heap 
[0, 1, 3, 6, 2, 8, 4, 7, 9, 5] 
>>> heappush(heap, 0.5) 
>>> heap 
[0, 0.5, 3, 6, 1, 8, 4, 7, 9, 5, 2]

# 有输出
tmp = heappop(heap) 
```



## TIPS

1.  python 中的交换操作， 直接如下两个交换即可 

   ```
   matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
   ```

2. 注意python中的转置操作 即只进行一个上三角的交换即可 （**注意j的范围选择**）

   ```
           n = len(matrix)
           for i in range(n):
               for j in range(i, n):
                   matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
   ```

   

3.  list 自定义排序 重点参考 有dict时 或者list都可以 的操作 

   ```
   dic = dict()
   s1 = sorted(dic, key = lambda x: x.keys())
   ```

   ![img](https://images2017.cnblogs.com/blog/890856/201802/890856-20180210164808466-1688363276.png)

4. python 二维数组的初始化

   ```
           # 正确的
           dp = [[0] * (n + 1) for _ in range(m + 1)]
           或者
           marked = [[False for_ in range(n)] for _ in range(m)]
           # 错误的 注意分辨
           dp = [[0] * (n+1)] * (m+1)
   ```

   

5. python中有关队列

   ```
   ​```
   # 引入deque
   from collections import deque
   
   #定义及初始化`
   q = deque()
   # q = deque(root) # 直接初始化了 
   
   # 队列的进出操作 有一次错误是没有使用 popleft 
   q.append(cur)
   tmp = q.popleft()
   
   # 另外的queue
   from queue import Queue 
   
   q = Queue()
   # 加入和取出节点的操作 
   q.put()
   q.get()
   ​```
   ```

   

6. python最大最小值

   > maxsum = float('-inf') 
   >
   > max_sum = float('inf') 

7. python 中的方向扩展以及边界判断 

   > #  注意 小于等于的 等号的位置 即从左到右看 
   >
   > if 0 <= new_i <m and 0 <= new_j < n :

8. python中链表定义

   ```
   # class ListNode:
   #     def __init__(self, x):
   #         self.val = x
   #         self.next = None
   ```

   

9. python 中两个元素list的排序 （通常见于区间合并 重复等题中）

   > ```
   > list = [[1,4],[3,6],[2,8]]
   > # 表示 对list排序 按照第一个元素升序排列 第一个元素相同 则按照第二个元素降序排列 
   > list.sort(key=lambda x : (x[0], -x[1]))
   > or
   > new = sorted(list, key=lambda x: x[0])
   > ```

10. 调用class的方法: treenode的定义 

    ```
    #coding=utf-8
    import sys 
    #str = input()
    #print(str)
    class treenode:
        def __init__(self, x):
            self.left = None 
            self.right = None
            self.val = x
    
    class solution:
        def __init__(self, x):
            self.res = x
        
        def max_path_sum(self, root):
            if not root:
                return 0
    
            left_sum = max(self.max_path_sum(root.left),0)
            right_sum = max(self.max_path_sum(root.right),0)
    
            self.res = max(self.res, root.val+left_sum+right_sum)
    
            return root.val + max(left_sum, right_sum)
    
    if __name__ == "__main__":
        root = treenode(2)
        root.left = treenode(-4)
        root.right = treenode(-1)
        root.left.left = treenode(0)
        root.left.right = treenode(1)
        
        s = solution(float("-inf"))
    
        s.max_path_sum(root)
        print(s.res)
    ```

    

11. 撒旦

