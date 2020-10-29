## 总结

- 回溯法有着固定的套路  首先是终止条件（如何限制树有尽头 不会一直深度搜索下去） 然后选择在变量可选的范围内 进行遍历（进入不同的兄弟节点） ，剪枝--在新的节点下依据题目要求判断是否符合要求（终止或者跳入下一循环），再深度搜索，递归调用自身（进入子树继续）

```
    def recuirsion(..., ...):
        if 终止条件判断：
            加入最终结果（或者在判断之前加入，见子集）
            符合则return
        
        for i in tmp_list:(遍历兄弟节点 遍历什么集？)
            剪枝处理<可选>  break 或者continue
            重复处理<可选>  （往往是跳过兄弟节点 continue）
            处理要进入子树的数据，调用自身进入子树 recursion(..., ...)
            ####### 十分重要的一点： 往往进入一个子树 前加一部分s 在进入其他子树时 要将这部分数据移除 （即进入不同子树的情况）      在上述很多题中出现的list直接可以 tmp+[nums[i]] 不再 需要这一步
            #######  正常情况下: tmp_res += tmp  #加一个元素 进入一个子树
                                recursion(tmp_res) #继续回溯
                                tmp_res = tmp_res[:-1]  # 不同子树遍历 要移除刚才加的元素
                     或者直接 recursion（tmp_res+tmp）  #则不再需要考虑 开始前增加 结束后移除的问题 同时加法是新建list
```

回溯法 == 深度优先搜索 + 状态重置（进去子树前修改 出来之后恢复子树 进行下一个选择） + 剪枝 
对于挑选子集，即在回溯中每次选择的数，可以使用used数组，可以看60题 的修正解法

迷宫问题： 二维平面的搜索
    - 使用偏移量数组  #遍历二维平面的四个方向 directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        - 还是DFS 注意这时候返回和原先的区别 要不断返回True 


可以查看下面的链接 内有模板
我认为“回溯搜索” = “深度优先遍历 + 状态重置 + 剪枝”。

1、“深度优先遍历” 就是不撞南墙不回头；

2、回头的时候要“状态重置”，即回到上一次来到的那个地方，“状态”要和上一次来的时候一样。

3、在代码上，往往是在执行下一层递归的前后，代码的形式是“对称的”。

作者：liweiwei1419
链接：https://leetcode-cn.com/problems/permutations/solution/hui-su-suan-fa-python-dai-ma-java-dai-ma-by-liweiw/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```
#回溯的一般结构：
void DFS(int 当前状态)  
{  
      if(当前状态为边界状态)  
      {  
        记录或输出  
        return;  
      }  
      for(i=0;i<n;i++)       //横向遍历解答树所有子节点  
      {  
           //扩展出一个子状态。  
           修改了全局变量  
           if(子状态满足约束条件)  
            {                                                                             					dfs(子状态)  
           }  
            恢复全局变量//回溯部分  
      }  
}  
```

递归法的解题思路

```
因此，也就有了我们解递归题的三部曲：

找整个递归的终止条件：递归应该在什么时候结束？

找返回值：应该给上一级返回什么信息？

本级递归应该做什么：在这一级递归中，应该完成什么任务？

------------------------------------------------------
广度优先搜索BFS- 参看二叉树102题
```

- 关于重复的剪枝，先排序，进入兄弟节点的时候进行剪枝处理，见47总结
- 在子集和子集2中有自己的想法trick 也可以参考力扣自己题解
- 如何分析回溯法的时间复杂度？
- 递归 回溯 以及DFS之间的区别与关联 


关于python传值的测试 见自己的有道云笔记

- 传切片 传tmp+ [num]  
- append和remove之后 传tmp 是否相同  
- 传普通的list
- 目前认为是 如果在传 append之后 要[:] 复制  如果是传【】+【】 不需要【：】复制





### 回溯算法（DFS）思路

**多叉树的遍历问题**

```
result = []
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.add(路径)
        return

    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择
```





### BFS思路

**bfs核心：问题的本质就是让你在一幅「图」中找到从起点** **`start`** **到终点** **`target`** **的最近距离**

几种常见的变种问题形式：

1. 走迷宫问题，部分格子有围墙，问最短距离 
2. 单词替换问题 通过某些替换，其中一个单词变为另外一个，每次只替换一个字符，最少几次替换？
3. 连连看游戏的方块消除 



**BFS的基本框架：**

注意python中有关队列

```
# 引入deque
from collections import deque

#定义及初始化
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

```



```
// 计算从起点 start 到终点 target 的最近距离
int BFS(Node start, Node target) {
	# 申请分配资源
    Queue<Node> q; // 核心数据结构
    Set<Node> visited; // 避免走回头路
	
	# 初始化状态
    q.offer(start); // 将起点加入队列
    visited.add(start);
    int step = 0; // 记录扩散的步数

    while (q not empty) {
        int sz = q.size();
        /* 将当前队列中的所有节点向四周扩散 */
        for (int i = 0; i < sz; i++) {
            Node cur = q.poll();
            /* 划重点：这里判断是否到达终点 */
            if (cur is target)
                return step;
            /* 将 cur 的相邻节点加入队列 */
            for (Node x : cur.adj())
                if (x not in visited) {
                    q.offer(x);
                    visited.add(x);
                }
        }
        /* 划重点：更新步数在这里 */
        step++;
    }
}
```



例子：

#### [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

#### [752. 打开转盘锁](https://leetcode-cn.com/problems/open-the-lock/)



经典例子-岛屿数量 

#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)  

答案复制不进来，直接参考leetcode解法 

```python
from collections import deque

class Solution:
    def __init__(self):
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

    def numIslands(self, grid: List[List[str]]) -> int:
        # 常见的4个方向 上右下左的次序 
        # directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
         if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])

        marked = [[False for _ in range(n)] for _ in range(m)]

        res = self.dfs_solution(grid, marked, m, n)
        #print(res)
        return res 
        
    def bfs_solution(self, grid, marked, m, n):
        res = 0 
        for i in range(m):
            for j in range(n):
                if not marked[i][j] and grid[i][j] == '1' :
                    res += 1
                    queue = deque()

                    marked[i][j] = True
…        
        return res 
        

```

