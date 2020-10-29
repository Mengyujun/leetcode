### 6. Z 字形变换
    将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。
    
    比如输入字符串为 "LEETCODEISHIRING" 行数为 3 时，排列如下：
    
    L   C   I   R
    E T O E S I I G
    E   D   H   N
    之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："LCIRETOESIIGEDHN"。
    
    请你实现这个将字符串进行指定行数变换的函数：
    
    string convert(string s, int numRows);
    示例 1:
    
    输入: s = "LEETCODEISHIRING", numRows = 3
    输出: "LCIRETOESIIGEDHN"
    示例 2:
    
    输入: s = "LEETCODEISHIRING", numRows = 4
    输出: "LDREOEIIECIHNTSG"
    解释:
    
    L     D     R
    E   O E   I I
    E C   I H   N
    T     S     G

链接：https://leetcode-cn.com/problems/zigzag-conversion

```
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        n = len(s)
        out = ""
        shang = n//(2*numRows - 2)
        yushu = n%(2*numRows - 2)
        
        for i in range(numRows):
            for j in range(shang):
                if i == 0:
                    out += s[i+j*(2*numRows - 2)]
                else:
                    out += s[i+j*(2*numRows - 2)] 
                    out += s[i+2*(numRows-i-1)+j*(2*numRows - 2)]
            if yushu <= numRows and i < yushu:
                out += s[i+(j+1)*(2*numRows - 2)]
            elif yushu > numRows:
                out += s[i+(j+1)*(2*numRows - 2)] 
                out += s[i+2*(numRows-i-1)+(j+1)*(2*numRows - 2)]
        return out

        #自己一开始的思路 企图通过数学上找规律 余数最后没有成功 划分的太复杂了 要考虑余数什么的
```
    查看别人解释 自己相差不多 自己考虑的复杂了 
    
    规律：
    每一个Z字的首字母差，numRows*2-2 位置
    除去首尾两行，每个 Z 字有两个字母，索引号关系为，一个为 i，另一个为 numsRows*2-2-i


    先按照numRows*2-2 划分出一个个的小数组，再按照规律以及首尾特殊性来区分，拼接时比自己的（余数，商简单很多）

```
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if not s :
            return ""
        if numRows == 1:
            return s 
        spilt_s_len = numRows * 2 - 2
        n = len(s)
        data = []
        for i in range(0, n, spilt_s_len):
            data.append(s[i:i+spilt_s_len])
        #print(data)
        res = ""
        for i in range(numRows):
            for tmp in data:
                #保证可以取到tmp[i] 下面的同理 
                if i < len(tmp):
                    #首尾的情况
                    if i == 0 or i  == numRows - 1:
                        res += tmp[i]
                    else:
                        res += tmp[i]
                        if spilt_s_len - i < len(tmp):
                            res += tmp[spilt_s_len - i]
        return res
```

### 12. 整数转罗马数字
    罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。
    
    字符          数值
    I             1
    V             5
    X             10
    L             50
    C             100
    D             500
    M             1000
    例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。
    
    通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：
    
    示例 5:
    输入: 1994
    输出: "MCMXCIV"
    解释: M = 1000, CM = 900, XC = 90, IV = 4.


​    
​    自己思路：先划分为不同位数，1000 100 10 1 不同位数里面先看也有没有特殊情况，再去分情况讨论判断（除去那6中特殊情况 再分别考虑5 50 500）
```
class Solution:
    def intToRoman(self, num: int) -> str:
        dic = {4: "IV", 9: "IX", 40: "XL", 90: "XC", 400: "CD", 900: "CM"}
        res = ""
        if num == 0:
            return ""
        if num in dic:
            return dic[num]
        Roman = [1000, 100, 10, 1]
        result= [0] * 4
        for i in range(len(Roman)):
            result[i] = num // Roman[i]
            num = num % Roman[i]
        for i in range(len(result)):
            #res += self.intToRoman(result[i] * Roman[i])
            tmp = result[i] * Roman[i]
            if tmp in dic:
                res += dic[tmp] 
            else:
                if tmp >= 1000:
                    tmp_res = tmp//1000
                    res += "M"*tmp_res
                elif tmp >= 500 :
                    tmp_res = (tmp-500)//100
                    res += "D"
                    res += "C"*tmp_res
                elif tmp >= 100:
                    tmp_res = tmp//100
                    res += "C"*tmp_res
                elif tmp >= 50 :
                    tmp_res = (tmp-50)//10
                    res += "L"
                    res += "X"*tmp_res
                elif tmp >= 10:
                    tmp_res = tmp//10
                    res += "X"*tmp_res
                elif tmp >= 5 :
                    tmp_res = (tmp-5)//1
                    res += "V"
                    res += "I"*tmp_res
                elif tmp >= 1:
                    res += "I"*tmp 
        return res    
```

    更好的一种解法： 主要是贪心思想 优先减去较大的数（尽量用较大的数来表示） 并把1000 和 900 当做相同待遇 不需要像自己一样区分开
    注意： 下面解法中 将dict逆序的操作 值得学习
```
class Solution:
    def intToRoman(self, num: int) -> str:
        dic = {
            1: "I",
            4: "IV",
            5: "V",
            9: "IX",
            10: "X",
            40: "XL",
            50: "L",
            90: "XC",
            100: "C",
            400: "CD",
            500: "D",
            900: "CM",
            1000: "M"}
        res = ""
        # 需要逆序遍历dict 贪心的思想 优先减去大的  直接通过商来判断
        for key in sorted(dic.keys())[::-1]:
            tmp_res = num//key
            if tmp_res > 0 :
                num -= tmp_res * key
                res += dic[key] * tmp_res
        return res
```

### 13. 罗马数字转整数
    题目内容和12题大致类似 
    
    采用和12题相同的模式  还是用相同的贪心算法 优先选择大的进行比较
    注意： 1 使用不同的dic 按照value值逆序 2 判断连续字母时while循环
```
class Solution:
    def romanToInt(self, s: str) -> int:
        dic = {
            "I": 1,
            "IV": 4,
            "V": 5,
            "IX": 9,
            "X": 10,
            "XL": 40,
            "L": 50,
            "XC": 90,
            "C": 100,
            "CD": 400,
            "D": 500,
            "CM": 900,
            "M": 1000
        }
        sort_dic = dict(sorted(dic.items(), key = lambda x:x[1], reverse=True))
        #print(sort_dic)
        res = 0
        cur_index = 0
        #这种写法正确吗？
        for i in sort_dic.keys():
            #print(i)
            #需要增加判断有几个 相同的字母 
            while s[cur_index : cur_index+ len(i)] == i:
                res += sort_dic[i]
                cur_index += len(i)
        
        return res
```

### 15. 三数之和 

    详见 双指针法.md
    
    几种错误情况没有考虑到（边界条件）：
    1. 输入为空 
    2. 输入个数不足
    3. 输入全为0  （背后根源- 判重的问题）
    4. 判重问题 tmp = 0 时 下一步的操作出现问题 break？ 还是继续 左右同时缩减
    5. [-2,0,0,2,2]  左右同时缩减 还是会出现重复元素  -> 重复时要 再判断是否重复 有重复则跳过去


### 24. 两两交换链表中的节点
    给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
    
    你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。


​     

    示例:
    
    给定 1->2->3->4, 你应该返回 2->1->4->3.

自己的解法  非递归解法 即直接两两互换 挪动指针往后移动
```
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        #如何考虑边界条件
        
        dummy = ListNode(0)
        dummy.next = head
        tmp_dummy = dummy
        
        while(tmp_dummy.next!= None and tmp_dummy.next.next != None):
            first = tmp_dummy.next
            second = tmp_dummy.next.next
        
            #实际的交换操作
            tmp = second.next
            tmp_dummy.next = second 
            second.next = first 
            first.next = tmp
            
            #后续的更新指针
            tmp_dummy = first
        
        return dummy.next
```

递归法的思路  只用处理前两个元素 后面的元素直接递归调用自身
- 找整个递归的终止条件：递归应该在什么时候结束？
- 找返回值：应该给上一级返回什么信息？
- 本级递归应该做什么：在这一级递归中，应该完成什么任务？

```
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        #递归的终止条件 只剩下一个节点或者没有节点了
        if not head or not head.next:
            return head
        
        #本层次递归做的操作 主要是 分为 head next 剩余递归处理部分 交换前两部分
        tmp = head.next
        head.next = self.swapPairs(head.next.next)
        tmp.next = head

        #递归返回的内容是什么 返回给上一级的是 完成交换之后，即已经处理好的链表部分的首指针 
        return tmp

```

### 31下一个排列
    实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
    
    如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
    
    必须原地修改，只允许使用额外常数空间。
    
    以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
    1,2,3 → 1,3,2
    3,2,1 → 1,2,3
    1,1,5 → 1,5,1

自己有问题的解答
```
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        flag = 0
        for i in range(1,n)[::-1]:
            #print(i)
            if nums[i] > nums[i-1]:
                #进行原地交换操作 交换之后可以跳出
                tmp = nums[i-1]
                nums[i-1] = nums[i]
                nums[i] = tmp
                flag = 1
                break
        #出现没有最大的情况 需要逆序
        if not flag:
            for i in range(n//2):
                #交换 nums[i] 和nums[n-1-i]                
                tmp = nums[n-1-i]
                nums[n-1-i] = nums[i]
                nums[i] = tmp
        
        
```
```
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        flag = 0
        def reverse(nums,i,j):
            while(i<j):
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1 
        # 注意有两个for循环 都需要跳出 两个break
        for i in range(1,n)[::-1]:
            #print(i)
            if nums[i] > nums[i-1]:
                #进行原地交换操作 交换之后可以跳出
                # 增加比原来多的操作 在i后找一个元素刚刚大于i-1 交换 然后逆序
                for j in range(i,n)[::-1]:
                    if nums[j] > nums[i-1]:
                        nums[j] ,nums[i-1] = nums[i-1], nums[j]
                        flag = 1
                        break
                #print(nums)
                #对之后i的元素进行逆序 目前是降序-》升序
                #nums[i:].reverse() 
                reverse(nums, i, n-1)
                break
        #出现没有最大的情况 需要逆序
        if not flag:
            for i in range(n//2):
                #交换 nums[i] 和nums[n-1-i]                
                nums[i], nums[n-1-i] = nums[n-1-i], nums[i]   
```



### 36. 有效的数独

    判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。
    
    数字 1-9 在每一行只能出现一次。
    数字 1-9 在每一列只能出现一次。
    数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
    
    自己原始解法即 循环遍历整个数组3次，每次调用函数来 进行 重复的判断

```
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        #需要初始化 这么多个dic  用map来保存
        row = []
        col = []
        box = []    
        for index in range(len(board)):
            dic = {}
            row.append(dic)
            col.append(dic)
            box.append(dic)      
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != '.':
                    tmp = board[i][j]
                    if tmp not in row[i].keys():
                        row[i][tmp]  = 1
                    else:
                        return False
                    
                    if tmp not in col[j].keys():
                        col[j][tmp] = 1
                    else:
                        return False
                    
                    box_index = (i//3)*3 +  (j//3)
                    if tmp not in box[box_index].keys():
                        box[box_index][tmp] = 1
                    else:
                        return False
        
        return True
```

更新解法： 一次遍历 同时注意下代码中python的简洁写法
```
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        #需要初始化 这么多个dic  用map来保存
        row = [{} for i in range(9)]
        col = [{} for i in range(9)]
        box = [{} for i in range(9)]
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != '.':
                    tmp = board[i][j]
                    box_index = (i//3)*3 +  (j//3)
                    row[i][tmp] = row[i].get(tmp, 0) + 1
                    col[j][tmp] = col[j].get(tmp, 0) + 1
                    box[box_index][tmp] = box[box_index].get(tmp, 0) + 1                     
               
                    if row[i][tmp] > 1 or col[j][tmp] > 1 or box[box_index][tmp] > 1:
                        return False
        
        return True
```

### 43. 字符串相乘

    给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
    
    示例 1:
    
    输入: num1 = "2", num2 = "3"
    输出: "6"
    示例 2:
    
    输入: num1 = "123", num2 = "456"
    输出: "56088"

仿照415 字符串相加方式 逆序 相乘  进位 取余 不断更新
```
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"
        num1 = num1[::-1]
        num2 = num2[::-1]
        m = len(num1)
        n = len(num2)
        res = [0] * (m+n)
        for i in range(m):
            for j in range(n):
                tmp = res[i+j] + int(num1[i]) * int(num2[j])
                res[i+j+1] += tmp // 10
                res[i+j] = tmp%10
        
        #print(res)
        res= res[::-1]
        #print(res)
        #去掉首部的0 
        while res and res[0] == 0:
            del res[0]
        res = [str(i) for i in res]
        #print(res)
        return ''.join(res)
```

### 50. Pow(x, n)
    实现 pow(x, n) ，即计算 x 的 n 次幂函数。
    
    示例 1:
    
    输入: 2.00000, 10
    输出: 1024.00000

```
class Solution:
    def myPow(self, x: float, n: int) -> float:
        abs_n = abs(n) if n < 0 else n
        res = 1.0
        cur_cir = 0 
        k = 1
        while cur_cir < abs_n :
            #print(k, cur_cir , res)
            if cur_cir+ k*2 > abs_n and cur_cir + k >abs_n:
                k = 1 
                continue
            if k ==1:
                tmp_res = x 
                res = res*tmp_res
                cur_cir += k
                k *= 2
            else:
                tmp_res = tmp_res * tmp_res
                res = res * tmp_res
                cur_cir += k
                k *= 2
                        
        
        #print(res)
        if n > 0:
            return res
        else:
            return 1.0/res
        
        # 第一次出现 超时 当n很大时 每次不断增加1太慢  考虑用过的倍增除数法 或者log级别  比较麻烦 自己没有写出来  写出来了 就是不断的倍增因子  同时不断更新当前的已经的倍数
        #问题的本质实际上时减治法  每次缩减一半的情况  
    
    # 也可以使用递归的方式 按照奇偶不断的调用自身
    class Solution:
    def myPow(self, x: float, n: int) -> float:
        #先全部转化为正数情况  用递归
        if n == 0:
            return 1.0
        if n < 0 :
            x = 1/x
            n = -n
        #分为奇数和偶数的情况 
        if n ==  1:
            return x
        if n%2 == 0:
            tmp = self.myPow(x,n//2) 
            return tmp*tmp
        else:
            tmp = self.myPow(x,n//2)
            return tmp*tmp *x
        
```

### 54. 螺旋矩阵
    给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。
    
    示例 1:
    
    输入:
    [
    [ 1, 2, 3 ],
    [ 4, 5, 6 ],
    [ 7, 8, 9 ]
    ]
    输出: [1,2,3,6,9,8,7,4,5]
    
    递归的思想 输出外面的一层 然后不断调用函数本身来递归  使用递归的3步来思考（何时终止-终止条件， 本次递归内部操作， 每次递归返回什么）
```
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m = len(matrix)
        n = len(matrix[0]) if m > 0 else 0
        if m == 0:
            return []
        if m == 1:
            return matrix[0]
        if n == 1:
            res = [matrix[i][0] for i in range(m)]
            return res
                
        res = []
        
        #何时终止？  m n 为1 或者2 ？ 待定
        tmp_res = []
        #在这层递归要进行的操作 循环输出最外层
        if m > 0:
            for i in matrix[0]:
                tmp_res.append(i)
            #if m > 1: m = 2 是可以的
            for j in range(1, m-1):
                tmp_res.append(matrix[j][n-1])
            for i in matrix[m-1][::-1]:
                tmp_res.append(i)
            for j in range(1, m-1)[::-1]:
                tmp_res.append(matrix[j][0])
        
        #print(tmp_res)
        next_matrix = []
        if m > 2 and n > 2:
            tmp = matrix[1:m-1]
            for i in tmp:
                i = i[1:n-1]
                next_matrix.append(i)
        #print(next_matrix)
        res = tmp_res+ self.spiralOrder(next_matrix)
        #print(res)
        return res
```


### 55. 跳跃游戏

    给定一个非负整数数组，你最初位于数组的第一个位置。
    
    数组中的每个元素代表你在该位置可以跳跃的最大长度。
    
    判断你是否能够到达最后一个位置。
    
    示例 1:
    
    输入: [2,3,1,1,4]
    输出: true
    解释: 从位置 0 到 1 跳 1 步, 然后跳 3 步到达最后一个位置。

自己理解有点错误： 
- 可以左右跳吗？  还是只有一个方向  只有一个方向 注意指的是最大长度 因此可以少跳
- 先使用递归的方式 出现超出内存限制 改进之后直接改变lastposition 既不需要递归 也不需要去创建新的数组list 可以对比一下  实际上是贪心算法 
- 还可以使用 动态规划

```
#一开始想使用递归
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        #可以左右跳吗？  还是只有一个方向  只有一个方向 注意指的是最大长度 因此可以少跳
        if not nums:
            return False
        n = len(nums)
        #print(nums)
        #每次递归内部的操作
        for i in range(n-1)[::-1]:
            #print(i, nums[i])
            if nums[i] >= n-1-i:
                #出现可以直接跳过去的情况 
                if i == 0:
                    return True
                return self.canJump(nums[:i+1])
        return False
    # 出现了超出内存限制的问题 递归调用太多次
```
```
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        #可以左右跳吗？  还是只有一个方向  只有一个方向 注意指的是最大长度 因此可以少跳
        if not nums:
            return False
        n = len(nums)
        #print(nums)
        #每次递归内部的操作
        lastposition = n-1
        for i in range(n-1)[::-1]:
            #print(i, nums[i])
            if nums[i] >= lastposition-i:
                #出现可以直接跳过去的情况 
                lastposition = i
            
        return True if lastposition == 0 else False
```

### 89. 格雷编码
    输入: 2
    输出: [0,1,3,2]
    解释:
    00 - 0
    01 - 1
    11 - 3
    10 - 2
    
    对于给定的 n，其格雷编码序列并不唯一。
    例如，[0,2,3,1] 也是一个有效的格雷编码序列。
    
    00 - 0
    10 - 2
    11 - 3
    01 - 1

先试用递归的方式 有很多的重复子问题 考虑用动态规划
```
class Solution:
    def grayCode(self, n: int) -> List[int]:
        if n == 0:
            return [0]
        if n == 1:
            return [0, 1]
        
        tmp = self.grayCode(n-1)
        res = tmp
        for tmp_num in tmp[::-1]:
            res.append(tmp_num+2**(n-1))
        
        return res
```
使用动态规划的方式 一开始忘记写range 出现了错误
```
class Solution:
    def grayCode(self, n: int) -> List[int]:
        if n == 0:
            return [0]
        res = [0]
        for i in range(1, n+1):
            tmp = res
            for tmp_num in tmp[::-1]:
                res.append(tmp_num+2**(i-1))
        return res
```


### 91. 解码方法 
    一条包含字母 A-Z 的消息通过以下方式进行了编码：
    
    'A' -> 1
    'B' -> 2
    ...
    'Z' -> 26
    给定一个只包含数字的非空字符串，请计算解码方法的总数。
    
    示例 1:
    
    输入: "12"
    输出: 2
    解释: 它可以解码为 "AB"（1 2）或者 "L"（12）。
    
    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/decode-ways
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```
class Solution:
    def numDecodings(self, s: str) -> int:
        # 用递归的思路 还是先考虑用 递归 再尝试用循环 动态规划的思路来解决   用递归出现超时
        if not s:
            return 1
        lis = [int(i) for i in s]
        #print(lis)
        if len(lis) == 1:
            return 0 if lis[0] == 0 else 1
        if lis[0] != 0:
            s1 = s[1:]
            s2 = s[2:]
        else:
            return 0
        tmp = lis[0]*10 +lis[1]
        #print(tmp)
        if tmp <= 26:
            #print(s1, s2)
            #print(self.numDecodings(s1))
            #print(self.numDecodings(s2))
            return self.numDecodings(s1) + self.numDecodings(s2)
        else:
            return self.numDecodings(s1)
```
尝试将递归转化为 循环的方法    可以继续优化 即减小dp数组 只保存两个数即可
```
class Solution:
    def numDecodings(self, s: str) -> int:
        # 用递归的思路 还是先考虑用 递归 再尝试用循环 动态规划的思路来解决   用递归出现超时
        if not s:
            return 1
        lis = [int(i) for i in s[::-1]]
        if len(lis) == 1:
            return 0 if lis[0] == 0 else 1
        #print(lis)
        res = [0]* len(lis) 
        for i in range(len(lis)):
            #print(i) 
            if lis[i] == 0:
                res[i] = 0
            else:  
                if i == 0:
                    res[i] = 1 
                if i == 1:
                    tmp = lis[i]*10 + lis[i-1]
                    if tmp <= 26:
                        res[1] = res[0] + 1
                    else:
                        res[1] = 0 if lis[0] == 0 else 1
                if i > 1:
                    tmp = lis[i]*10 + lis[i-1]
                    #print(tmp)
                    if tmp <= 26:
                        res[i] = res[i-1] + res[i-2]
                    else:
                        res[i] = res[i-1]
        #print(res)
        return res[-1]
                
```

### 146. LRU缓存机制

出现了超时问题 主要原因还是  将队列中某一项移到末尾  要从头遍历到该节点需要O（n）的时间复杂度 -》改进方法是使用双向链表
```
class ListNode:
    def __init__(self, x, y):
        self.key = x
        self.val = y
        self.next = None
        
class LRUCache:

    def __init__(self, capacity: int):
        self.num = capacity
        self.cur = 0
        self.dummy = ListNode(0, -1)
        self.curnode = self.dummy
        self.dic = {}
    def get(self, key: int) -> int:
        #一个链表如何能直接得到O(1)  用dict保存？ 空间换时间？
        #print(self.dic)
        if key in self.dic.keys():
            #挪动这个元素到链表的末尾
            tmp = self.dummy.next
            last = self.dummy
            while tmp and tmp.key != key:
                tmp = tmp.next
                last = last.next
            if tmp.next != None:
                last.next = last.next.next
                tmp.next = None
                self.curnode.next = tmp
                self.curnode = self.curnode.next
            #print("get:", key)
            #print(self.dummy.next.key)
            return self.dic[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        #put是更新操作 如同get那样要更新  
        if key in self.dic.keys():
            tmp = self.dummy.next
            last = self.dummy
            while tmp and tmp.key != key:
                tmp = tmp.next
                last = last.next
            if tmp.next != None:
                last.next = last.next.next
                tmp.next = None
                self.curnode.next = tmp
                self.curnode = self.curnode.next
            self.dic[key] = value
        else:
            if self.cur+1 <= self.num:
                node = ListNode(key, value)
                self.cur += 1
                self.curnode.next = node
                self.curnode = self.curnode.next
                self.dic[key] = value

            #超出缓存大小 需要删除首元素 再插入新的节点 调用自身即可
            else:
                #print("put", key, value)
                #print(self.dummy.next.key)
                self.dic.pop(self.dummy.next.key)
                self.dummy.next = self.dummy.next.next
                if self.dummy.next == None:
                    self.curnode = self.dummy
                self.cur -= 1
                self.put(key, value)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
    #目前考虑用队列来实现 
    #在put时（暂时认为不会出现重复的元素？ 会有吗？ 有 出现了bug） 有空则插入 没空则出 队首元素，再入队  
    #put时 出现重复的元素要注意更新  更新的方式可以简单点 直接删除再加新的元素
    #在get时 没有元素直接return-1  若有这个元素 则应该提到队列的末尾（不好移动？ 用链表？）
    #dic 增加删除 注意位置 加时修改之后加 删除时 修改之前删除（否则已发生变化）
    #在测试用例容量为1 时出现错误 增加改变curnode 
    #在get 末尾元素时不应该 去删除 简化操作既可
    #内容正确之后出现超时操作  主要问题出现在了 while寻找链表的位置 就是找要 交换的位置用了 O(n)
```

修改为双指针版本
```
class ListNode:
    def __init__(self, key = None, val = None):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None
        
class LRUCache:

    def __init__(self, capacity: int):
        self.num = capacity
        self.cur = 0
        #新增两个节点 首尾各增加一个
        self.dummy = ListNode()
        self.tail  = ListNode()
        
        #初始化链表
        self.dummy.next = self.tail
        self.tail.prev  = self.dummy
        #self.curnode = self.dummy
        self.dic = {}
    
    #get和put方法 都需要用到将一个元素挪到末尾 定义出一个方法
    def move_node_to_tail(self, key):
        item = self.dic[key]
        item.prev.next = item.next
        item.next.prev = item.prev
        
        item.prev = self.tail.prev 
        item.next = self.tail
        self.tail.prev.next = item
        self.tail.prev = item
    
        
    def get(self, key: int) -> int:
        #一个链表如何能直接得到O(1)  用dict保存？ 空间换时间？
        #print(self.dic)
        if key in self.dic:
            self.move_node_to_tail(key)
            return self.dic[key].val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        #put是更新操作 如同get那样要更新
        if key in self.dic:
            self.dic[key].val = value
            self.move_node_to_tail(key)

        else:
            if len(self.dic) == self.num:
                #相等需要删除队首节点 再进行插入操作
                self.dic.pop(self.dummy.next.key)
                self.dummy.next = self.dummy.next.next
                self.dummy.next.prev = self.dummy
            
            #删除队首节点之后或者直接进行插入操作队尾
            node = ListNode(key, value)
            #此处就是关键 不是只存val值 而是将这个节点作为字典的value 从而保存下了前后指针的信息
            self.dic[key] = node
            #和自己单向链表的区别 因为 有尾指针tail的存在 所以可以直接插入 不需要遍历 也不需要curnode来保存信息（增加了信息 更新维护麻烦）
            #那么要注意增加尾节点 注意顺序(先写node的可交换顺序 后面的不能)
            node.next = self.tail
            node.prev = self.tail.prev
            self.tail.prev.next = node
            self.tail.prev = **node**
```

### 215. 数组中的第K个最大元素 top-k问题 面试的高频考题
    在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
    
    示例 1:
    
    输入: [3,2,1,5,6,4] 和 k = 2
    输出: 5
    示例 2:
    
    输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
    输出: 4
    
    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/kth-largest-element-in-an-array
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

典型的top-k问题，重点关注下，大根堆和小根堆  保持堆内元素个数不变，最小的元素在堆顶，每次选择比较堆顶元素（用的是优先队列的）
    具体思路： 建立只能存k个数字的小顶堆，其中最小元素在堆顶，遍历原数组，比堆顶元素（最小元素）大的应该放入堆内，则最终第k大的元素即为小顶堆的堆顶元素
    向大小为 k的数组中添加元素的时间复杂度为O(logk)
    另外的一个重点就是 partition的思想

用java的优先级队列，其中用到了lambda表达式  lambda 表达式：(a, b) -> a - b 表示最小堆 关于堆需要进一步了解
```
import java.util.PriorityQueue;

public class Solution {

    public int findKthLargest(int[] nums, int k) {
        int len = nums.length;
        // 使用一个含有 k 个元素的最小堆
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(k, (a, b) -> a - b);
        for (int i = 0; i < k; i++) {
            minHeap.add(nums[i]);
        }
        for (int i = k; i < len; i++) {
            // 看一眼，不拿出，因为有可能没有必要替换
            Integer topEle = minHeap.peek();
            // 只要当前遍历的元素比堆顶元素大，堆顶弹出，遍历的元素进去
            if (nums[i] > topEle) {
                minHeap.poll();
                minHeap.add(nums[i]);
            }
        }
        return minHeap.peek();
    }
}

作者：liweiwei1419
链接：https://leetcode-cn.com/problems/kth-largest-element-in-an-array/solution/partitionfen-er-zhi-zhi-you-xian-dui-lie-java-dai-/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



使用最大堆（最小堆），在python中使用优先级队列 heapq来实现 （heapq 见python数据结构）

```
from typing import List 
import heapq


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 注意小顶堆和大顶堆的区别 前者是最小元素在堆顶 后者是最大元素在堆顶
        # 注意是优先级队列 
        # 注意heapq 就是小顶堆 

        # 相当于维护一个 k长度的小顶堆 小顶堆-堆顶元素即为第k大的元素
        size = len(nums)
        
        L = []

        # 初始化放入k个元素
        for i in range(k):
            heapq.heappush(L, nums[i])

        # 开始遍历剩下的元素
        for index in range(k, size):
            top = L[0]
            # 此元素大于堆顶元素 说明是前k个元素 则应该加入到堆中
            if nums[index] > top:
                # 看一看堆顶的元素，只要比堆顶元素大，就替换堆顶元素
                heapq.heapreplace(L, nums[index])
        
        return L[0]

```



用经典的partition的思想来解决问题（快速排序和核心，partition即以某元素为基准，小于该元素的放在这前面，大于此元素的放在后面，后续再递归操作）

注意下不同快排思想  主要采用方法3  主要注意观察这个partition内部的区别 

1. 交换法partition
```
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        #尝试使用一下快排-partition的思想
        #将之逆序 用升序排序 则找的是n-k   注意此时的n-k即为index  可以直接比较
        return self.partition(nums, 0, len(nums)-1, len(nums)-k)
    
    def partition(self, nums, left, right, index):
        l, r = left, right
        x = nums[l]
        while l < r:
            while (l < r and nums[r] >= x):
                r -= 1
            nums[l], nums[r] = nums[r], nums[l]
            while (l < r and nums[l] <= x):
                l += 1
            nums[l], nums[r] = nums[r], nums[l]
        
        if l == index:
            return nums[l]
        if l < index:
            return self.partition(nums, l+1, right, index)
        else:
            return self.partition(nums, left, l-1, index)
```

2. 挖坑法快排
```
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        #尝试使用一下快排-partition的思想
        #将之逆序 用升序排序 则找的是n-k   注意此时的n-k即为index  可以直接比较
        return self.partition(nums, 0, len(nums)-1, len(nums)-k)
    
    def partition(self, nums, left, right, index):
        l, r = left, right
        x = nums[l]
        while l < r:
            while (l < r and nums[r] >= x):
                r -= 1
            if l < r:
                nums[l] = nums[r]
                l += 1
            while (l < r and nums[l] <= x):
                l += 1
            if l < r:
                nums[r] = nums[l]
                r -= 1
        
        nums[l] = x
        
        if l == index:
            return nums[l]
        if l < index:
            return self.partition(nums, l+1, right, index)
        else:
            return self.partition(nums, left, l-1, index)
```
3. 另外一种 主要是partition内部的细微区别 (这种更加简单 主要用这个 不需要加if判断和++) 还有就是外层循环 实质一样
```
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        #尝试使用一下快排-partition的思想
        #将之逆序 用升序排序 则找的是n-k   注意此时的n-k即为index  可以直接比较
        if not nums:
            return -1
        l, r = 0, len(nums)-1
        target = len(nums) - k
        while l < r:
            index = self.partition(nums, l , r)
            if index == target:
                return nums[index]
            elif index < target:
                l = index + 1
            else:
                r = index - 1
        return nums[target]
    
    def partition(self, nums, left, right):
        l, r = left, right
        x = nums[l]
        while l < r:
            while (l < r and nums[r] >= x):
                r -= 1
            nums[l] = nums[r]
            while (l < r and nums[l] <= x):
                l += 1
            nums[r] = nums[l]
        
        nums[l] = x
        return l
```


### 219. 存在重复元素 II
    给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，使得 nums [i] = nums [j]，并且 i 和 j 的差的绝对值最大为 k。
    
    示例 1:
    
    输入: nums = [1,2,3,1], k = 3
    输出: true
    示例 2:
    
    输入: nums = [1,0,1,1], k = 1
    输出: true
    
    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/contains-duplicate-ii
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

一开始的常规解法，出现了超时的问题
```
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        if k >= len(nums):
            k = len(nums)-1
        #注意题目中要求的是 最大为k，可以小于k
        for i in range(len(nums)-k):
            for j in range(i+1, i+k+1):
                if nums[i] == nums[j]:
                    return True
        #注意在最后的k个数中没有进行比较
        if len(nums[len(nums)-k:]) != len(set(nums[len(nums)-k:])):
            return True
        return False
        #最后还是会出现超时 最后一个测试用例数目很大

        #或者使用另外的一种遍历方式  主要是其中的min函数，来考察范围，而不是自己上面的那种
        #for i in range(len(nums)-1):
        #    for j in range(i+1, min(len(nums), i+k+1)):
        #        if nums[i] == nums[j]:
```

使用hash table 来维护k个滑动窗口   需要一个支持在常量时间内完成 搜索，删除，插入 操作的数据结构，那就是散列表(hash table) 注意一下
很好的方法，记忆学习，相当于i在前，用一个k长度的散列表来维护之前的k个元素，每次遍历新的元素，判断是否在散列表内
```
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        hs = set() 
        for i in range(len(nums)):
            if nums[i] in hs:
                return True
            else:
                hs.add(nums[i])
            if len(hs) > k:
                hs.remove(nums[i-k])
        return False
```





