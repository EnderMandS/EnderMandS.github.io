---
layout: post
title: "Algorithm Learning Linear structure"
date: 2025-10-15 17:07:00 +0800
categories: [Algorithm]
---

# 数组与字符串

## 双指针

双指针（Two Pointers）是一种常用的算法技巧，适用于数组、字符串、链表等数据结构的高效遍历和操作。

通过使用两个指针（可以是索引或引用），实现更优的时间复杂度，通常从 O(n2)O(n^2)O(n2) 优化到 O(n)O(n)O(n)。

双指针的优势:
- 高效性：通过减少嵌套循环，优化时间复杂度。
- 灵活性：适用于多种数据结构和问题类型。
- 易实现：逻辑清晰，代码简洁。

### 同向双指针（快慢指针）

两个指针从同一位置出发，一个移动较快，一个移动较慢。

应用场景：

- 链表中检测环（如 Floyd 判圈算法）。
- 删除链表中的重复元素。
- 滑动窗口问题（如求子数组的最大和）。

### 相向双指针

两个指针分别从序列的两端向中间移动。
应用场景：

- 判断数组是否是回文。
- 求两数之和（如 LeetCode 经典题目 Two Sum）。
- 盛水最多的容器问题。


### 固定与滑动指针

一个指针固定，另一个指针滑动。

应用场景：

- 滑动窗口问题（如求满足条件的子数组长度）。
- 字符串匹配问题。


### 经典例题

#### 两数之和（相向双指针）

问题：在一个有序数组中，找到两个数，使它们的和等于目标值。

思路：使用相向双指针，一个从头开始，一个从尾开始，根据和的大小调整指针位置。

``` Python
def two_sum(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []
```

#### 链表检测环（快慢指针）

问题：判断链表中是否存在环。

思路：快指针每次移动两步，慢指针每次移动一步。如果快慢指针相遇，则存在环。

``` Python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

#### 盛水最多的容器（相向双指针）

问题：给定一个数组，表示容器的高度，求能盛水的最大面积。

思路：使用相向双指针，计算面积并调整较短的指针。

``` Python
def max_area(height):
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        max_area = max(max_area, min(height[left], height[right]) * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_area
```

## 滑动窗口

滑动窗口算法是一种通过维护一个可动的区间来遍历线性序列，并在窗口内进行高效计算的技巧，能够将暴力枚举的O(n²)优化为O(n)的线性复杂度。

概述: 滑动窗口主要用于处理数组或字符串中的连续子区间问题。核心思路是用左右两个指针界定窗口边界，右指针不断向右扩展窗口范围，当窗口不满足条件时，左指针向右移动以收缩窗口，再次判断并更新结果。

原理: 给定长度为N的序列S，定义两个指针left和right表示当前窗口的左右边界，初始均指向0。 每次将right向右移动，将新元素加入窗口计算；当窗口内状态达到或超过所需条件时，移动left以试图收缩窗口并更新最终答案。

### 典型应用场景
- 求长度为 k 的子数组最大和或平均值

- 查找字符串中无重复字符的最长子串

- 寻找最短覆盖子串，如最小窗口子串问题

- 统计流数据中满足特定条件的滑动区间

### 基本步骤

1. 初始化 left=0, right=0，并根据题目需求初始化窗口内累计状态（如sum、map等）。

2. 在主循环中，将 right 向右移动，每次将新元素的影响累加到窗口状态。

3. 检查当前窗口是否满足题目条件：
    - 若不满足，继续扩大 right；
    - 若满足，则进入收缩阶段——移动 left，更新最优解，并从窗口状态中删除 left 指向的元素影响，直到窗口不再满足条件为止。

4. 重复步骤2–3，直至 right 扫描到序列末尾

### 代码示例

#### 求连续3个元素的最大和
``` python
# 示例：求连续3个元素的最大和 (固定窗口大小)
def fixed_sliding_window(arr, k):
    if len(arr) < k:
        return None
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        if window_sum > max_sum:
            max_sum = window_sum
    return max_sum

arr = [1, 4, 2, 10, 2, 3, 1, 0, 20]
print(fixed_sliding_window(arr, 3)) 
# 输出 15
```

#### 无重复字符的最长子串

问题：给定一个字符串 s，找出其中无重复字符的最长子串的长度。 

思路：滑动窗口＋哈希表维护字符最近出现位置，右指针扩张窗口，遇到重复时左指针跳过历史索引，从而保证窗口内无重复。

``` Python
def lengthOfLongestSubstring(s):
    char_index = {}
    left = 0
    max_len = 0
    for right, c in enumerate(s):
        if c in char_index and char_index[c] >= left:
            left = char_index[c] + 1
        char_index[c] = right
        max_len = max(max_len, right - left + 1)
    return max_len
```

#### 包含字符的最小子串
问题：给定字符串 s 和 t，在 s 中找出包含 t 所有字符的最小子串。 

思路：动态滑动窗口＋双指针，用两个哈希表 need 和 window 分别记录所需与当前窗口字符频次，维护已满足种类数 valid，当 valid == len(need) 时尝试收缩窗口并更新答案。

``` Python
from collections import Counter, defaultdict

def minWindow(s, t):
    if not s or not t:
        return ""
    need = Counter(t)
    window = defaultdict(int)
    left = right = 0
    valid = 0
    start, length = 0, float('inf')
    
    while right < len(s):
        c = s[right]
        right += 1
        if c in need:
            window[c] += 1
            if window[c] == need[c]:
                valid += 1
        
        while valid == len(need):
            # 更新最小覆盖子串
            if right - left < length:
                start, length = left, right - left
            d = s[left]
            left += 1
            if d in need:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1
    
    return "" if length == float('inf') else s[start:start+length]
```

#### 找不重复三元组

问题：给定一个整型数组 nums，找出所有和为 0 的三元组，且不包含重复三元组。 

思路：先对数组排序，固定第一个指针后在剩余区间用左右指针寻找两数和为目标，双指针同时去重。整体复杂度 O(n²)。

``` Python
def threeSum(nums):
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n - 2):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return res
```

## 快慢指针

快慢指针是一种在链表或数组等线性结构中同时维护两条不同速度指针的技巧，常用于环检测、找中点、寻找倒数第 k 个节点以及数组中重复数等问题。它通过「快指针每次走两步、慢指针每次走一步」的方式，将一些原本 O(n²) 的暴力解法降为 O(n) 时间复杂度

快慢指针（Floyd’s Tortoise and Hare）在遍历过程中，慢指针每次向前移动一步，快指针每次向前移动两步。如果结构中存在环，快指针会在环内追上慢指针；如果不存在环，快指针将率先到达终点（如链表的 null 或数组边界）。这两条指针速度差异带来的相遇性质是该算法的核心

常见应用场景
- 判断链表是否有环（LeetCode 141）
- 找到环形链表的入口节点（LeetCode 142）
- 查找链表的中点（LeetCode 876）
- 返回链表倒数第 k 个节点（剑指 Offer 22）
- 数组中找到重复数字（LeetCode 287）

### 代码示例

#### 判断链表是否有环

思路：慢指针每次走 1 步，快指针每次走 2 步，若两者相遇则存在环，否则不存在。

``` python
def hasCycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

#### 环入口

思路：先检测环的存在并相遇，然后将一指针移回头节点，以同速同行，相遇点即为环的入口。

``` python
def detectCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    if not fast or not fast.next:
        return None
    fast = head
    while fast != slow:
        fast = fast.next
        slow = slow.next
    return fast
```

#### 链表中点

思路：当快指针走到末尾，慢指针正好走到中点；若链表长度为偶数，可根据需求选取第二中点。

``` python
def middleNode(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

#### 数组中找到重复数字

思路：将数组视作「环形链表」，索引和数值映射到节点与指针。通过典型的快慢指针相遇与重定位操作，得到重复值。

``` python
def findDuplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    fast = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow
```

# 链表

## 单链表和双链表基础操作

以下内容分别介绍单链表与双链表的增删改查，以及在链表中使用快慢指针找中点和检测两链表相交的方法。每段均附简要思路与 Python 示例代码。

### 单链表增删改查

单链表节点只含一个指针域，指向下一节点。

#### 结构定义  
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

#### 插入节点  
- 头插法：新节点指向原头，更新头指针  
- 尾插法：遍历到尾节点，将其 `next` 指向新节点  
- 中间插入：找到插入位置前驱，调整指针  
```python
def insert_after(prev_node, val):
    if not prev_node:
        return
    new_node = ListNode(val)
    new_node.next = prev_node.next
    prev_node.next = new_node
```

#### 删除节点  
给定删除值或索引，遍历定位前驱节点，`prev.next = prev.next.next`  
```python
def delete_node(head, val):
    dummy = ListNode(0, head)
    prev = dummy
    while prev.next and prev.next.val != val:
        prev = prev.next
    if prev.next:
        prev.next = prev.next.next
    return dummy.next
```

#### 查找与修改  
线性遍历，遇到目标即返回或更新  
```python
def find_node(head, val):
    curr = head
    while curr:
        if curr.val == val:
            return curr
        curr = curr.next
    return None

def update_node(head, old_val, new_val):
    node = find_node(head, old_val)
    if node:
        node.val = new_val
```

### 双链表增删改查

双链表每个节点含 `prev` 和 `next` 两个指针，可双向遍历。

#### 结构定义  
```python
class DListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
```

#### 插入节点  
在节点 `p` 之后插入新节点 `s`  
```python
def insert_after(p, val):
    s = DListNode(val)
    s.next = p.next
    if p.next:
        p.next.prev = s
    p.next = s
    s.prev = p
```
在头部或尾部插入同理，只需调整相应边界指针

####  删除节点  
直接断开目标节点 `x` 前后指针  
```python
def delete_node(x):
    if not x:
        return
    if x.prev:
        x.prev.next = x.next
    if x.next:
        x.next.prev = x.prev
```

#### 查找与修改  

正向或反向遍历，定位节点后读取或更新 `val`

## 快慢指针找中点

在单链表中，快指针每次走两步、慢指针每次走一步，快指针到尾时慢指针即在中点。

```python
def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

- 如果节点数为偶数，返回的是第 2 个中点  
- 时间复杂度 O(n)，空间 O(1)

## 链表相交检测

判断两单链表是否相交并返回第一个公共节点。

### 方法一：长度差对齐

1. 分别遍历两表求长度 `lenA`、`lenB`  
2. 长表先走 `|lenA–lenB|` 步  
3. 两指针同时向后，首次相等即为相交节点

```python
def getIntersectionNode(headA, headB):
    def length(head):
        l = 0
        curr = head
        while curr:
            l += 1
            curr = curr.next
        return l

    lenA, lenB = length(headA), length(headB)
    a, b = headA, headB
    if lenA > lenB:
        for _ in range(lenA - lenB):
            a = a.next
    else:
        for _ in range(lenB - lenA):
            b = b.next

    while a and b:
        if a == b:
            return a
        a = a.next
        b = b.next
    return None
```

### 方法二：双指针切换

1. 指针 `p`、`q` 分别从 `headA`、`headB` 出发  
2. 到尾后跳到另一表头继续  
3. 当 `p==q` 时即为相交或都到 `None` 结束

```python
def getIntersectionNode(headA, headB):
    if not headA or not headB:
        return None
    p, q = headA, headB
    while p is not q:
        p = p.next if p else headB
        q = q.next if q else headA
    return p
```

