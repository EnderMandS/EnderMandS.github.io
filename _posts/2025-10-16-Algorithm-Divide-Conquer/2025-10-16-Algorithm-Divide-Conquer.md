---
layout: post
title: "Algorithm Divide and Conquer"
date: 2025-10-16 09:49:00 +0800
categories: [Algorithm]
---

# 分治与回溯算法概述

分治（Divide and Conquer）将复杂问题递归拆分为多个相同或相似的子问题，分别求解后再合并结果

回溯（Backtracking）通过在解空间中进行深度优先搜索、逐步尝试并在不符合条件时撤销选择，来枚举所有可行解或找到满足约束的解。

掌握通用模板与剪枝策略，可高效解决子集、全排列、组合、子矩阵等枚举问题

## 分治算法通用三步

1. **划分**：将规模为 n 的问题拆分成若干个规模更小的子问题。  
2. **解决**：递归地解决每个子问题，直到子问题规模足够小时直接求解。  
3. **合并**：将子问题的解整合，得到原问题的解。  

分治算法的时间复杂度通常可由主定理（Master Theorem）分析。

## 快速幂（Exponentiation by Squaring）

快速幂用于在 O(log n) 时间内计算 \(a^n\)：

```python
def fast_pow(a, n):
    res = 1
    base = a
    while n > 0:
        if n & 1:            # 如果当前 n 为奇数，则乘上 base
            res *= base
        base *= base         # base 平方，n 右移
        n >>= 1
    return res
```

- 当 n 二进制最低位为 1 时，将当前基值 base 累乘到 res；  
- 不断将 base 平方，n 右移，循环 O(log n) 次完成计算。

---

## 归并排序（Merge Sort）

归并排序是分治思想的典型应用，时间复杂度 O(n log n)：

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    # 合并两个有序子数组
    res, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            res.append(left[i]); i += 1
        else:
            res.append(right[j]); j += 1
    # 追加剩余元素
    res.extend(left[i:]); res.extend(right[j:])
    return res
```

- 划分：将数组对半分为左右两部分；  
- 解决：递归地对两半排序；  
- 合并：线性扫描左右有序子数组，得到整体有序结果。

## 分治求解子问题示例：最大子数组和

使用分治可在 O(n log n) 时间计算数组的最大子段和：

```python
def max_subarray(arr):
    def helper(l, r):
        if l == r:
            return arr[l]
        m = (l + r) // 2
        left_max = helper(l, m)
        right_max = helper(m+1, r)
        # 跨中间的最大和
        left_suffix = right_prefix = float('-inf')
        s = 0
        for i in range(m, l-1, -1):
            s += arr[i]; left_suffix = max(left_suffix, s)
        s = 0
        for i in range(m+1, r+1):
            s += arr[i]; right_prefix = max(right_prefix, s)
        cross = left_suffix + right_prefix
        return max(left_max, right_max, cross)
    return helper(0, len(arr)-1)
```

- 分别计算左、右、跨中点的三种最大子段和，再取其最大值；  
- 递归深度 O(log n)，每层 O(n) 合并，整体 O(n log n)。

## 回溯算法核心思想

回溯在解空间树上进行深度优先搜索，对每个节点尝试所有可能选择，并在遇到不满足条件或已达目标后「撤销」上一步选择，继续探索其他分支。

### 回溯通用模板

```python
def backtrack(path, choices):
    if 达到结束条件:
        记录或处理 path
        return
    for choice in choices:
        path.append(choice)           # 1. 做选择
        backtrack(path, 更新 choices)  # 2. 递归探索
        path.pop()                    # 3. 撤销选择
```

- path 存当前构造的解
- choices 表示可选项
- 结束条件可由路径长度、累计值、矩阵边界等决定

### 回溯示例：子集生成

```python
def subsets(nums):
    res = []
    def dfs(index, path):
        res.append(path[:])
        for i in range(index, len(nums)):
            path.append(nums[i])
            dfs(i+1, path)
            path.pop()
    dfs(0, [])
    return res
```

- 每次可选或跳过当前元素，枚举所有 \(2^n\) 种子集；  
- 通过 `path` 保存当前解，通过 `pop()` 撤销选择，回溯至上一个状态。

### 子集枚举（Subsets）

```python
def subsets(nums):
    res = []
    def dfs(start, path):
        res.append(path[:])              # 每个 path 都是一个子集
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i+1, path)               # 只往后选，避免重复
            path.pop()
    dfs(0, [])
    return res
```

剪枝策略: 固定大小 k 时，可在 if len(path)>k: return 或限制 i<=n-(k-len(path))

元素无重复，枚举所有 2^n 种无需额外剪枝

### 全排列枚举（Permutations）

```python
def permute(nums):
    res = []
    used = [False]*len(nums)
    def dfs(path):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]: continue
            used[i] = True
            path.append(nums[i])
            dfs(path)
            path.pop()
            used[i] = False
    dfs([])
    return res
``` 

重复元素剪枝: 先对 nums 排序, 同层遇到相同元素且前一个同层未用： if i>0 and nums[i]==nums[i-1] and not used[i-1]: continue 这样能跳过等价分支，避免生成重复排列

### 组合枚举（Combination Sum 系列）

#### 元素可重复（Combination Sum I）
```python
def combinationSum(candidates, target):
    res = []
    def dfs(start, path, total):
        if total == target:
            res.append(path[:]); return
        if total > target:
            return                        # 累加超标剪枝
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            dfs(i, path, total + candidates[i])  # 允许重复选 i
            path.pop()
    dfs(0, [], 0)
    return res
``` 

#### 元素只能用一次（Combination Sum II）
``` python
def combinationSum2(candidates, target):
    candidates.sort()
    res = []
    def dfs(start, path, total):
        if total == target:
            res.append(path[:]); return
        for i in range(start, len(candidates)):
            if total + candidates[i] > target:
                break                       # 排序后遇大值可提前停止
            if i>start and candidates[i]==candidates[i-1]:
                continue                    # 同层去重
            path.append(candidates[i])
            dfs(i+1, path, total + candidates[i])
            path.pop()
    dfs(0, [], 0)
    return res
``` 

剪枝要点
- 累加超标时立即返回
- 事先对候选集排序，遇到不合法可 break
- 同层去重跳过重复分支

### 子矩阵与网格回溯

#### 矩阵路径求和
在 m×n 网格中，从 (0,0) 出发，枚举所有和等于 target 的路径，不可重复访问。

```python
def pathSumGrid(grid, target):
    m, n = len(grid), len(grid[0])
    res = []
    visited = [[False]*n for _ in range(m)]
    def dfs(x, y, path, total):
        if total == target:
            res.append(path[:]); return
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<m and 0<=ny<n and not visited[nx][ny]:
                if total + grid[nx][ny] > target:
                    continue              # 累加超标剪枝
                visited[nx][ny] = True
                path.append((nx, ny))
                dfs(nx, ny, path, total + grid[nx][ny])
                path.pop()
                visited[nx][ny] = False
    visited[0][0] = True
    dfs(0, 0, [(0,0)], grid[0][0])
    return res
```

剪枝技巧
- 边界与已访问判断
- 累加超标直接跳过
- 可加启发式估计：剩余最小值不足以凑齐 target 可提前返回

### 剪枝策略汇总
- 边界与状态剪枝：索引越界、已访问、累加超标
- 排序与早停：对候选集排序后遇到不合条件直接 break
- 同层去重：跳过同层相同元素导致的冗余分支
- 启发式估价：在复杂场景中估计下界或上界，不满足则剪去

## 延伸阅读

- 快速幂的模幂运算与矩阵快速幂  
- 归并排序的自顶向下与自底向上实现  
- 分治在 FFT（快速傅里叶变换）、最近点对、矩阵乘法（Strassen）中的应用  
- 回溯经典题型：全排列、N 皇后、组合总和  
- 剪枝与启发式优化：分支限界、DLX 算法在精确覆盖问题中的应用
