---
layout: post
title: "Algorithm Dynamic Programming"
date: 2025-10-16 10:07:00 +0800
categories: [Algorithm]
---

# 动态规划

动态规划（Dynamic Programming, DP）通过将原问题拆解为重叠子问题、记录子问题最优解并自底向上或自顶向下组合，避免重复计算并构建全局最优。线性DP包括背包、最短路径、最长公共子序列/子串等；状态压缩DP利用位掩码处理子集枚举；区间DP在$l,r$区间上分治合并；树形DP在树结构上自底向上计算。下文配合典型例题与 Python 实现详解。

## 线性 DP

线性DP问题中，状态往往是一维或二维数组，按顺序迭代即可求解。

### 0/1 背包

问题：给定 \(N\) 件物品和容量为 \(W\) 的背包，每件物品重量 \(w_i\)、价值 \(v_i\)，求最大总价值。  

思路：定义 `dp[j]` 为当前容量恰为 \(j\) 时能取得的最大价值，倒序遍历容量避免重复使用同一物品。  

```python
def knapsack_01(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        w, v = weights[i], values[i]
        for j in range(W, w - 1, -1):
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[W]
```

### 最短路径（Bellman–Ford）

问题：给定带负权但无负环的有向图，求单源最短路径。  

思路：用 `dist[i]` 记录源点到 \(i\) 的最短距离，重复松弛所有边 \(V-1\) 轮。  

```python
def bellman_ford(n, edges, src):
    dist = [float('inf')] * n
    dist[src] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist
```

### 最长公共子序列（LCS）

问题：给定两个字符串 `A` 和 `B`，求它们的最长公共子序列长度。  

思路：定义 `dp[i][j]` 为 `A[:i]` 与 `B[:j]` 的 LCS 长度，若 `A[i-1]==B[j-1]`，则 `dp[i][j]=dp[i-1][j-1]+1`，否则取上或左最大值。  

```python
def lcs(A, B):
    n, m = len(A), len(B)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if A[i-1] == B[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]
```

### 最长公共子串

问题：求 `A` 与 `B` 的最长公共**连续**子串长度。  

思路：定义 `dp[i][j]` 为以 `A[i-1]` 和 `B[j-1]` 结尾的最长公共子串长度，若相等则 `dp[i][j]=dp[i-1][j-1]+1`，否则为 0。  

```python
def longest_common_substr(A, B):
    n, m = len(A), len(B)
    dp = [[0]*(m+1) for _ in range(n+1)]
    res = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if A[i-1] == B[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                res = max(res, dp[i][j])
    return res
```

## 状态压缩 DP

当子问题涉及对集合的枚举时，可用位掩码表示状态，状态空间为 \(2^N\)。

### 示例：旅行商问题（TSP）

问题：给定完全图中 \(N\) 个城市和对应距离矩阵 `dist[i][j]`，求最短环路。  

思路：定义 `dp[mask][i]` 为已访问集合 `mask` 且最后停在城市 `i` 的最短距离，转移枚举上一步城市 `j`：  

```python
def tsp(dist):
    N = len(dist)
    INF = float('inf')
    dp = [[INF]*N for _ in range(1<<N)]
    dp[1][0] = 0  # 从城市0出发
    for mask in range(1, 1<<N):
        for i in range(N):
            if not (mask & (1<<i)): continue
            prev_mask = mask ^ (1<<i)
            for j in range(N):
                if prev_mask & (1<<j):
                    dp[mask][i] = min(dp[mask][i], dp[prev_mask][j] + dist[j][i])
    # 回到起点
    ans = min(dp[(1<<N)-1][i] + dist[i][0] for i in range(N))
    return ans
```

## 区间 DP

区间DP 常用于合并与分治场景，定义 `dp[l][r]` 为区间 \([l,r]\) 的最优解，转移枚举分割点 `k`。

### 合并石头

问题：给定石头数组 `stones`，每次只能合并相邻两堆，合并代价为重量之和，求最小总代价。  

思路：定义 `dp[l][r]` 为区间 \([l,r]\) 合并为一堆的最小代价，预处理 `prefix[i]` 为前缀和；  

```python
def mergeStones(stones):
    n = len(stones)
    pre = [0]*(n+1)
    for i in range(n): pre[i+1] = pre[i] + stones[i]
    dp = [[0]*n for _ in range(n)]
    for length in range(2, n+1):
        for l in range(n-length+1):
            r = l + length - 1
            dp[l][r] = float('inf')
            for k in range(l, r):
                cost = dp[l][k] + dp[k+1][r] + pre[r+1] - pre[l]
                dp[l][r] = min(dp[l][r], cost)
    return dp[0][n-1]
```

## 树形 DP

树形DP 在树结构上自底向上计算，每个节点的状态基于子节点结果。

### 二叉树直径

问题：求二叉树中任意两节点路径长度的最大值。  

思路：对每个节点定义返回值为“从该节点向下的最大路径长度”，并在递归过程中更新全局直径。  

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def diameterOfBinaryTree(root):
    ans = 0
    def dfs(node):
        nonlocal ans
        if not node: return 0
        left = dfs(node.left)
        right = dfs(node.right)
        ans = max(ans, left + right)
        return max(left, right) + 1
    dfs(root)
    return ans
```

以上内容覆盖线性DP、状态压缩DP、区间DP与树形DP，并配合典型例题与 Python 实现，助你系统梳理动态规划各大类模型与解题思路。

