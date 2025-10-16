---
layout: post
title: "Algorithm Tree and Graph"
date: 2025-10-16 09:24:00 +0800
categories: [Algorithm]
---

# 二叉树

二叉树是一种每个节点最多有两个子节点的树形结构。常用的节点定义如下：  
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

节点之间按层级组织，根节点在顶层，叶子节点没有子节点。二叉树在计算机科学中用于表达分层关系、支持快速查找与排序等场景。

## 二叉树遍历

遍历是访问二叉树中所有节点的过程，主要分深度优先（DFS）和广度优先（BFS）两大类。

### 递归遍历（DFS）

- 前序遍历（根→左→右）  
- 中序遍历（左→根→右）  
- 后序遍历（左→右→根）  

它们都可通过简单的递归函数实现。

```python
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```
这些模板展示了递归遍历的核心思路，即函数自身调用左子树与右子树，再根据访问顺序组合结果。

### 迭代遍历（基于栈）

当递归深度可能过大或需手动控制顺序时，可用显式栈模拟调用栈。

```python
def inorder_iter(root):
    stack, res = [], []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res
```

相同思路可改写为前序或后序遍历，只需调整压栈和访问节点的时机即可。

### 层序遍历（BFS）

层序遍历按层从上到下、从左到右访问每个节点，需借助队列。

```python
from collections import deque

def level_order(root):
    if not root:
        return []
    queue = deque([root])
    res = []
    while queue:
        node = queue.popleft()
        res.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return res
```

该方法广泛用于计算二叉树的最大宽度、二叉树序列化等场景。

## 树的深度

计算二叉树的深度可通过一次递归或迭代遍历获得, 树的深度（高度）指从根节点到最远叶节点的最长路径长度。

- 递归实现  
  ```python
  def maxDepth(root):
      if not root:
          return 0
      left_h = maxDepth(root.left)
      right_h = maxDepth(root.right)
      return max(left_h, right_h) + 1
  ```
- 迭代（层序遍历）  
  ```python
  from collections import deque

  def maxDepth(root):
      if not root:
          return 0
      depth = 0
      queue = deque([root])
      while queue:
          depth += 1
          for _ in range(len(queue)):
              node = queue.popleft()
              if node.left:
                  queue.append(node.left)
              if node.right:
                  queue.append(node.right)
      return depth
  ```

## 最小公共祖先

最小公共祖先可用「递归分治」或「父指针哈希」等方法求解. 在二叉树中找两个节点的最近公共祖先，可用递归分治或哈希记录父节点两种思路。

- 递归分治  
  ```python
  def lowestCommonAncestor(root, p, q):
      if not root or root == p or root == q:
          return root
      left = lowestCommonAncestor(root.left, p, q)
      right = lowestCommonAncestor(root.right, p, q)
      if left and right:
          return root
      return left or right
  ```
- 父指针＋哈希  
  1. 用哈希表记录每个节点的父节点。  
  2. 从 p 向上收集所有祖先到集合。  
  3. 从 q 向上遍历第一个在集合中的节点即为答案。

---

## 序列化与反序列化

序列化／反序列化常用先序＋空节点标记或层序＋队列的方式实现. 将树转为字符串（序列化），再由字符串复原树（反序列化），常见两种方式。

### 层序（BFS）  
```python
from collections import deque

def serialize(root):
    res = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        res.append(str(node.val) if node else '#')
        if node:
            queue.append(node.left)
            queue.append(node.right)
    return ','.join(res)

def deserialize(data):
    vals = iter(data.split(','))
    root_val = next(vals)
    if root_val == '#':
        return None
    root = TreeNode(int(root_val))
    queue = deque([root])
    while queue:
        node = queue.popleft()
        left_val = next(vals)
        if left_val != '#':
            node.left = TreeNode(int(left_val))
            queue.append(node.left)
        right_val = next(vals)
        if right_val != '#':
            node.right = TreeNode(int(right_val))
            queue.append(node.right)
    return root
```
以上思路即 LeetCode 297 题解。

### 先序（DFS）  
```python
def serialize(root):
    def dfs(node):
        if not node:
            vals.append('#')
            return
        vals.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    vals = []
    dfs(root)
    return ','.join(vals)

def deserialize(data):
    def dfs():
        val = next(vals)
        if val == '#':
            return None
        node = TreeNode(int(val))
        node.left = dfs()
        node.right = dfs()
        return node
    vals = iter(data.split(','))
    return dfs()
```

## 二叉搜索树性质

二叉搜索树（BST）在普通二叉树基础上额外满足：  
- 任意节点的左子树上所有节点值都小于该节点值  
- 任意节点的右子树上所有节点值都大于该节点值  

基于此特性，BST支持高效的查找、插入和删除操作。  
- 查找：从根节点比较大小，沿左或右子树向下递归或迭代，时间复杂度为 \(O(h)\)，其中 \(h\) 是树的高度。  
- 插入：与查找类似，找到空位后链接新节点。  
- 删除：分三种情况处理（叶子、单子树、双子树），在节点替换和指针重连后保持 BST 性质。  

在平衡 BST（如 AVL、红黑树）中，树高度 \(h=O(\log n)\)，则上述操作均为 \(O(\log n)\)。

## 更多延伸

- 平衡二叉搜索树：AVL 树与红黑树原理及实现  
- 二叉树序列化与反序列化：Preorder+Null 标记、层序+索引法  
- 树的高级算法：最低公共祖先、路径总和、二叉堆  
- 地图与图的区分：何时选用二叉树存储结构 vs. 通用图结构  
- 面试常考变式：Morris 遍历（O(1) 空间的中序遍历）、Zigzag 层序遍历

# 图


## 图的邻接矩阵与邻接表表示

图的顶点之间的关系可用两种常见数据结构存储。

- 邻接矩阵  
  - 定义：用大小为 n×n 的二维数组 `adj[i][j]` 表示顶点 i 到顶点 j 的边信息。  
  - 无向图：若存在边则 `adj[i][j]=adj[j][i]=1`，否则为 0；带权图可存储权值或 ∞；  
  - 有向图：若存在 i→j 则 `adj[i][j]=1`（或权值），`adj[j][i]` 保持原值。  
  - 优点：判断任意两顶点是否相邻时间 O(1)；  
  - 缺点：空间复杂度 O(n²)，不适合稀疏图。

- 邻接表  
  - 定义：为每个顶点维护一条链表（或动态数组），存储它的所有出边或邻接点。  
  - 无向图可在两顶点链表中各插入一次；有向图只在源点链表插入。  
  - 优点：空间与实际边数线性相关，O(n+e)；遍历顶点所有邻接边更高效；  
  - 缺点：判断 i、j 是否直接相邻需遍历链表，最坏 O(deg(i))。

## 图的遍历：DFS 与 BFS

图遍历可分为深度优先（DFS）和广度优先（BFS），二者都是从某一顶点出发，访问所有可达顶点。

### 深度优先搜索（DFS）  
- 思路：沿一条路径不断深入，直到无法继续再回溯；  
- 递归实现：  
    ```python
    def dfs(u, adj, visited, res):
        visited[u] = True
        res.append(u)
        for v in adj[u]:
            if not visited[v]:
                dfs(v, adj, visited, res)
    ```
- 迭代实现（显式栈）：  
    ```python
    def dfs_iter(start, adj):
        visited = set()
        stack = [start]
        res = []
        while stack:
            u = stack.pop()
            if u in visited: continue
            visited.add(u)
            res.append(u)
            for v in adj[u]:
                if v not in visited:
                    stack.append(v)
        return res
    ```
- 时间复杂度 O(n+e)，空间 O(n)（递归或栈）。

### 广度优先搜索（BFS）  
- 思路：先访问起点所有直接邻居，再依次向外层拓展；  
- 实现（队列）：  
    ```python
    from collections import deque

    def bfs(start, adj):
        visited = set([start])
        queue = deque([start])
        res = []
        while queue:
            u = queue.popleft()
            res.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return res
    ```
- 常用于找最短路径（无权图）、层次结构遍历。  
- 时间复杂度 O(n+e)，空间 O(n)（队列）。

## 拓扑排序

仅对有向无环图（DAG）定义，输出顶点线性序列，保证每条有向边 u→v 中 u 在 v 之前。

### Kahn 算法（基于入度）  
1. 计算每个顶点的入度；  
2. 将所有入度为 0 的顶点入队；  
3. 出队并加入结果序列，遍历其所有出边，目标顶点入度–1，若减为 0 则入队；  
4. 重复直到队列空。若输出序列长度<n，则存在环。  
```python
from collections import deque

def topo_sort_kahn(n, adj):
    indeg = [0]*n
    for u in range(n):
        for v in adj[u]:
            indeg[v] += 1
    q = deque([u for u in range(n) if indeg[u]==0])
    res = []
    while q:
        u = q.popleft()
        res.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return res if len(res)==n else []
```

### DFS 后序法  
- 对每个未访问顶点执行 DFS，递归完成后将该顶点压入栈；  
- 最终弹栈顺序即为拓扑序。  
```python
def topo_sort_dfs(n, adj):
    visited = [0]*n  # 0=未,1=访中,2=完成
    stack = []
    def dfs(u):
        visited[u] = 1
        for v in adj[u]:
            if visited[v] == 0:
                dfs(v)
        visited[u] = 2
        stack.append(u)
    for u in range(n):
        if visited[u] == 0:
            dfs(u)
    return stack[::-1]
```

## 最短路径算法

### Dijkstra 算法

适用于边权非负的单源最短路。用优先队列维护候选顶点。

```python
import heapq

def dijkstra(n, adj):
    dist = [float('inf')]*n
    dist[0] = 0      # 源点定为 0
    pq = [(0, 0)]    # (距离, 顶点)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
```

- 时间复杂度 O((n+e) log n)，空间 O(n+e)。

### Bellman–Ford 算法

可处理带负权边但无负环的单源最短路，顺序松弛所有边 V−1 轮。

```python
def bellman_ford(n, edges):
    dist = [float('inf')]*n
    dist[0] = 0
    for _ in range(n-1):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break
    # 检测负环
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None  # 发现负环
    return dist
```

- 时间复杂度 O(n·e)，空间 O(n+e)。

### A*

A* 算法是一种启发式搜索算法，用于在图或网格中高效地寻找从起点到目标点的最短路径。它综合了 Dijkstra 保证最短路径的优点和贪心最佳优先搜索的高效性，通过在「已走距离」与「预估剩余距离」之间做加权和来动态选择拓展节点，时间复杂度通常优于单纯的 Dijkstra 或 BFS。

#### 核心公式与启发式函数

A* 针对每个待扩展节点 \(n\) 计算代价函数  
$$
f(n) = g(n) + h(n)
$$
- \(g(n)\)：从起点走到节点 \(n\) 的实际代价  
- \(h(n)\)：从节点 \(n\) 预估到目标的代价（启发式函数）  

要保证算法找到最优解，启发式函数必须满足「可采纳性」（admissible）：对所有节点都不会高估到终点的真实最小代价，即  
$$
\forall n,\quad h(n)\le h^*(n)
$$
且「一致性」（consistent）或「单调性」：对任意相邻节点 \(n\) 和 \(n'\)，满足  
$$
h(n)\le c(n,n') + h(n')
$$
其中 \(c(n,n')\) 是 \(n\) 到 \(n'\) 的实际代价。

#### 数据结构与伪代码

- Open List（优先队列）：存放待考察节点，按 \(f\) 值从小到大排序  
- Closed List（集合）：记录已访问过的节点，避免重复扩展  
- 节点结构通常包含：位置坐标、父节点指针、g 值、h 值、f 值

核心流程：  
```text
初始化：将起点加入 Open List，g(起点)=0, 求 h(起点)
循环直到 Open List 为空或找到目标：
  1. 从 Open List 弹出 f 值最小的节点 cur
  2. 若 cur 为目标，回溯 parent 指针得到路径并退出
  3. 将 cur 加入 Closed List
  4. 对 cur 的每个邻居 next：
     - 若 next 在 Closed List，跳过
     - 计算 tentative_g = g(cur)+cost(cur,next)
     - 若 next 不在 Open List 或 tentative_g<g(next)：
         更新 next.g = tentative_g
         next.h = 预估(next,目标)
         next.f = next.g + next.h
         next.parent = cur
         若 next 不在 Open List，加入 Open List
```
该流程结合优先队列可保证 \(O((V+E)\log V)\) 的时间复杂度，在网格图上通常写作 \(O(b^d)\)，其中 \(b\) 是分支因子，\(d\) 是解深度。

#### 实现示例（Python）

```python
import heapq

def astar(start, goal, h_func, neighbors):
    open_heap = []
    g = {start: 0}
    f = {start: h_func(start, goal)}
    parent = {}
    heapq.heappush(open_heap, (f[start], start))

    closed = set()
    while open_heap:
        cur_f, cur = heapq.heappop(open_heap)
        if cur == goal:
            # 回溯路径
            path = []
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            return path[::-1] + [goal]
        closed.add(cur)

        for nxt, cost in neighbors(cur):
            if nxt in closed:
                continue
            tentative_g = g[cur] + cost
            if tentative_g < g.get(nxt, float('inf')):
                parent[nxt] = cur
                g[nxt] = tentative_g
                h = h_func(nxt, goal)
                f[nxt] = tentative_g + h
                heapq.heappush(open_heap, (f[nxt], nxt))
    return None
```

- `h_func(node, goal)`：启发式函数，如曼哈顿距离或欧几里得距离  
- `neighbors(cur)`：返回节点 `cur` 的所有邻居及到它们的实际代价  

#### 应用与注意事项

- 常见于路径规划、机器人导航、游戏 AI 等场景  
- 启发式函数选择决定效率：  
  - 网格四向移动常用曼哈顿距离  
  - 八向移动可用切比雪夫距离或欧几里得距离  
- 在静态场景下可保证最优；动态场景或高维搜索需结合增量更新或抽样方法（如 Jump Point Search）以提升速度。

## 更多延伸

- 最小生成树：Prim、Kruskal 算法  
- 强连通分量：Tarjan、Kosaraju 算法  
- 单源多目标最短路变种：A* 搜索  
- 最大流：Edmonds–Karp、Dinic 算法  
- 动态连通：并查集在图上的应用

