---
layout: post
title: "Algorithm Learning Linear structure"
date: 2025-10-15 17:07:00 +0800
categories: [Algorithm]
---

# 线性结构

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
