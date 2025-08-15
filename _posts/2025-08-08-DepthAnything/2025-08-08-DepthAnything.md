---
layout: post
title:  "Depth Anything"
date:   2025-08-08 00:00:00 +0800
categories: [Paper, Vision]
---

# Depth Anything

## v1
Depth Anything 沿用典型的 encoder–decoder 结构，通过两阶段训练（Teacher → Student）打造一个通用且高性能的单目深度估计基础模型

- Teacher 阶段
  - 在 1.5M 张标注深度图像上训练，使用仿射不变损失对齐多数据集的尺度漂移
  - 编码器加载 DINOv2 预训练权重，解码器采用 DPT 结构
- Student 阶段
  - 用训练好的 Teacher 模型对 62M 张未标注图像生成稠密伪标签
  - 在每个 batch 中按 1:2 比例混合标注与伪标注样本，对后者施加强扰动（色彩抖动、模糊 + CutMix）并加入语义对齐约束，令网络学到更鲁棒的深度表征

### Encoder
Backbone：DINOv2 预训练的 Vision Transformer
- 输入预处理
  - 训练时将短边 resize 至 518 px、中心裁剪为 518×518 px
  - 推理时保持原始长宽，保证二者均为 14 (DINOv2)的整数倍

### Decoder
结构：Dense Prediction Transformer（DPT）

- 特征融合
  - 从 ViT 的多个 stage（patch 分辨率不同）抽取特征图
  - 用 1×1 卷积降维后，逐级上采样并与浅层特征做 skip-connection
  - 最终输出 H×W 的 disparity map
- 深度转换
  - 训练时将真实深度 d 转为 disparity d′=1/d ，归一化到 [0,1]
  - 推理输出后再反向映射为实际深度

### 语义辅助特征对齐
冻结一个 DINOv2 Encoder（不更新权重）作为语义教师
- 提取两套特征：
  - f: Depth Anything Student Encoder 的中间特征
  - f′: 冻结 DINOv2 Encoder 输出的语义特征
- 对齐损失
  - 只对余弦相似度低于阈值 α （默认 0.85）的像素施加，兼顾深度判别与语义一致性

### 损失与训练
总损失 $$ L = L_{labeled} + L_{unlabeled} + L_{feat} $$
- 仿射不变 MAE：忽略不同数据集的尺度和平移漂移
- 强扰动目标：在伪标注样本上加色彩抖动、高斯模糊 + 50% 概率 CutMix
- 语义保留：特征对齐避免将深度模型导向离散分割标签

## Pipeline
### 1. 数据引擎构建
  - 标注数据：使用公开深度数据集（如 NYU, KITTI）训练高性能教师模型；
  - 未标注数据：从 LAION-400M 等图像集合中筛选高质量图像，作为伪标签训练的输入；
  - 多模态处理：数据预处理包括尺度对齐、图像增强、遮挡模拟等。

### 2. 教师模型（Teacher）训练
  - 构建基础深度估计器（如 DPT）并在带标签数据上训练；
  - 生成伪标签：对未标注图像推理，输出伪深度图；
  - 此步骤是 V1 的核心创新之一：利用模型生成标签，扩大训练集规模。

### 3. 学生模型（Student）训练
  - 学生模型采用 Transformer 编码器（ViT-b/l/g 等） + DPT 解码头；
  - 在未标注图像 + 教师伪标签组合上进行训练；
  - 优化目标是提高对真实图像的泛化能力。

## 模型架构
- 编码器 ViT：从 DINOv2 或 BEiT 初始化，可选 base/large/giant
- 解码器 DPT：Transformer 解码架构，包括多层特征融合模块
- 上采样结构：RefineNet风格结构，用于恢复高分辨率深度图

# v2
相比v1:
- 用合成图像替代标注的真实图像
- 扩大教师模型容量（使用更强编码器）
- 利用大规模真实图像构建伪标签桥梁训练学生模型

## Pipeline
### 1. 数据引擎：构建训练数据集
- 使用合成图像（如 Blender 或 Unity 渲染）制作高质量深度图
- 收集大规模真实世界图像（如来自 LAION 数据集），用于伪标签生成

### 2. 教师模型训练（Teacher）
- 教师模型结构：DINOv2-G 编码器 + DPTHead 解码器
- 在合成图像上进行监督训练，得到高精度深度预测能力

### 3. 自动标注（伪标签生成）
- 用教师模型推理大规模真实图像，生成伪深度标签
- 构建出跨模态学习的“桥梁数据集”，弥合 domain gap

### 4. 学生模型训练（Student）
- 用真实图像 + 伪标签进行训练，提升模型对真实世界的泛化能力
- 可根据需求选用轻量版本（ViTs、ViTb）或大型版本（ViTl、ViTg）

## 模型架构
- 编码器：DINOv2 系列 ViT 模型，使用 ViTl（或 ViTg）进行高质量特征提取，提取多层特征（例如第4、11、17、23层），用于构建深度图
- DPTHead + RefineNet： 包括四个投射层 + 上采样层，使用一系列 FeatureFusionBlock 进行多层级融合，