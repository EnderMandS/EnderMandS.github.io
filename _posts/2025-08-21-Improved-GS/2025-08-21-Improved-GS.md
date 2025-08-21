---
layout: post
title: "Improved-GS"
date: 2025-08-21 15:09:00 +0800
categories: [Paper, 3DGS]
---

# Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering

Paper: [https://arxiv.org/abs/2508.12313](https://arxiv.org/abs/2508.12313)

Code: [https://github.com/XiaoBin2001/Improved-GS](https://github.com/XiaoBin2001/Improved-GS)

1. 提出 Edge-Aware Score 来更准确地挑选需要 split 的高价值高斯

2. 用 Long-Axis Split 沿椭球长轴进行分裂，降低 clone/split 引入的几何扭曲

3. 用 Recovery-Aware Pruning、Multi-step Update、Growth Control 组合拳，抑制过拟合与失控增长

## Edge‑Aware Score 

让 densify 更聪明地挑选目标

在密度增长（densification）阶段，原生 3DGS 多数是依赖重投影误差或透明度等阈值来判断哪些高斯需要分裂。但这些指标容易“漏掉”轮廓、遮挡边、细杆、薄片等关键结构——这些区域的视觉重要性极高，却未必有很高的像素误差

在计算 densification 触发分数时，结合图像空间的边缘强度（通常可由 Sobel / Laplacian 等算子或深度导数提取）作为加权因子，得到 Edge‑Aware Score。这样，高斯对应像素的分裂优先级就不只是误差高才触发，而是在误差不高但边缘显著的地方也会触发。

在有限的增长预算内，更多资源投到真正决定感知质量的地方。减少平坦区域的无效 split，提升边缘锐度，减少后期过密导致的噪点。

## Long‑Axis Split

顺着“长轴”切，几何更稳

3DGS 的高斯是椭球形状，各向异性参数来源于协方差矩阵分解。常规的 clone 或 isotropic split，会在细长几何或薄片表面引入扭曲

对候选高斯的协方差矩阵求主成分分解（PCA），得到三个主轴方向和长度。找出最长主轴（variance 最大的方向），沿该方向一分为二，位置微调到长轴的正负方向上。这样做等于顺着“不确定性最大”的方向去细化形状——就像先把一根粗棒切成两根细棒，而不是横着切成两块奇怪的扁片。

分裂后的子高斯几何更贴合真实表面，减少异形椭球导致的渲染闪烁。在相同 split 次数下，能更快收敛到目标形状，提升训练稳定性。

#### 获取主轴方向和长度

特征分解：

$$ \Sigma = R \Lambda R^{\mathrm{T}}, \quad \Lambda = \mathrm{diag}\left ( \lambda_1, \lambda_2, \lambda_3 \right ) , \space \lambda_i > 0 $$

其中 $ R = \left [ u_1 \space u_2 \space u_3 \right ] $ 的列向量是单位正交的特征向量，给出三条主轴方向

通常令 $\lambda_1 \ge \lambda_2 \ge \lambda_3 > 0$, 主轴方向 $ u_i $, 方差尺度 $ \lambda_i $, 标准差尺度 $ \sigma_i = \sqrt{\lambda_i} $

椭球半轴长度为

$$ a_i = c \sqrt{\lambda_i} = c \sigma_i $$

使用针对对称矩阵的特征分解（如 `torch.linalg.eigh` / `numpy.linalg.eigh`），数值更稳，且保证实特征值与正交特征向量

训练早期协方差可能接近奇异，可做微正则:

$$ \Sigma \leftarrow \Sigma + \epsilon I, \space \epsilon \in \left [ 10^{-6}, 10^{-4} \right ]  $$

特征向量符号不唯一（$u$ 与 $-u$等价）。为避免帧间抖动，常施加“符号固定”规则（例如令$u$在分量绝对值最大的那一维取非负）

若数值误差导致 $ \mathrm{det} \left ( R \right ) < 0 $ ，可对最后一列取反以恢复右手系，使 $\mathrm{det} \left ( R \right ) = +1$ 。

选取长轴:

$$ i^* = \mathrm{arg}\max_{i} \lambda_i, \quad u_{\mathrm{max}} = u_{i^*}, \quad \lambda_\mathrm{max} =  \lambda _{i^*} $$

长轴方向 $ u_{\mathrm{max}} $ 是最不确定、最该细化的方向

#### 矩量守恒的对称分裂

将一个高斯替换为两个等权重子高斯:

$$ \mu_\pm = \mu \pm d u_{\mathrm{max}} $$

$$ \Sigma_\mathrm{child} = \Sigma - d^2 u_{\mathrm{max}}u_{\mathrm{max}}^\mathrm{T}  $$

其混合的一阶、二阶矩与原高斯匹配（在等权前提下）:

$$ \frac{1}{2} \left [ \Sigma_\mathrm{child} + \left ( \mu_+ - \mu \right ) \left ( \mu_+ - \mu \right )^\mathrm{T}   \right ] + \frac{1}{2} \left [ \Sigma_\mathrm{child} + \left ( \mu_- - \mu \right ) \left ( \mu_- - \mu \right )^\mathrm{T}   \right ] = \Sigma $$

因为两项的均值外积均为 $ d^2 u_{\mathrm{max}}u_{\mathrm{max}}^\mathrm{T} $

正定性要求: 
$$ \Sigma_\mathrm{child}\succ 0 \Leftrightarrow d^2 < \lambda_\mathrm{max} $$

分裂强度:

$$ d = \sqrt{\gamma \lambda_\mathrm{max}} , \quad \gamma \in \left ( 0,1 \right )  $$

其则子协方差在长轴上的特征值变为 $ \left ( 1-\gamma  \right ) \lambda_\mathrm{max} $ 其余轴不变；两个子中心沿长轴对称偏移 $\pm \sqrt{\gamma \lambda_\mathrm{max}} u_\mathrm{max}$

$d$越大（$\gamma$ 越大），子中心分得越开、每个子高斯在长轴方向越“瘦”，整体更细化，但数值更敏感。

该构造在二阶矩意义上严格守恒，训练更稳，避免“越分越胖/越抖”的副作用。

## 过拟合抑制

密度增长会提高表达力，但也容易“过雕”噪声。为此，论文结合了三个控制策略：

### Recovery‑Aware Pruning（恢复感知剪枝）

剪掉渲染贡献低甚至有害的高斯，然后让场景训练几个 step，观察误差变化。如果误差不回升，说明剪掉的是“无用/有害”高斯；反之就恢复，避免删错。

### Multi‑step Update（多步更新）

在 densify 后的几个迭代里，降低学习率或分阶段更新参数，让新生高斯有时间稳定下来，不被大步长更新“打飞”。

### Growth Control（增长控制）

设定一个硬性预算上限，防止模型规模失控膨胀。可根据硬件资源设定不同等级（standard / small budget）。
