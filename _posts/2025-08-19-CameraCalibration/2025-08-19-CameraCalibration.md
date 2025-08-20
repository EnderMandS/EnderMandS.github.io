---
layout: post
title: "Camera Calibration"
date: 2025-08-19 10:46:00 +0800
categories: [Paper, Vision]
---

# Camera Calibration

相机标定的目标是确定从三维点到像素的映射参数：内参、外参和畸变

## 单应矩阵
在三维世界里，如果两个相机都拍了同一个平面（比如棋盘格标定板），那么这个平面上的点从一张图到另一张图之间的映射，不需要考虑深度变化，可以用一个 3×3 的矩阵直接描述。这个矩阵就是单应矩阵（Homography）。它处理的是齐次坐标，所以只关心方向关系，不在乎整体缩放

单应矩阵其实把几个几何因素揉在了一起：
- 相机的内参（焦距、主点等）
- 相机在两次拍摄中的姿态变化（旋转和平移）
- 平面的空间位置和朝向（法向量和到相机的距离）

组合起来，它能回答：“这个平面上的点在另一张图里应该落在哪”

在针孔相机模型下，如果两个相机拍摄的是同一个平面，那么这个平面在两幅图像中的投影之间存在一个唯一的单应矩阵。它是二维平面到二维平面的映射，保持直线为直线，但不保持长度、角度等度量性质。

把第一张图像的平面“拉伸、旋转、透视变形”后，就能和第二张图像的平面区域对齐。这种变换就是单应性变换。

### 定义

$ 3 \times 3 $可逆矩阵，它描述了同一平面上的点在两个不同视角的图像之间的射影变换关系：

$$ q_2 \propto H q_1 $$

其中：
- $ q1 = \left [ u_1, v_1, 1 \right ]^{\mathrm{T}} $ 是第一张图像中的点（齐次坐标）
- $ q2 = \left [ u_2, v_2, 1 \right ]^{\mathrm{T}} $ 是第二张图像中的对应点
- $ \propto $ 表示两边相差一个非零尺度因子（齐次坐标的特性）

世界坐标系中的平面方程：

$$ n^{\mathrm{T}}P + d = 0 $$

其中 $n$ 是平面法向量，$d$ 是相机到平面的距离

相机投影模型：

$$ sq = K \left [ R \mid t \right ] P $$

对同一平面上的点，消去深度 $z$ 后，可以得到：

$$ H_{21} = K_2 \left ( R_{21} + \frac{t_{21}n^{\mathrm{T}}}{d}  \right ) K_1^{-1} $$

这就是单应矩阵的解析形式

单应矩阵有 9 个元素，但由于齐次坐标的尺度不唯一，只有 8 个自由度。每对匹配点提供 2 个独立方程，因此至少需要 4 对非共线的匹配点才能求解 $H$。实际应用中会用更多点并结合 RANSAC 等方法来提高鲁棒性。

## 坐标系
### 世界坐标系

三维点 

$$ X_W = \left [ X\space Y \space Z \space  1 \right ] ^T $$

### 相机坐标系

- 原点：相机光心
- x轴：水平向右（从相机自身视角看）
- y轴：竖直向下（从相机自身视角看）
- z轴：沿着相机的光轴指向前方，即镜头看出去的方向

通过外参 $ R, t $ 将世界点变换到相机坐标系

$$ X_c = RX + t $$

### 归一化像平面 Normalized Image Plane

透视投影到焦距为 1 的平面

$$ x = \frac{X_c}{Z_c}, \quad y = \frac{Y_c}{Z_c} $$

其中: 
- $X_c, Y_c, Z_c$ : 三维点在相机坐标系中的坐标（单位通常是米或毫米）、$Z_c$是空间点在相机坐标系中的深度
- $ x,y $ : 归一化平面上的坐标，单位是“以焦距为 1 的长度单位”，此时没有考虑像素的缩放和平移
- $ \frac{X_c}{Z_c} , \frac{Y_c}{Z_c} $: 透视投影关系，即点的横纵位置除以深度，表示投影到距离相机原点 1 个单位焦距的成像平面

理想针孔相机的成像平面距离投影中心为 1 个单位长度（焦距）

归一化平面是一个理想针孔相机的成像结果，没有畸变，也没有传感器像素网格的影响。归一化像平面处理的是“几何投影”，对应空间比例

### 像素坐标系 Pixel Coordinate System

- 原点在图像左上角
- x轴：水平向右
- y轴：竖直向下
- z轴：向前

由内参矩阵 $ K $ 将归一化坐标映射到像素

$$ s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} =
\mathbf{K}
\begin{bmatrix}
\mathbf{R} & \mathbf{t}
\end{bmatrix}
\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix} $$

其中：

$$ \mathbf{K} = 
\begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix} $$

- $ u, v $: 像素坐标系中的横纵坐标（单位：像素）
- $ s $: 整体缩放因子（归一化平面到像素坐标的齐次比例系数）
- $ K $: 相机内参矩阵
    - $ f_x = f/m_x $: 水平方向焦距，单位是像素（即物理焦距乘以水平方向像素密度）
    - $ f_y = f/m_y $: 垂直方向焦距，单位是像素
    - $ m_x, m_y $: 为像元尺寸（pixel size）在水平方向和垂直方向上的物理长度，单位一般是毫米/像素或微米/像素
    - $ s $: 主轴倾斜系数（skew），当成像平面横纵轴不垂直时非零，大部分相机可近似为0
    - $ c_x, c_y $: 主点坐标，即光轴与成像平面交点在像素坐标系中的位置（通常接近图像中心）

像素坐标也可写为：
$$ u = f_x \cdot x + c_x, \quad v = f_y \cdot y + c_y $$

$ f_x, f_y $ 是比例因子，用来把真实世界的物理距离（mm）投影换算成像平面上的像素距离

像素坐标系处理的是“成像传感器采样与像素网格映射”，对应像素单位

## 透视投影 & 畸变模型

### 无畸变投影

$$ s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} =
\begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
\mathbf{R} & \mathbf{t}
\end{bmatrix}
\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix} $$

### 径向与切向畸变（常用 Brown–Conrady 模型）

先计算归一化坐标与半径：
$$ x = \frac{X_c}{Z_c}, \quad y = \frac{Y_c}{Z_c}, \quad r^2 = x^2+y^2 $$

径向畸变：
$$ x_r = x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6), \quad y_r = y\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) $$

切向畸变：
$$ \Delta x_t = 2p_1xy + p_2(r^2 + 2x^2),\quad \Delta x_t = 2p_1xy + p_2(r^2 + 2x^2) $$

合成畸变后的归一化坐标:
$$ \tilde{x} = x_r + \Delta x_t,\quad \tilde{y} = y_r + \Delta y_t $$

映射到像素：
$$ \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} =
\mathbf{K}
\begin{bmatrix} \tilde{x} \\ \tilde{y} \\ 1 \end{bmatrix} $$

（可选）薄棱镜畸变项：
$$ \Delta x_p = s_1 r^2 + s_2 r^4, \quad \Delta y_p = s_3 r^2 + s_4 r^4 $$

## 基于平面标定板的线性初始化（Zhang 方法）

Zhang 方法把“拍平面板的多张照片”转成“线性求内参 + 非线性细化”的问题。核心抓手是平面单应性，它把每一张图里标定板的平面和图像平面用一个 3×3 矩阵连起来

### 单应矩阵做桥梁

在图像配准任务中，单应矩阵连接的是两个已知图像。在 Zhang 标定里，单应矩阵连接的是标定板平面的世界坐标与当前照片的像素坐标。

当一个平面被拍到照片里时，平面上的点和图像上的点之间存在一个 3×3 的单应矩阵关系。

每一张照片都会有一个自己的单应矩阵，它“编码”了该视角下相机的姿态（外参）和内部几何特性（内参）的组合

关键想法：多个不同视角的单应矩阵中，内参部分是相同的，因为相机没换；而外参是不同的，因为相机位置和角度变了

### 线性方程锁定内参

从每个单应矩阵中，能写出一些只依赖于相机内参的约束公式。

把多张照片的约束凑在一起，就是一个线性方程组，直接解就能得到内参的初值。

意义：这是一个闭式解（直接计算），快速、直观，不需要从一开始就做复杂的非线性优化。

### 内参反推每一张图的外参

有了内参，就可以把单应矩阵分解成旋转矩阵和位移向量，即相机在拍那张图时是怎么“站”的。

### 引入畸变模型

实际镜头都有畸变，尤其是广角和廉价镜头。

Zhang 方法在外参、内参固定的情况下，引入畸变参数（如径向畸变、切向畸变），再做一次全局优化，把这些畸变系数估出来。

这样可以保证将来用这个模型去校正图像时，直线不会被拉弯，几何测量更准确。

### 全局优化

最后一步，把内参、外参、畸变参数全部放在一起，通过最小化重投影误差（实际像素位置和模型预测位置的差距），用非线性优化方法同时更新，得到最精确的一组参数。

## Zhang方法 公式

### 平面单应与内参约束

设标定板在世界坐标平面上 $Z=0$, 世界点 $ X_p = \left [ X \space Y \space 1 \right ] ^T $。

素与平面之间的单应关系（每幅图像一个 $ H $）：
$$s\begin{bmatrix}u\\v\\1\end{bmatrix} = \mathbf{H}\,\mathbf{X}_p,\quad
\mathbf{H}=\mathbf{K}\begin{bmatrix}\mathbf{r}_1&\mathbf{r}_2&\mathbf{t}\end{bmatrix}$$

用 DLT 从点对应估计单应矩阵 $ H $（每对对应点给两行）:
$$ \text{给定}\ (X,Y)\leftrightarrow(u,v),\ \text{构造}\ \mathbf{A}_h:
\quad
\begin{bmatrix}
X & Y & 1 & 0 & 0 & 0 & -uX & -uY & -u\\
0 & 0 & 0 & X & Y & 1 & -vX & -vY & -v
\end{bmatrix} $$

$$ \text{堆叠多点得到}\ \mathbf{A}_h\mathbf{h}=\mathbf{0},\ \|\mathbf{h}\|=1,\ 
\mathbf{h}=\operatorname*{argmin}_{\|\mathbf{h}\|=1}\|\mathbf{A}_h\mathbf{h}\|\ \Rightarrow\ \mathbf{H} $$

把内参“装进”可线性求解的约束:
$$ \mathbf{B}=\mathbf{K}^{-T}\mathbf{K}^{-1},\quad
\mathbf{H}=[\mathbf{h}_1\ \mathbf{h}_2\ \mathbf{h}_3] $$

$$ \mathbf{h}_1^{\top}\mathbf{B}\mathbf{h}_2=0,\qquad
\mathbf{h}_1^{\top}\mathbf{B}\mathbf{h}_1=\mathbf{h}_2^{\top}\mathbf{B}\mathbf{h}_2 $$

向量化形成线性系统 $ Vb = 0 $（每幅图两行）：
$$ \mathbf{b}=
\begin{bmatrix}
B_{11}&B_{12}&B_{22}&B_{13}&B_{23}&B_{33}
\end{bmatrix}^{\top} $$

$$ \mathbf{v}_{ij}=
\begin{bmatrix}
h_{i1}h_{j1}\\
h_{i1}h_{j2}+h_{i2}h_{j1}\\
h_{i2}h_{j2}\\
h_{i3}h_{j1}+h_{i1}h_{j3}\\
h_{i3}h_{j2}+h_{i2}h_{j3}\\
h_{i3}h_{j3}
\end{bmatrix},\quad
\begin{cases}
\mathbf{v}_{12}^{\top}\mathbf{b}=0\\
(\mathbf{v}_{11}-\mathbf{v}_{22})^{\top}\mathbf{b}=0
\end{cases} $$

$$ \mathbf{V}\mathbf{b}=\mathbf{0},\ \|\mathbf{b}\|=1\ \Rightarrow\ \mathbf{b}\ \text{取 }\mathbf{V}\text{最小奇异值对应向量} $$

### 从 $B$ 闭式恢复内参矩阵 $K$

把 $b$还原成对称矩阵 $B$
$$ \mathbf{B}=
\begin{bmatrix}
B_{11}&B_{12}&B_{13}\\
B_{12}&B_{22}&B_{23}\\
B_{13}&B_{23}&B_{33}
\end{bmatrix} $$

常用闭式解（对应 K 的上三角形式）：
$$ v_0=\frac{B_{12}B_{13}-B_{11}B_{23}}{B_{11}B_{22}-B_{12}^2} $$

$$ \lambda = B_{33} - \frac{B_{13}^2 + v_0\,(B_{12}B_{13}-B_{11}B_{23})}{B_{11}} $$

$$ f_x=\sqrt{\frac{\lambda}{B_{11}}},\qquad
f_y=\sqrt{\frac{\lambda B_{11}}{B_{11}B_{22}-B_{12}^2}} $$

$$ s=-\frac{B_{12}\,f_x^2 f_y}{\lambda},\qquad
c_x=\frac{s\,v_0}{f_y}-\frac{B_{13}\,f_x^2}{\lambda},\qquad
c_y=v_0 $$

$$ \mathbf{K}=
\begin{bmatrix}
f_x & s & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix} $$

### 逐幅恢复外参

先做去内参变换，得到未归一的旋转列与位移

$$ \hat{\mathbf{r}}_1=\mathbf{K}^{-1}\mathbf{h}_1,\quad
\hat{\mathbf{r}}_2=\mathbf{K}^{-1}\mathbf{h}_2,\quad
\hat{\mathbf{t}}=\mathbf{K}^{-1}\mathbf{h}_3 $$

统一尺度并构造右手正交基：

$$ \lambda=\frac{1}{\|\hat{\mathbf{r}}_1\|}=\frac{1}{\|\hat{\mathbf{r}}_2\|},\quad
\mathbf{r}_1=\lambda\hat{\mathbf{r}}_1,\quad
\mathbf{r}_2=\lambda\hat{\mathbf{r}}_2,\quad
\mathbf{r}_3=\mathbf{r}_1\times \mathbf{r}_2,\quad
\mathbf{t}=\lambda\hat{\mathbf{t}} $$

正交化到最近的 $SO\left( 3 \right)$（抑制噪声）：

$$ \mathbf{R}_0=[\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{r}_3],\quad
\mathbf{R}=\operatorname{Proj}_{SO(3)}(\mathbf{R}_0)\ \ (\text{如对 }\mathbf{R}_0\text{做 SVD 的极分解}) $$

### 畸变建模与全局非线性优化

Brown–Conrady 畸变（径向 + 切向）：
$$ r^2=x^2+y^2 $$

$$ x_r=x(1+k_1 r^2+k_2 r^4+k_3 r^6),\quad
y_r=y(1+k_1 r^2+k_2 r^4+k_3 r^6) $$

$$ \Delta x_t=2p_1xy+p_2(r^2+2x^2),\quad
\Delta y_t=p_1(r^2+2y^2)+2p_2xy $$

$$ \tilde{x}=x_r+\Delta x_t,\quad \tilde{y}=y_r+\Delta y_t $$

$$ \begin{bmatrix}u\\v\\1\end{bmatrix}
\sim
\mathbf{K}\begin{bmatrix}\tilde{x}\\\tilde{y}\\1\end{bmatrix} $$

全局重投影误差最小化（LM 等）：

$$ \min_{\Theta}\ \sum_{i=1}^{N_{\text{view}}}\sum_{j=1}^{N_{\text{pt}}}
\left\|
\mathbf{z}_{ij}-
\pi\big(\mathbf{K},\mathbf{d},\mathbf{R}_i,\mathbf{t}_i;\mathbf{X}_j\big)
\right\|_2^2 $$

$$ \Theta=\{\mathbf{K},\mathbf{d},\{\mathbf{R}_i,\mathbf{t}_i\}_{i=1}^{N_{\text{view}}}\},\quad
\mathbf{d}=[k_1,k_2,k_3,p_1,p_2] $$

$$ \text{可选鲁棒核（Huber）: }\
\rho(r)=
\begin{cases}
\frac{1}{2}r^2,& |r|\le \delta\\
\delta(|r|-\frac{1}{2}\delta),& |r|>\delta
\end{cases} $$

$$  $$
