<!-- # 

视频：三维数据

灰度图像：二维矩阵

fps 帧每秒，视频的效果，在时间上多了一维，每一帧是一个单纯的图像

正弦、余弦，谐波、基波

傅里叶变换 -> 逆傅里叶变换

对称特性 -->

# 实验五 - 声音、图像的基本理解和 GMM 背景建模

|学号|姓名|日期
|:--:|:--:|:--:|
|1160300625|李一鸣|2018 年 11 月 30 日|

- [实验五 - 声音、图像的基本理解和 GMM 背景建模](#%E5%AE%9E%E9%AA%8C%E4%BA%94---%E5%A3%B0%E9%9F%B3%E5%9B%BE%E5%83%8F%E7%9A%84%E5%9F%BA%E6%9C%AC%E7%90%86%E8%A7%A3%E5%92%8C-gmm-%E8%83%8C%E6%99%AF%E5%BB%BA%E6%A8%A1)
    - [信号的傅里叶变换及其幅度、相位](#%E4%BF%A1%E5%8F%B7%E7%9A%84%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2%E5%8F%8A%E5%85%B6%E5%B9%85%E5%BA%A6%E7%9B%B8%E4%BD%8D)
        - [傅里叶变换](#%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2)
            - [定义](#%E5%AE%9A%E4%B9%89)
            - [傅里叶级数](#%E5%82%85%E9%87%8C%E5%8F%B6%E7%BA%A7%E6%95%B0)
            - [离散傅里叶变换](#%E7%A6%BB%E6%95%A3%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2)
                - [例子](#%E4%BE%8B%E5%AD%90)
                - [离散傅里叶变换的作用](#%E7%A6%BB%E6%95%A3%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2%E7%9A%84%E4%BD%9C%E7%94%A8)
                - [`numpy` 实现细节](#numpy-%E5%AE%9E%E7%8E%B0%E7%BB%86%E8%8A%82)
                    - [归一化](#%E5%BD%92%E4%B8%80%E5%8C%96)
                    - [实变换和 Hermitian 变换](#%E5%AE%9E%E5%8F%98%E6%8D%A2%E5%92%8C-hermitian-%E5%8F%98%E6%8D%A2)
                    - [高维空间](#%E9%AB%98%E7%BB%B4%E7%A9%BA%E9%97%B4)
        - [wav 文件](#wav-%E6%96%87%E4%BB%B6)
            - [wav 的格式](#wav-%E7%9A%84%E6%A0%BC%E5%BC%8F)
            - [wav 的读取](#wav-%E7%9A%84%E8%AF%BB%E5%8F%96)
            - [wav 转换结果](#wav-%E8%BD%AC%E6%8D%A2%E7%BB%93%E6%9E%9C)
                - [`crane_bump.wav`](#cranebumpwav)
                - [`engine.wav`](#enginewav)
                - [`guitartune.wav`](#guitartunewav)
            - [幅度谱](#%E5%B9%85%E5%BA%A6%E8%B0%B1)
                - [`crane_bump.wav-amplitude.svg`](#cranebumpwav-amplitudesvg)
                - [`engine.wav-amplitude.svg`](#enginewav-amplitudesvg)
                - [`guitartune.wav-amplitude.svg`](#guitartunewav-amplitudesvg)
            - [相位谱](#%E7%9B%B8%E4%BD%8D%E8%B0%B1)
                - [`crane_bump.wav-phase.svg`](#cranebumpwav-phasesvg)
                - [`engine.wav-phase.svg`](#enginewav-phasesvg)
                - [`guitartune.wav-phase.svg`](#guitartunewav-phasesvg)
        - [bmp 文件](#bmp-%E6%96%87%E4%BB%B6)
            - [bmp 的格式](#bmp-%E7%9A%84%E6%A0%BC%E5%BC%8F)
            - [bmp 的读取](#bmp-%E7%9A%84%E8%AF%BB%E5%8F%96)
            - [bmp 转换结果](#bmp-%E8%BD%AC%E6%8D%A2%E7%BB%93%E6%9E%9C)
                - [原始图片](#%E5%8E%9F%E5%A7%8B%E5%9B%BE%E7%89%87)
                - [幅度谱](#%E5%B9%85%E5%BA%A6%E8%B0%B1-1)
                - [相位谱](#%E7%9B%B8%E4%BD%8D%E8%B0%B1-1)
                - [仅对幅度进行逆傅里叶变换](#%E4%BB%85%E5%AF%B9%E5%B9%85%E5%BA%A6%E8%BF%9B%E8%A1%8C%E9%80%86%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2)
                - [仅对相位进行逆傅里叶变换](#%E4%BB%85%E5%AF%B9%E7%9B%B8%E4%BD%8D%E8%BF%9B%E8%A1%8C%E9%80%86%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2)
                - [同时使用幅度、相位进行逆傅里叶变换](#%E5%90%8C%E6%97%B6%E4%BD%BF%E7%94%A8%E5%B9%85%E5%BA%A6%E7%9B%B8%E4%BD%8D%E8%BF%9B%E8%A1%8C%E9%80%86%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2)
        - [正余弦信号之和](#%E6%AD%A3%E4%BD%99%E5%BC%A6%E4%BF%A1%E5%8F%B7%E4%B9%8B%E5%92%8C)
            - [原始信号](#%E5%8E%9F%E5%A7%8B%E4%BF%A1%E5%8F%B7)
            - [幅度谱](#%E5%B9%85%E5%BA%A6%E8%B0%B1-2)
            - [相位谱](#%E7%9B%B8%E4%BD%8D%E8%B0%B1-2)
        - [加窗口的傅里叶变换](#%E5%8A%A0%E7%AA%97%E5%8F%A3%E7%9A%84%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2)
            - [窗口变换](#%E7%AA%97%E5%8F%A3%E5%8F%98%E6%8D%A2)
            - [矩形窗](#%E7%9F%A9%E5%BD%A2%E7%AA%97)
            - [变换结果](#%E5%8F%98%E6%8D%A2%E7%BB%93%E6%9E%9C)
                - [窗口长度 = 128](#%E7%AA%97%E5%8F%A3%E9%95%BF%E5%BA%A6--128)
                - [窗口长度 = 512](#%E7%AA%97%E5%8F%A3%E9%95%BF%E5%BA%A6--512)
    - [低通滤波器](#%E4%BD%8E%E9%80%9A%E6%BB%A4%E6%B3%A2%E5%99%A8)
        - [幅度谱和相位谱](#%E5%B9%85%E5%BA%A6%E8%B0%B1%E5%92%8C%E7%9B%B8%E4%BD%8D%E8%B0%B1)
        - [低通、高通、带通](#%E4%BD%8E%E9%80%9A%E9%AB%98%E9%80%9A%E5%B8%A6%E9%80%9A)
        - [例子](#%E4%BE%8B%E5%AD%90-1)
        - [高斯白噪声](#%E9%AB%98%E6%96%AF%E7%99%BD%E5%99%AA%E5%A3%B0)
        - [图像对齐](#%E5%9B%BE%E5%83%8F%E5%AF%B9%E9%BD%90)
    - [视频中的高斯背景建模](#%E8%A7%86%E9%A2%91%E4%B8%AD%E7%9A%84%E9%AB%98%E6%96%AF%E8%83%8C%E6%99%AF%E5%BB%BA%E6%A8%A1)
        - [混合高斯模型](#%E6%B7%B7%E5%90%88%E9%AB%98%E6%96%AF%E6%A8%A1%E5%9E%8B)
        - [视频处理](#%E8%A7%86%E9%A2%91%E5%A4%84%E7%90%86)
        - [生成图片](#%E7%94%9F%E6%88%90%E5%9B%BE%E7%89%87)
        - [均值聚类](#%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB)
        - [分离结果](#%E5%88%86%E7%A6%BB%E7%BB%93%E6%9E%9C)
    - [参考文献](#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE)

## 信号的傅里叶变换及其幅度、相位

编程阅读一个 wav 音频文件和一幅灰度 bitmap 图像，然后对音频文件和灰度图像分别做离散傅里叶变换，画出其幅度和相位信息。

### 傅里叶变换

经傅里叶变换生成的函数 $\hat{f}$ 称作原函数 $f$ 的傅里叶变换、亦称频谱。在许多情况下，傅里叶变换是可逆的，即可通过 $\hat{f}$ 得到其原函数 $f$。通常情况下，$f$ 是实数函数，而 $\hat{f}$ 则是复数函数，用一个复数来表示振幅和相位。

『傅里叶变换』一词既指变换操作本身（将函数 $f$ 进行傅里叶变换），又指该操作所生成的复数函数（$\hat{f}$ 是 $f$ 的傅里叶变换）。

#### 定义

连续傅里叶变换将可积函数 $f : \mathbb R \rightarrow \mathbb C$ 表式成负指数函数的积分或级数形。

负指数函数的积分：

$$
\hat{f}(\xi) = \int_{-\infty}^{\infty}f(x)e^{-2\pi ix\xi}dx, \xi 为任意实数

\tag{1}
$$

自变量 $x$ 表示时间（以秒为单位），变换变量 $\xi$ 表示频率（以赫兹为单位）。在适当的条件下，$\hat{f}$ 可由逆变换（inverse Fourier transform）由下式确定 $f$：

$$
f(x) = \int_{-\infty}^{\infty}\hat{f}(\xi)e^{2\pi i\xi x}d\xi, x 为任意实数

\tag{2}
$$

$f$ 和 $\hat{f}$ 常常被称为傅里叶积分对或傅里叶变换对。

#### 傅里叶级数

连续形式的傅里叶变换其实是傅里叶级数（Fourier series）的推广，因为积分其实是一种极限形式的求和算子而已。对于周期函数，其傅里叶级数是存在的：

$$

f(x) = \sum_{n=-\infty}^{\infty}F_ne^{inx}

\tag{3}
$$

其中 $F_n$ 为复振幅。对于实值函数，函数的傅里叶级数可以写为：

$$
\begin{aligned}
    f(x)
    &= \frac{a}{2} + \sum_{n=1}^{\infty}F_ne^{inx} \\
    &= \frac{a}{2} + \sum_{n=1}^{\infty}[a_n\cos{nx}+b_n\sin(nx)]
\end{aligned}

\tag{4}
$$

对于周期为 $L$ 而不是 $2\pi$ 的情况，傅里叶级数将转变为：

$$
f(x) = \sum_{n=-\infty}^{\infty}F_ne^{i2\pi nx/L}

\tag{5}
$$

将复数形式的傅里叶级数 $(5)$ 式作如下修改：

1. 用 $\frac{n}{L}$ 替换 $x$
2. 将 $F_n$ 替换为 $F(k)dk$
3. 将作和改为积分

就可以得到了连续型的 $(2)$ 式。

The Fourier transform is a generalization of the complex Fourier series in the limit as L->infty. Replace the discrete A_n with the continuous F(k)dk while letting  n/L->k. Then change the sum to an integral, and the equations become

#### 离散傅里叶变换

离散傅里叶变换（Discrete Fourier Transform，缩写为 DFT）是离散时间傅里叶变换（DTFT）的特例，DTFT 在时域上离散，在频域上则是周期的。DTFT 可以被看做是傅里叶级数的逆转换。

将函数 $x_n$ 定义在离散点而非连续域内，且需满足有限性或周期性条件。在这种情况下，使用离散傅里叶变换，将函数 $x_n$ 表示为下面的求和形式：

$$
x_n = \frac{1}{N}\sum_{k=0}^{N-1}X_ke^{i\frac{2\pi}{N}kn}, n = 0, ..., N-1

\tag{6}
$$

其中 $X_k$ 是傅里叶振幅，直接利用这个公式计算的时间复杂度为 ${\displaystyle {\mathcal {O}}(n^{2})}$，使用快速傅里叶变换可以将时间复杂度降低到 $\mathcal{O}(n\log n)$。

傅里叶振幅的计算与式 $(6)$ 中的傅里叶逆变换具有对称性：

$$
X_k = \sum_{n=0}^{N-1}x_ne^{-i\frac{2\pi}{N}kn}, k = 0, ..., N-1

\tag{7}
$$

特别地，当 $x_n$ 是实数时，$X_k$ 与 $X_{N-k}$ 共轭。($k = 1, ..., N-1$)。1）$N$ 为偶数时，$X_0$ 与 $X_{N/2}$ 各为其值，且肯定是实数；$N$ 为奇数时，$X_0$ 为实数。

另外还有：

$X_0$ 等于数组 $x$ 各元素之和；当 $N$ 为偶数时，$X_{N/2}=\sum_i[(-1)^ix_i] , i=0,...,N-1$

比如：

$$
x = [1, 2, 3, 4, 5, 6, 7, 8] \\
X = [ 36.0000, -4.0000 + 9.6569i, -4.0000 + 4.0000i,  -4.0000 + 1.6569i,  \\ -4.0000,-4.0000 - 1.6569i,  -4.0000 - 4.0000i,  -4.0000 - 9.6569i]

\tag{8}
$$

我们可以从公式中推出这些结论，根据式 $(6)$，令

$$
\begin{aligned}
    W_N &= e^{-i\frac{2\pi}{N}} \\
    &\xlongequal{欧拉公式} \cos(-\frac{2\pi}{N}) + i\sin(-\frac{2\pi}{N}) \\
    &= \cos(\frac{2\pi}{N}) - i\sin(\frac{2\pi}{N}) \\
\end{aligned}

\tag{9}
$$

式 $(6)$ 可以变成：

$$
X_k = \sum_{n=0}^{N-1}x_nW_N^{kn}, k = 0, ..., N-1

\tag{10}
$$

1. 当 $k=0$ 时，显然 $(10)$ 式结果为所有 $x_n$ 之和。
2. 当 $k=\frac{N}{2}$ 时，$W_N^{kn} = W_N^{\frac{N}{2}n} = (-1)^n$，显然 $(9)$ 式为对 $x_n$ 按权重 $(-1)^n$ 求加权和。
3. $X_{N-k} = \sum_{n=0}^{N-1}x_nW_N^{(N-k)n} = \sum_{n=0}^{N-1}x_n(W_N^{kn})^* = X_{k}$

    其中共轭的证明如下：

    a. 要证 $W_N^{(N-k)n}$ 与 $W_N^{kn}$ 共轭，只需证 $e^{-i\frac{2\pi}{N}(N-k)n}$ 与 $e^{-i\frac{2\pi}{N}kn}$ 共轭

    b. 令 $t_1 = \frac{2\pi}{N}(N-k)n, t_2 = \frac{2\pi}{N}kn$，只需证 $t_1 = -t_2 + 2m\pi$

    c. 显然 $t_1 + t_2 = 2k\pi$，得证

##### 例子

令 ${\displaystyle N=4}, {\displaystyle \mathbf {x} ={\begin{pmatrix}x_{0}\\x_{1}\\x_{2}\\x_{3}\end{pmatrix}}={\begin{pmatrix}1\\2-i\\-i\\-1+2i\end{pmatrix}}}$

我们可以利用 $(7)$ 式计算离散傅里叶变换：

$$
\begin{aligned}
 X_{0}
=&\ e^{-i2\pi 0\cdot 0/4}\cdot 1 \\
&+e^{-i2\pi 0\cdot 1/4}\cdot (2-i) \\
&+e^{-i2\pi 0\cdot 2/4}\cdot (-i) \\
&+e^{-i2\pi 0\cdot 3/4}\cdot (-1+2i) \\
=&\ 2 \\

X_{1}
=&\ e^{-i2\pi 1\cdot 0/4}\cdot 1 \\
&+e^{-i2\pi 1\cdot 1/4}\cdot (2-i) \\
&+e^{-i2\pi 1\cdot 2/4}\cdot (-i) \\
&+e^{-i2\pi 1\cdot 3/4}\cdot (-1+2i) \\
=&\ -2-2i \\

X_{2}
=&\ e^{-i2\pi 2\cdot 0/4}\cdot 1 \\
&+e^{-i2\pi 2\cdot 1/4}\cdot (2-i) \\
&+e^{-i2\pi 2\cdot 2/4}\cdot (-i) \\
&+e^{-i2\pi 2\cdot 3/4}\cdot (-1+2i) \\
=&\ -2i \\

X_{3}
=&\ e^{-i2\pi 3\cdot 0/4}\cdot 1 \\
&+e^{-i2\pi 3\cdot 1/4}\cdot (2-i) \\
&+e^{-i2\pi 3\cdot 2/4}\cdot (-i) \\
&+e^{-i2\pi 3\cdot 3/4}\cdot (-1+2i) \\
=&\ 4+4i

\end{aligned}
$$

从而得到：

$$
{\displaystyle \mathbf {X} ={\begin{pmatrix}X_{0}\\X_{1}\\X_{2}\\X_{3}\end{pmatrix}}={\begin{pmatrix}2\\-2-2i\\-2i\\4+4i\end{pmatrix}}}
$$

##### 离散傅里叶变换的作用

离散傅里叶变换（DFT）非常有用，因为它们揭示了输入数据的周期性以及任何周期性分量的相对强度。然而，在离散傅立叶变换的解释中存在一些微妙之处。通常，**实数**序列的离散傅里叶变换将是具有相同长度的**复数**序列。

<!-- #### 快速傅里叶变换

用 FFT（快速傅里叶变换） 计算 DFT 会得到与直接用 DFT 定义计算相同的结果；最重要的区别是FFT更快。由于舍入误差的存在，许多 FFT 算法还会比直接运用定义求值精确很多。

直接按照这个 $(5)$ 式的定义求值需要 $\mathcal{O}(N^2)$ 次运算：$x_k$ -->

##### `numpy` 实现细节

有许多方法可以定义DFT，在指数的符号，归一化等方面有所不同。在此实现中，DFT定义为

$$
A_k = \sum_{m=0}^{n-1} a_m \exp\left\{-2\pi i{mk \over n}\right\} \qquad k = 0,\ldots,n-1

\tag{11}
$$

上式的输入和输出都是复数，其中每个频率分量的振幅 $a_m = \exp\{2\pi i\,f m\Delta t\}$，其中 $\Delta$ 为采样区间。

返回的结果符合定义的『标准』顺序：如果 `A = fft(a, n)`，那么：

1. `A[0]` 表示零频率项（信号的总和），如果输入为实值，输出也一定为实值
2. `A[1:n/2]` 包含正频率项
3. `A[n/2+1:]`包含负频率项

对于偶数个输入，`A[n/2]` 表示同时表示正和负的奈奎斯特频率，对于实数输入，输出也一定为实数。对于奇数个输入，`A[(n-1)/2]` 包含最大正频率的组分，而 `A[(n+1)/2]` 包含最大负频率的组分。`np.fft.fftfreq(n)` 返回一个数组，给出了输出中相应组分的频率。`np.fft.fftshift(A)` 将 `A` 进行平移，使其中零频率项位于正中间，`np.fft.ifftshift(A)` 则用来撤销此平移操作。

例如，对于正弦波取 8 个样本点，得到如下结果：

```bash
a = [
    0., 0.84147098,
    0.90929743, 0.14112001,
    -0.7568025,  -0.95892427,
    -0.2794155   0.6569866
]
A = [
    0.55373275+0.j,             2.39464696-2.09701186j,
    -1.38668442+0.9155599j,     -0.88104197+0.28041399j,
    -0.80757388+0.j,            -0.88104197-0.28041399j,
    -1.38668442-0.9155599j,     2.39464696+2.09701186j
]
shift_A = [
    -0.80757388+0.j,        -0.88104197-0.28041399j,
    -1.38668442-0.9155599j, 2.39464696+2.09701186j,
    0.55373275+0.j,         2.39464696-2.09701186j,
    -1.38668442+0.9155599j, -0.88104197+0.28041399j
]
freq = [
      0.,     0.125,
    0.25,     0.375,
    -0.5,    -0.375,
   -0.25,    -0.125
]
```

给定输入 `a`，输出及物理意义如下：

- `A = fft(a)`：幅度谱
- `np.abs(A)**2`：功率谱
- `np.angle(A)`：相位谱

另外，逆 $DFT$ 定义为：

$$
a_m = \frac{1}{n}\sum_{k=0}^{n-1}A_k\exp\left\{2\pi i{mk\over n}\right\} \qquad m = 0,\ldots,n-1.

\tag{12}
$$

与正向转换不同之处在于指数中的变量为 $k$，并且需要乘以 $\frac{1}{n}$ 来归一化。

###### 归一化

默认情况下，正弦变换不缩放，而逆变换乘上 $\frac{1}{n}$，通过指定 `norm='ortho'` 参数可以让正向变换和逆向变换都乘上 $\frac{1}{\sqrt{n}}$。

###### 实变换和 Hermitian 变换

当输入全部为实数时，称为 Hermitian 变换，频率为 $f_k$ 的分量是频率为 $-f_k$ 的分量的共轭复数，也就意味着对于实数输入，在负频率分量中的信息全都包含在正频率分量当中了。`rfft` 族函数用来操作实数输入，通过仅计算正频率分量（包括奈奎斯特频率）来利用这种对称性。因此，在这一族函数中，`n` 个输入点产生 `n/2+1` 个复数输出。逆对输入的对称性做同样的假设，输入 `n/2+1` 个点，产生 $n$ 个点的输出。

如果频谱是实数，信号是 Hermitian 的。`hfft` 族函数利用这种对城乡，仅使用 `n/2+1` 个复数点作为时间的输入，就可以产生 `n` 个频率域的实数点。

在高维空间中，FFTs 被用来图像分析和滤波。FFT 的计算效率越好，就越容易计算大的卷积，在时域上使用正确的卷积性质相当于频域的点对点相乘。

###### 高维空间

在二维空闲中，FFT 定义如下：

$$
A_{kl} = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} a_{mn}\exp\left\{-2\pi i \left({mk\over M}+{nl\over N}\right)\right\} \\
\qquad k = 0, \ldots, M-1;\quad l = 0, \ldots, N-1,

\tag{13}
$$

如此往下可定义更高维的 FFT，以及其逆变换。

### wav 文件

#### wav 的格式

来自维基百科的 wav 的简介：

<blockquote>
Waveform Audio File Format（WAVE，又或者是因为扩展名而被大众所知的WAV），是微软与IBM公司所开发在个人计算机存储音频流的编码格式，在Windows平台的应用软件受到广泛的支持，地位上类似于麦金塔计算机里的AIFF。此格式属于资源交换档案格式(RIFF)的应用之一，通常会将采用脉冲编码调制的音频资存储在区块中。也是其音乐发烧友中常用的指定规格之一。由于此音频格式未经过压缩，所以在音质方面不会出现失真的情况，但因而文件的体积在众多音频格式中较大。
</blockquote>

#### wav 的读取

在 python 3 中可以使用 [`wave` 模块](https://docs.python.org/3/library/wave.html)来读取和处理 wav 文件。

在实验中封装好了一个 `wav_helper.py` 脚本，用来将 `wav` 文件转换为 `svg` 图片。

#### wav 转换结果

<object data="./frames/wavs.txt" width="400" height="340"></object>

##### `crane_bump.wav`

![crane_bump.wav](./figures/crane_bump.wav.svg)

由于帧数太多，看不出什么规律，中间略有稀疏。

##### `engine.wav`

![engine.wav](./figures/engine.wav.svg)

很明显的周期波，有多个波峰。

##### `guitartune.wav`

![guitartune.wav.svg](./figures/guitartune.wav.svg)

同样由于帧数太多，无法用肉眼分辨。

#### 幅度谱

针对上述三个波形，分别绘制幅度谱、相位谱。幅度谱是幅度随频率变化的图像，参见下图，从右侧看过去就是幅度谱：

<center>

![time-frequency](https://upload.wikimedia.org/wikipedia/commons/6/61/FFT-Time-Frequency-View.png)

FFT 的时域和频域视图，来自于维基百科

</center>

需要注意的是，这里的每个音频都是信号值恒大于 0 的，如果直接绘制，会发现零频率分量的值（由上文的证明可知，零频率分量的值就是所有信号值的和）远远大于其他频率的值，这在平时遇到的例如 $\sin(x) + \sin(2x)$ 这种均值为 0 的波中是不会出现的。为了绘图方便，我们绘制之前将信号每个点减去均值，这样最终的均值就是 0 了，得出的零频率分量的值也会是 0。

##### `crane_bump.wav-amplitude.svg`

![crane_bump.wav-amplitude.svg](./figures/crane_bump.wav-amplitude.svg)

##### `engine.wav-amplitude.svg`

![engine.wav-amplitude.svg](./figures/engine.wav-amplitude.svg)

##### `guitartune.wav-amplitude.svg`

![guitartune.wav-amplitude.svg](./figures/guitartune.wav-amplitude.svg)

#### 相位谱

相位谱则是相位随着频率变化的图像，可以算作 FFT 三维视图中的俯视图。

##### `crane_bump.wav-phase.svg`

![crane_bump.wav-phase.svg](./figures/crane_bump.wav-phase.svg)

##### `engine.wav-phase.svg`

![engine.wav-phase.svg](./figures/engine.wav-phase.svg)

##### `guitartune.wav-phase.svg`

![guitartune.wav-phase.svg](./figures/guitartune.wav-phase.svg)

### bmp 文件

#### bmp 的格式

来自维基百科的介绍：

<blockquote>
BMP取自位图BitMaP的缩写，也称为DIB（与设备无关的位图），是一种与显示器无关的位图数字图像文件格式。常见于微软视窗和OS/2操作系统，Windows GDI API内部使用的DIB数据结构与 BMP 文件格式几乎相同。

图像通常保存的颜色深度有2（1位）、16（4位）、256（8位）、65536（16位）和1670万（24位）种颜色（其中位是表示每点所用的数据位）。8位图像可以是索引彩色图像外，也可以是灰阶图像。表示透明的alpha通道也可以保存在一个类似于灰阶图像的独立文件中。带有集成的alpha通道的32位版本已经随着Windows XP出现，它在视窗的登录和主题系统中都有使用。
</blockquote>

#### bmp 的读取

使用 [matplotlib.pyplot.imread](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imread.html) 可以将 `bmp` 文件读取成 `numpy.array`，但是需要安装 [Pillow](https://pillow.readthedocs.io/en/latest/)：

```bash
pip install Pillow
```

之后就可以使用 `pyplot.imread(filename, 'bmp')` 来读取 `bmp` 文件了。

#### bmp 转换结果

##### 原始图片

![LENA.BMP](./bmp/LENA.BMP)
![Sonic.bmp](./bmp/Sonic.bmp)

##### 幅度谱

![LENA.BMP-amplitude.svg](./figures/LENA.BMP-amplitude.svg)
![Sonic.bmp-amplitude.svg](./figures/Sonic.bmp-amplitude.svg)

##### 相位谱

![LENA.BMP-phase.svg](./figures/LENA.BMP-phase.svg)
![Sonic.bmp-phase.svg](./figures/Sonic.bmp-phase.svg)

##### 仅对幅度进行逆傅里叶变换

![LENA.BMP-inverse-by-amplitude.svg](./figures/LENA.BMP-inverse-by-amplitude.svg)
![Sonic.bmp-inverse-by-amplitude.svg](./figures/Sonic.bmp-inverse-by-amplitude.svg)

可以看到，得到的信息在每个点的强度基本上是一样的，没有任何价值。

##### 仅对相位进行逆傅里叶变换

![LENA.BMP-inverse-by-phase.svg](./figures/LENA.BMP-inverse-by-phase.svg)
![Sonic.bmp-inverse-by-phase.svg](./figures/Sonic.bmp-inverse-by-phase.svg)

可以看到，原始图片的轮廓依稀可见，这可以证明：【相位】包含了图片最主要的信息。

##### 同时使用幅度、相位进行逆傅里叶变换

![LENA.BMP-inverse.svg](./figures/LENA.BMP-inverse.svg)
![Sonic.bmp-inverse.svg](./figures/Sonic.bmp-inverse.svg)

可以看到，图片完全恢复原样。

### 正余弦信号之和

使用正弦合成信号：

$$

\sin(x) + \sin(3x) + ... + \sin(5x) + ... + \sin(19x)

\tag{14}
$$

#### 原始信号

![sum-of-sin.svg](./figures/sum-of-sin.svg)

#### 幅度谱

![sum-of-sin-amplitude.svg](./figures/sum-of-sin-amplitude.svg)

可以看到切好划分为了 10 个频率分量，这与 $(14)$ 的构成是相符的。幅度值大体相当。

#### 相位谱

![sum-of-sin-phase.svg](./figures/sum-of-sin-phase.svg)

可以看到，在有分量的频率值处，相位会突然增大并保持一段频率距离，最后回到较小值。

### 加窗口的傅里叶变换

有如下信号：

$$
x(t) = \sin(2\pi f_1t) + \sin(2\pi f_2t) + \sin(2\pi f_3t)\\
 f_1 = 20Hz, f_2 = 30Hz, f_3 = 40Hz

\tag{15}
$$

取采样频率 $f_s = 100Hz$ 对 $x(t)$ 进行等间隔采样得到 $x(k)$，然后对 $x(k)$ 加长度为 128 的矩形窗口进行截断得到 $x_1(k)$，

1. 对 $x_1(k)$ 进行 128 点的 DFT ，画出此时信号的频谱图
2. 对 $x_1(k)$ 进行 512 点的DFT ，画出此时信号的频谱图
3. 若加窗长度为 512 ，再进行 512 点的DFT ，此时信号的频谱图如何？

采样方案如下：

1. 令时间 `t = np.linspace(0, 5.11, num=512)`，这样将会在 0 到 5.11s 的时间内采 512 个信号点，刚好满足 $f_s = 100Hz$
2. 利用 $(15)$ 模拟计算出采样结果 $x$
3. 将 $x$ 加长度为 128 的窗口，得到 $x_1$

得到采样结果后，对其进行相应的傅里叶变换即可。

#### 窗口变换

对于下面的方波，可以看到进行傅里叶变换之后的旁瓣很多：

![rectangular-puls](https://www.oreilly.com/library/view/elegant-scipy/9781491922927/assets/elsp_0413.png)

理论上也可以证明，方波需要无穷多个不同频率的正弦波组合而成，它们所占的比例也就是幅度值，是我们上图中看到的脉冲高度。

DFT 假设输入信号是具有周期性的。如果输入不具有周期性，假设就会变为了在信号结束之后，后续会跳到信号最开始的输入进行重复。如下图所示：

![dft-assumption](https://www.oreilly.com/library/view/elegant-scipy/9781491922927/assets/elsp_0414.png)

我们只采样了 $T_{eff}$ 时间段的信号，傅里叶变换会假设 $x(8) = x(0)$，假设的信号如虚线所示，而不是实线表示的实际情况。这就会造成边界处大的跳变，因此在频谱中会出现震荡：

```python
t = np.linspace(0, 1, 500)
x = np.sin(49 * np.pi * t)

X = fftpack.fft(x)

f, (ax0, ax1) = plt.subplots(2, 1)

ax0.plot(x)
ax0.set_ylim(-1.1, 1.1)

ax1.plot(fftpack.fftfreq(len(t)), np.abs(X))
ax1.set_ylim(0, 190);
```

在这个例子中信号函数为：

$$
x = \sin(19\pi t), t = [0, 1]

\tag{16}
$$

周期 $T = \frac{2\pi}{19\pi} = \frac{2}{19}$ 采样宽度 $\Delta t = 1 = \frac{19}{2}T = 9.5T$ ，出现了零头（$0.5T$），没有满足周期性，所以会出现震荡。

![oscillation](https://www.oreilly.com/library/view/elegant-scipy/9781491922927/assets/elsp_0415.png)

我们可以通过增加窗口来抵消这种影响，我们将原始函数乘以一个窗口函数，例如 Kaiser 窗口 $K(N, \beta)$。这里我们将 $\beta$ 取 $0 - 5$ 进行可视化。

![kaiser-window](https://www.oreilly.com/library/view/elegant-scipy/9781491922927/assets/elsp_0416.png)

通过改变 $\beta$ 的大小，我们可以将窗口的形状从矩形（$\beta = 0$, 无窗口）变为在采样区间内从零平滑上升到最高点再平滑下降到零的图像，这样产生的旁瓣很小。

通过乘以 Kaiser 窗口，旁瓣减少了许多，但代价是主瓣变宽：

```python
win = np.kaiser(len(t), 5)
X_win = fftpack.fft(x * win)

plt.plot(fftpack.fftfreq(len(t)), np.abs(X_win))
plt.ylim(0, 190)
```

![kaiser-windowing-result](https://www.oreilly.com/library/view/elegant-scipy/9781491922927/assets/elsp_0417.png)

#### 矩形窗

矩形窗主瓣窄，旁瓣大，频率识别精度最高，幅值识别精度最低，如果仅要求精确读出主瓣频率，而不考虑幅值精度，则可选用矩形窗，例如测量物体的自振频率等；布莱克曼窗主瓣宽，旁瓣小，频率识别精度最低，但幅值识别精度最高；如果分析窄带信号，且有较强的干扰噪声，则应选用旁瓣幅度小的窗函数，如汉宁窗、三角窗等；对于随时间按指数衰减的函数，可采用指数窗来提高信噪比。

对于强度相当的正弦曲线，矩形窗口具有出色的分辨率特性，但对于不同幅度的正弦曲线，它是一个不好的选择。该特性有时被描述为低动态范围。

根据维基百科，窗口大小为 n 的矩形窗只需要保留中间的 n 个信号值，其余全部置零即可。

矩形窗可以使用 [scipy.signal.boxcar](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.boxcar.html) 生成。文档中写到：

> Included for completeness, this is equivalent to no window at all.

下面是长度为 51 的矩形窗口的时域图像和频域图像：

![rect-51](https://docs.scipy.org/doc/scipy-0.14.0/reference/_images/scipy-signal-boxcar-1_00.png)

![rect-51-fft](https://docs.scipy.org/doc/scipy-0.14.0/reference/_images/scipy-signal-boxcar-1_01.png)

#### 变换结果

##### 窗口长度 = 128

原始信号：

![boxcar-128.svg](./figures/boxcar-128.svg)

幅度谱：

![boxcar-128-amplitude.svg](./figures/boxcar-128-amplitude.svg)

相位谱：

![boxcar-128-phase.svg](./figures/boxcar-128-phase.svg)

##### 窗口长度 = 512

原始信号：

![boxcar-512.svg](./figures/boxcar-512.svg)

幅度谱：

![boxcar-512-amplitude.svg](./figures/boxcar-512-amplitude.svg)

相位谱：

![boxcar-512-phase.svg](./figures/boxcar-512-phase.svg)

## 低通滤波器

差分方程用来表示 1，2，3，4 阶低通和高通滤波器，以及 2， 4，6 阶带通，带阻和陷波滤波器以及共振补偿（RES_COMP）滤波器。

最简单的低通滤波器由下面的差分方程给出：

$$
y(n) = x(n) + x(n-1)

\tag{17}
$$

其中 $x(n)$ 是在时间 $n$ 的输入幅度，$y(n)$ 是在时间 $n$ 的输出幅度。此滤波器的信号流图如下所示，其中 $z^{-1}$ 意味着采样前推 1 个，即 $z^{-1}x(n) = x(n-1)$：

![signal-flow-graph-low-pass](https://ccrma.stanford.edu/~jos/fp/img92.png)

更一般的滤波器的差分形式为：

$$
\sum_{k=0}^{N}a_ky(n - k) = \sum_{k=0}^{M}b_kx(n-k)

\tag{18}
$$

对每一项进行离散傅里叶变换，得到：

$$
a_0Y(\Omega) + a_1e^{-j\Omega}Y(\Omega) + a_2e^{-j2\Omega}Y(\Omega) + ... + a_Ne^{-jN\Omega}Y(\Omega) \\
= b_0X(\Omega) + b_1e^{-j\Omega}X(\Omega) + b_2e^{-j2\Omega}X(\Omega) + ... + b_Ne^{-jM\Omega}X(\Omega) \\

\tag{19}
$$

得到：

$$
H(\Omega) = \frac{Y(\Omega)}{X(\Omega)} = \frac{b_0 + b_1e^{-j\Omega} + b_2e^{-j2\Omega} + ... + b_Me^{-jM\Omega}}{a_0 + a_1e^{-j\Omega} + a_2e^{-j2\Omega} + ... + a_Ne^{-jN\Omega}}

\tag{20}
$$

称 $H(\Omega)$ 为滤波器的**频率响应**。$H(\Omega)$ 是复数，可以用极坐标形式表示 $H(\Omega) = |H(\Omega)|e^{j\theta\Omega}$，$|H(\Omega)|$ 表示滤波器在数字频率 $\Omega$ 处的增益，$\theta(\Omega)$ 是相位差。

### 幅度谱和相位谱

给定三阶低通滤波器：

$$
y(n) + (-1.76)y(n-1) + 1.1829y(n-2) - 0.2781y(n-3) \\
= 0.0181x(n) + 0.0543x(n-1) + 0.0543x(n-2) + 0.0181x(n)

\tag{21}
$$

$$
\begin{aligned}
    \mathrm{\mathbf{a}} &= (1, -1.76, 1.1829, -0.2781) \\
\mathrm{\mathbf{b}} &= (0.0181, 0.0543, 0.0543, 0.0181) \\
\end{aligned}
\tag{22}
$$

我们根据 $(20)$ 式就可以计算出 $H(\Omega)$ 了，根据计算结果分别绘制幅度谱和相位谱即可。

![low-filter-amplitude.svg](./figures/low-filter-amplitude.svg)

![low-filter-phase.svg](./figures/low-filter-phase.svg)

### 低通、高通、带通

理解低通、高通、带通的频率特性：

- 低通：允许通过低频分量，削减高频分量
- 高通：允许通过高频分量，削减低频分量
- 带通：允许通过指定频率范围内的分量，削减其他分量

### 例子

对于二维图片，如果：

$$
\begin{aligned}
    y(i, j) &= 0.5x(i-1, j) + 0.5x(i+1, j) \\
&\quad + 0.5x(i, j-1) + 0.5x(i, j+1)
\end{aligned}

\tag{23}
$$

在数值上，这是一种"平滑化"。在图形上，就相当于产生"模糊"效果，"中间点"失去细节。

在频率上，通过取平均将高频分量减少，因此是低通滤波器。

### 高斯白噪声

生成 $M = 100$ 个长度为 $N = 1000$ 的高斯白噪声随机序列 $\mathrm{\mathbf{X}}_{M\times N}$ ，选择其中的 $\mathrm{\mathbf{X}}_0$ 与 100 个序列做相关性，画出其示意图。

![white-noise-correlation.svg](./figures/white-noise-correlation.svg)

可以看到自身和自身是相关性最大的。

- 白噪声的自相关函数满足 $R_X(\tau) = \frac{N_0}{2}\delta(\tau)$

    ![noise-transform](https://i1.wp.com/www.gaussianwaves.com/gaussianwaves/wp-content/uploads/2013/11/Wiener-Khinchin-Theorem.png?w=615&ssl=1)

- 功率谱密度均匀分布在 $(-\infty, +\infty)$ 的整个频率区间
- 应用参见维基百科：
    <blockquote>
    白噪声的应用领域之一是建筑声学，为了减弱内部空间中分散人注意力并且不希望出现的噪声（如人的交谈），使用持续的低强度噪声作为背景声音。一些紧急车辆的警报器也使用白噪声，因为白噪声能够穿过如城市中交通噪声这样的背景噪声并且不会引起反射，所以更加容易引起人们的注意。

    在电子音乐中也有白噪声的应用，它被直接或者作为滤波器的输入信号以产生其它类型的噪声信号，尤其是在音频合成中，经常用来重现类似于铙钹这样在频域有很高噪声成分的打击乐器。

    白噪声也用来产生冲激响应。为了在一个演出地点保证音乐会或者其它演出的均衡效果，从PA系统发出一个瞬间的白噪声或者粉红噪声，并且在不同的地方监测噪声信号，这样工程师就能够建筑物的声学效应能够自动地放大或者削减某些频率，从而就可以调整总体的均衡效果以得到一个平衡的和声。

    白噪声可以用于放大器或者电子滤波器的频率响应测试，有时它与响应平坦的话筒或和自动均衡器一起使用。这个设计的思路是系统会产生白噪声，话筒接收到扬声器产生的白噪声，然后在每个频率段进行自动均衡从而得到一个平坦的响应。这种系统用在专业级的设备、高端的家庭立体声系统或者一些高端的汽车收音机上。

    白噪声也作为一些随机数字生成器的基础使用。

    白噪声也可以用于审讯前使人迷惑，并且可能用于感觉剥夺技术的一部分。上市销售的白噪声机器产品有私密性增强器、睡眠辅助器以及掩饰耳鸣。
    </blockquote>

### 图像对齐

- 给定两幅有相对偏移的图像，思考如何对其进行对齐操作？

    在频域上，很容易看出相频率的幅度值，可以将两者的相位谱（与幅度谱相比，包含了最主要信息）互相进行补全，之后再结合幅度信息进行逆傅里叶变换即可。

- 对一幅图像计算其幅度和相位谱，然后单纯只用其一进行逆变换，其结果如何？

    前文已经看到了，只用幅度原有信息几乎全部丢失，只用相位谱则还可以较好地还原

- 两幅图像有水平和垂直方向上的偏移，请问如何将其对齐？

    相位谱按照频率对齐即可

## 视频中的高斯背景建模

### 混合高斯模型

混合高斯模型在实验一中已经完成过了。有两类水果，它们的水分、含糖率、营养成分服从不同的高斯分布，构成了两个高斯分量。通过 EM 算法我们可以将这两个分量分离出来，也就实现了参数的估计。

### 视频处理

- 根据初始的一些视频帧，对其中的像素点进行混合高斯建模
- 然后根据模型，对新出现的视频帧进行模型更新，并对像素点进行判断，是前景像素点还是背景像素点

将问题简化，只考虑前后两幅图片的处理。在前一幅图片已经将照片分为前景和背景的情况下，对下一幅图片中的像素点进行归类。

例如第一副图是这样及其分类结果如下：

![20181207043159.png](https://i.loli.net/2018/12/07/5c098753797e4.png)
![20181207043245.png](https://i.loli.net/2018/12/07/5c09876e9363a.png)

在下一帧中，对每个像素点寻找距离最近的类（前景、背景）归入即。

将新的帧分为前景、背景，重复以上操作。

理论上跟水果分类问题是完全一样的，只是需要动态考虑前后的变化而已了。

### 生成图片

安装好 `ffmpeg`，执行 `make avi2png` 将视频转换为图片：

```powershell
Input #0, avi, from 'visiontraffic.avi':
  Metadata:
    encoder         : Lavf52.31.0
  Duration: 00:00:17.72, start: 0.000000, bitrate: 11475 kb/s
    Stream #0:0: Video: mjpeg (Baseline) (MJPG / 0x47504A4D), yuvj420p(pc, bt470bg/unknown/unknown), 640x360, 11488 kb/s, 29.97 fps, 29.97 tbr, 29.97 tbn, 29.97 tbc
Stream mapping:
  Stream #0:0 -> #0:0 (mjpeg (native) -> png (native))
Press [q] to stop, [?] for help
[swscaler @ 00000241926aeb80] deprecated pixel format used, make sure you did set range correctly
Output #0, image2, to 'video-frame%05d.png':
  Metadata:
    encoder         : Lavf58.23.102
    Stream #0:0: Video: png, rgb24, 640x360, q=2-31, 200 kb/s, 29.97 fps, 29.97 tbn, 29.97 tbc
    Metadata:
      encoder         : Lavc58.41.101 png
frame=  531 fps= 17 q=-0.0 Lsize=N/A time=00:00:17.71 bitrate=N/A speed=0.568x
video:195496kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown
```

一共生成了 531 张图片，其中第 1 张和第 150 张分别为：

![./avi/video-frame00001.png](./avi/video-frame00001.png)
![./avi/video-frame00150.png](./avi/video-frame00150.png)

### 均值聚类

使用 `scipy.cluster.vq.kmeans` 对第一张图片进行均值聚类。

<object data="./png_diff/png_diff.txt" width="500"></object>

### 分离结果

利用第 1 张图片计算出的中心点，对第 150 张图片进行前背景分离得到的结果如下：

![png_diff.svg](./figures/png-diff.svg)

## 参考文献

1. [WAV | Wikipedia](https://en.wikipedia.org/wiki/WAV)
2. [Fourier transform | Wikipedia](https://en.wikipedia.org/wiki/Fourier_transform#Definition)
3. [Discrete Fourier transform | Wikipedia](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
4. [Discrete Fourier Transform | Wolfram](http://mathworld.wolfram.com/DiscreteFourierTransform.html)
5. [复数形式傅里叶变换的物理意义 | 知乎](https://www.zhihu.com/question/20977844)
6. [傅里叶变换的感性认识 | 知乎](https://zhuanlan.zhihu.com/p/19763358)
7. [实数序列离散傅里叶变换（DFT）的共轭对称性质 | CSDN](https://blog.csdn.net/jbb0523/article/details/6668533)
8. [Blurring an image with a two-dimensional FFT](https://scipython.com/book/chapter-6-numpy/examples/blurring-an-image-with-a-two-dimensional-fft/)
9. [图像傅里叶变换的幅度谱和相位谱的以及反变换](https://blog.csdn.net/sinat_32290679/article/details/72793584)
10. [什么是窗函数](https://zhuanlan.zhihu.com/p/24318554)
11. [有关信号处理中的窗函数](https://blog.csdn.net/Qsir/article/details/78247179)
12. [Boxcar function](https://en.wikipedia.org/wiki/Boxcar_function)
13. [Window_function#Rectangular_window | Wikipedia](https://en.wikipedia.org/wiki/Window_function#Rectangular_window)
14. [从傅里叶变换到加窗傅里叶变换](https://blog.csdn.net/Yuejiang_Li/article/details/78762201)
15. [Elegant SciPy by Harriet Dashnow, Stéfan van der Walt, Juan Nunez-Iglesias - Chapter 4](https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html)
16. [Low-pass filter](https://en.wikipedia.org/wiki/Low-pass_filter)
17. [Difference Equations and Digital Filters](http://www.mee.tcd.ie/~corrigad/3c1/DSP1_2012_students.pdf)
18. [Digital Filter Design Writing Difference Equations For Digital Filters](http://www.apicsllc.com/apics/Sr_3/Sr_3.htm)
19. [Definition of the Simplest Low-Pass](https://ccrma.stanford.edu/~jos/fp/Definition_Simplest_Low_Pass.html)
20. [傅里叶变换与滤波器形状](http://blog.neu.edu.cn/luanfeng/files/2015/10/DSP07.pdf)
21. [信号处理——滤波器](https://blog.csdn.net/VictoriaW/article/details/62233462)
22. [高斯模糊的算法](http://www.ruanyifeng.com/blog/2012/11/gaussian_blur.html)
23. [Simulation and Analysis of White Noise in Matlab](https://www.gaussianwaves.com/2013/11/simulation-and-analysis-of-white-noise-in-matlab/)