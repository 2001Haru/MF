下面是一版比较清晰的整理，可以看成我们这个想法的 **研究草稿 / proposal skeleton**。我会先固定符号，再讲动机、数学形式、方法设计、风险和实验方案。

---

# Plug-in $x_1$-Prediction for Euler MeanFlow

## 0. 约定

这里沿用 Euler MeanFlow 里的记号：

$$

x_t=(1-t)x_0+t x_1,\qquad t\in[0,1],

$$

其中：

$$

x_0=\text{noise endpoint},\qquad x_1=\text{data endpoint}.

$$

所以 $t=0$ 是噪声端，$t=1$ 是数据端。这个约定和很多 diffusion 论文里 “$x_0$ 是数据” 的习惯相反，后面不要混淆。

---

# 1. 背景：从 MeanFlow 到 Euler MeanFlow

MeanFlow 学的是平均速度场：

$$

u_{t\to r}(x)

\frac{\phi_{t\to r}(x)-x}{r-t},

$$

其中 $\phi_{t\to r}(x)$ 是从时间 $t$ 到时间 $r$ 的 flow map。

因此：

$$

\phi_{t\to r}(x)

x+(r-t)u_{t\to r}(x).

$$

原始 MeanFlow 的训练需要类似：

$$

\frac{d}{dt}u_{t\to r}(x_t)

\partial_t u+\nabla_xu\cdot v_t

$$

这样的 JVP 项。

Euler MeanFlow 的思想是：不用 JVP，而是用一个小时间步 $\Delta t$ 的有限差分近似：

$$

\frac{d}{dt}u_{t\to r}(x_t)
\approx
\frac{
u_{t+\Delta t\to r}(x_{t+\Delta t})

u_{t\to r}(x_t)
}{
\Delta t
}.

$$

其中：

$$

x_{t+\Delta t}
\approx
x_t+\Delta t,u_{t\to t}(x_t).

$$

所以 EMF 可以看作：

$$

\text{MeanFlow identity}
+
\text{Euler finite difference}

$$

即一种 **JVP-free MeanFlow**。

---

# 2. 为什么 $x_1$-prediction 很重要？

直接预测平均速度 $u_{t\to r}$ 在数学上自然，但在图像生成里未必是好的神经网络输出目标。

原因是：

$$

u_{t\to r}

$$

是速度/平均速度，不一定长得像自然图像，也不一定在数据流形附近。尤其在 pixel-space generation 或 SDF 任务中，直接预测 velocity-like quantity 可能很难。

因此 Euler MeanFlow 后文引入了 $x_1$-prediction 版本。

它定义：

$$

\tilde{x}_{t\to r}(x) = 

x+(1-t)u_{t\to r}(x).

$$

也就是说，不直接输出 $u_{t\to r}$，而是输出一个 endpoint-like quantity。

反过来：

$$

u_{t\to r}(x) = 

\frac{\tilde{x}_{t\to r}(x)-x}{1-t}.

$$

真正的 flow map 也可以恢复为：

$$

\phi_{t\to r}(x) = 

x+\frac{r-t}{1-t}\left(\tilde{x}_{t\to r}(x)-x\right).

$$

所以 $x_1$-prediction 并不是直接预测 $x_r$，而是用一个更像数据端点的变量来参数化平均速度。

---

## 关键边界性质

当 $r=t$ 时：

$$

u_{t\to t}(x_t)=v_t(x_t),

$$

于是：

$$

\tilde{x}_{t\to t}(x_t)

x_t+(1-t)v_t(x_t).

$$

在线性路径下，条件速度为：

$$

v_{t|1}=x_1-x_0.

$$

又因为：

$$

x_t=(1-t)x_0+t x_1,

$$

所以：

$$

x_t+(1-t)(x_1-x_0)=x_1.

$$

因此：

$$

\tilde{x}_{t\to t}(x_t)=x_1.

$$

这说明 $x_1$-prediction EMF 在 $r=t$ 时自然退化成 clean-data prediction：

$$

\tilde{x}_{t\to t}^{\theta}(x_t)\approx x_1.

$$

这比直接预测 $u$ 更符合图像数据流形。

---

# 3. Plug-in velocity 的本质

Shortcutting 文章里的 plug-in velocity 试图解决一个问题：

真实 marginal velocity 是：

$$

v_t(x_t)=\mathbb E[v_{t|1}\mid x_t],

$$

但训练时通常只能用单个条件速度：

$$

v_{t|1}=x_1-x_0.

$$

这个条件速度是对 $v_t(x_t)$ 的无偏样本，但方差很大。

Plug-in velocity 用 batch 中的多个样本来近似 posterior expectation。对给定 $x_t=x$，batch 中每个数据样本 $x_1^{(i)}$ 都被看作可能的 clean endpoint。在线性路径下：

$$

x_t\mid x_1^{(i)}
\sim
\mathcal N(t x_1^{(i)},(1-t)^2I).

$$

于是定义权重：

$$

\pi_i

\frac{
\exp\left(
-\frac{|x_t-tx_1^{(i)}|^2}{2(1-t)^2}
\right)
}{
\sum_j
\exp\left(
-\frac{|x_t-tx_1^{(j)}|^2}{2(1-t)^2}
\right)
}.

$$

Plug-in velocity 本质上是：

$$

\hat v_B(x_t)

\sum_i \pi_i v_{t|1}^{(i)}.

$$

它是 batch empirical distribution 下的 marginal velocity estimator。

它降低方差，但会引入偏置，因为它估计的是：

$$

v_t(x_t;p_B)

$$

而不是：

$$

v_t(x_t;p_{\text{data}}).

$$

其中 $p_B$ 是 batch empirical distribution。

---

# 4. 我们的核心想法：Plug-in $x_1$-Prediction

既然：

$$

\tilde{x}_{t\to t}(x_t)

x_t+(1-t)v_t(x_t),

$$

那么 marginal velocity 对应的 endpoint posterior mean 是：

$$

m_t(x_t)

x_t+(1-t)v_t(x_t).

$$

而由于：

$$

x_t+(1-t)v_{t|1}=x_1,

$$

所以：

$$

m_t(x_t)

\mathbb E[x_1\mid x_t].

$$

这一步很关键：

$$

\boxed{
\text{marginal velocity}
\quad\Longleftrightarrow\quad
\text{posterior mean endpoint}
}

$$

因此，plug-in velocity 自然对应一个 plug-in endpoint estimator：

$$

\hat m_B(x_t)

x_t+(1-t)\hat v_B(x_t).

$$

代入 plug-in velocity 的 batch 加权形式，可以得到更直接的表达：

$$

\boxed{
\hat m_B(x_t)

\sum_i \pi_i x_1^{(i)}.
}

$$

这就是我们提出的 **plug-in $x_1$-prediction**。

它不是先估计速度再变换，而是直接在 endpoint space 里估计：

$$

\mathbb E[x_1\mid x_t].

$$

---

# 5. 为什么这可能有价值？

原始 $x_1$-prediction EMF 使用单个训练样本的 $x_1$ 作为监督：

$$

\tilde{x}_{t\to t}^{\theta}(x_t)\approx x_1.

$$

但从 MSE 的角度看，真正的 Bayes optimal predictor 是：

$$

\mathbb E[x_1\mid x_t],

$$

而不是某一个随机采样的 $x_1$。

因此，单样本 $x_1$ target 是：

$$

\text{unbiased but high-variance}.

$$

Plug-in endpoint target 是：

$$

\hat m_B(x_t)=\sum_i\pi_i x_1^{(i)},

$$

它是：

$$

\text{biased but lower-variance}.

$$

所以它和 plug-in velocity 一样，本质上是在做 bias-variance tradeoff。

但相比 plug-in velocity，plug-in $x_1$-prediction 有一个额外优势：

$$

\hat m_B

$$

处在 endpoint / image-like 坐标里，更适合 $x_1$-prediction EMF 的输出空间。

---

# 6. 不应该直接全量替换：需要 shrinkage

直接把 $x_1$ 全部替换成：

$$

\hat m_B=\sum_i\pi_i x_1^{(i)}

$$

可能有风险。

因为 posterior mean 可能是多个图像模式的平均，容易产生 off-manifold 或 blurry target。

因此更合理的是使用 shrinkage：

$$

\boxed{
m_{\lambda}

(1-\lambda)x_1+\lambda\hat m_B.
}

$$

其中：

$$

\lambda\in[0,1].

$$

解释：

* $\lambda=0$：退化为原始 $x_1$-prediction；
* $\lambda=1$：完全使用 plug-in endpoint；
* $0<\lambda<1$：在单样本 endpoint 和 batch posterior endpoint 之间折中。

这比 Shortcutting 文章里的 Bernoulli 替换更平滑。Bernoulli 替换是随机二选一，而 shrinkage 是确定性连续插值。

---

# 7. 自适应 $\lambda$：用 ESS 衡量 plug-in 可信度

Plug-in endpoint 的可靠性取决于 posterior weights 是否集中。

定义：

$$

\mathrm{ESS} = 

\frac{1}{\sum_i\pi_i^2}.

$$

如果 $\mathrm{ESS}$ 很小，说明 batch posterior 几乎由一两个样本支配，此时 $\hat m_B$ 不可靠。

如果 $\mathrm{ESS}$ 较大，说明 batch 中有多个合理候选，plug-in posterior mean 更稳定。

因此可以令：

$$

\lambda(x_t,t) = 

\operatorname{clip}
\left(
\frac{\mathrm{ESS}(x_t,t)-1}{B-1},
0,1
\right).

$$

最终 target 为：

$$

\boxed{
m_{\text{target}} = 

(1-\lambda(x_t,t))x_1
+
\lambda(x_t,t)\hat m_B(x_t).
}

$$

这就是 **ESS-adaptive plug-in $x_1$-prediction**。

---

# 8. 温度化 posterior weights

高维图像 latent 中，原始权重：

$$

\pi_i
\propto
\exp
\left(
-\frac{|x_t-tx_1^{(i)}|^2}{2(1-t)^2}
\right)

$$

容易 collapse。

可以加入温度：

$$

\pi_i^{(\tau)}
\propto
\exp
\left(
-\frac{|x_t-tx_1^{(i)}|^2}{2\tau(1-t)^2}
\right),
\qquad \tau\ge 1.

$$

$\tau$ 越大，posterior 越平滑，ESS 越高；但 bias 也可能增加。

可以固定 $\tau$，也可以让 $\tau$ 自适应，使 ESS 达到某个目标范围：

$$

\mathrm{ESS}_{\tau}(x_t,t)\ge k.

$$

这样可以避免 plug-in endpoint 被单个 batch 样本支配。

---

# 9. 插入 $x_1$-EMF loss 的方式

原始 $x_1$-EMF 的监督大致是：

$$

\tilde{x}_{t\to r}^{\theta}(x_t)
\approx
x_1
+
\text{Euler consistency correction}.

$$

我们的替换是：

$$

x_1
\quad\longrightarrow\quad
m_{\text{target}}.

$$

因此可以写成：

$$

\mathcal L_{\text{Plug-}x_1}

\mathbb E
\left[
\left|
\tilde{x}^{\theta}_{t\to r}(x_t)

\operatorname{sg}
\left(
m_{\text{target}}
+
C(t,r,\Delta t)
\frac{
\tilde{x}^{\theta}_{t+\Delta t\to r}(x_{t+\Delta t})

\tilde{x}^{\theta}_{t\to r}(x_t)
}{
\Delta t}
\right)
\right|^2
\right],

$$

其中 (C(t,r,\Delta t)) 是 EMF 原公式中的时间系数。

同时，用 plug-in endpoint 来推进局部 Euler step：

$$

x_{t+\Delta t} = 

x_t+
\frac{\Delta t}{1-t}
\left(
m_{\text{target}}-x_t
\right).

$$

因为：

$$

\hat v(x_t) = 

\frac{m_{\text{target}}-x_t}{1-t}.

$$

也就是说，plug-in $x_1$ 可以出现在两个位置：

1. **anchor target**：替代 $r=t$ 时的 $x_1$ 监督；
2. **Euler local step**：替代局部推进方向中的条件速度。

这两个位置可以分别 ablate。

---

# 10. CFG 下的 plug-in $x_1$

在 class-conditional generation 中，不能简单地在混类别 batch 上做 plug-in average，否则会稀释类别信号。

更合理的是估计两个 endpoint posterior mean：

$$

m_t^c(x) = 

\mathbb E[x_1\mid x_t=x,c],

$$

以及：

$$

m_t^\emptyset(x) = 

\mathbb E[x_1\mid x_t=x].

$$

batch 版本：

$$

\hat m_B^c(x) = 

\sum_{i:c_i=c}\pi_i^c x_1^{(i)},

$$

$$

\hat m_B^\emptyset(x) = 

\sum_i\pi_i^\emptyset x_1^{(i)}.

$$

然后在 endpoint space 中做 CFG：

$$

m_{\text{cfg}} = 

\omega \hat m_B^c
+
(1-\omega)\hat m_B^\emptyset.

$$

这是合法的，因为：

$$

x_t+(1-t)v_{\text{cfg}} = 

\omega m_c+(1-\omega)m_\emptyset.

$$

所以 velocity-space CFG 可以等价转成 endpoint-space CFG。

---

# 11. 和 SoFlow 的关系

SoFlow 学的是 solution map：

$$

\Phi_\theta(x,t,r)\approx x_r.

$$

它的结构是：

$$

\text{Flow Matching anchor}
+
\text{solution-map consistency}.

$$

而 $x_1$-EMF 学的是：

$$

\tilde{x}_{t\to r} = 

x+(1-t)u_{t\to r}(x).

$$

它不是直接预测 $x_r$，而是用 endpoint-like coordinate 表达平均速度。

因此：

$$

\text{SoFlow} = 

\text{solution-map-first}.

$$

$$

x_1\text{-EMF} = 

\text{MeanFlow-first, endpoint-parameterized}.

$$

我们的 plug-in $x_1$-prediction 不是 SoFlow 的直接扩展，而是：

$$

\boxed{
\text{在 endpoint-parameterized EMF 中加入 empirical posterior endpoint supervision。}
}

$$

它处在三者之间：

$$

\text{Plug-in velocity}
+
x_1\text{-prediction EMF}
+
\text{flow-map/solution consistency}.

$$

---

# 12. 方法命名

可以考虑以下名字：

$$

\textbf{Plug-in Endpoint EMF}

$$

或者：

$$

\textbf{PiX1-EMF}

$$

或者：

$$

\textbf{Posterior Endpoint MeanFlow}

$$

我更倾向于：

$$

\boxed{
\textbf{Plug-in Endpoint Euler MeanFlow}
}

$$

因为这个名字准确表达了三件事：

1. plug-in：来自 batch empirical posterior；
2. endpoint：预测对象是 $x_1$-like variable；
3. Euler MeanFlow：基础框架是 JVP-free EMF。

---

# 13. 预期优点

这个方法可能带来四个好处：

## 第一，降低监督方差

原始 $x_1$ target 是 posterior sample：

$$

x_1\sim p(x_1\mid x_t).

$$

Plug-in target 近似 posterior mean：

$$

\hat m_B(x_t)\approx \mathbb E[x_1\mid x_t].

$$

因此 target variance 更低。

---

## 第二，保持 image-like 输出空间

相比 plug-in velocity：

$$

\hat v_B,

$$

plug-in endpoint：

$$

\hat m_B

$$

更接近图像/数据端坐标，更适合 $x_1$-prediction。

---

## 第三，和 EMF 的 JVP-free consistency 兼容

它不引入 JVP，只是在 EMF 的 anchor 和 local Euler step 中替换监督信号。

---

## 第四，可以自然配合 CFG

endpoint-space CFG 与 velocity-space CFG 线性等价。

---

# 14. 主要风险

## 风险一：posterior mean 可能 blurry

如果 posterior 多峰，则：

$$

\hat m_B=\sum_i\pi_i x_1^{(i)}

$$

可能是不自然的平均图像。

解决方式：

$$

m_{\lambda} = 

(1-\lambda)x_1+\lambda \hat m_B

$$

以及 ESS-adaptive $\lambda$。

---

## 风险二：finite-batch bias

Plug-in endpoint 估计的是：

$$

\mathbb E_{p_B}[x_1\mid x_t],

$$

不是：

$$

\mathbb E_{p_{\text{data}}}[x_1\mid x_t].

$$

所以 batch size 小时会有偏置。

解决方式：

* larger per-device batch；
* memory bank；
* nearest-neighbor candidate pool；
* shrinkage；
* temperature smoothing。

---

## 风险三：高维 softmax collapse

权重可能集中到单个样本。

解决方式：

$$

\tau\text{-temperature}

$$

和 ESS-controlled temperature。

---

## 风险四：CFG 类别信号被稀释

如果 batch 中类别混杂，plug-in endpoint 会平均掉 class-specific information。

解决方式：

* class-consistent batch；
* conditional/unconditional endpoint posterior 分开估计；
* endpoint-space CFG。

---

# 15. 最小实验方案

建议从小规模开始，不要一上来做 ImageNet-256 XL。

## Baseline

1. $u$-EMF；
2. $x_1$-EMF；
3. plug-in velocity EMF；
4. plug-in $x_1$-EMF。

---

## Ablation 1：替换位置

| 设置          | anchor target | Euler local step     |
| ----------- | ------------- | -------------------- |
| baseline    | $x_1$         | conditional velocity |
| only anchor | plug-in $x_1$ | conditional velocity |
| only step   | $x_1$         | plug-in direction    |
| both        | plug-in $x_1$ | plug-in direction    |

---

## Ablation 2：plug-in 形式

| 方法                         | target                                     |
| -------------------------- | ------------------------------------------ |
| full plug-in               | $\hat m_B$                                 |
| fixed shrinkage            | ($1-\lambda$x_1+\lambda\hat m_B)           |
| ESS-adaptive shrinkage     | ((1-\lambda(x,t))x_1+\lambda(x,t)\hat m_B) |
| temperature plug-in        | (\hat m_B^{$\tau$})                        |
| ESS-controlled temperature | choose $\tau$ by target ESS                |

---

## Ablation 3：CFG

1. mixed-class batch plug-in；
2. class-consistent batch plug-in；
3. conditional/unconditional posterior separately estimated；
4. endpoint-space CFG。

---

## Metrics

除了 FID，还应该看：

* training loss variance；
* gradient norm stability；
* ESS distribution；
* posterior weight entropy；
* sample sharpness；
* precision/recall；
* class fidelity；
* local artifact frequency。

---

# 16. 核心研究问题

这个方向可以凝练成一个问题：

> 在 $x_1$-prediction Euler MeanFlow 中，单样本 clean endpoint supervision 是否可以被 empirical posterior endpoint supervision 改进？

或者更数学化：

> Can we reduce the variance of endpoint prediction in JVP-free MeanFlow by replacing the conditional endpoint $x_1$ with an adaptive plug-in estimator of $\mathbb E[x_1\mid x_t]$?

最核心公式是：

$$

\boxed{
\hat m_B(x_t) = 

\sum_i \pi_i x_1^{(i)}
}

$$

以及：

$$

\boxed{
m_{\text{target}} = 

(1-\lambda)x_1+\lambda\hat m_B(x_t).
}

$$

---

# 17. 一句话总结

我们提出的想法是：

> **把 Shortcutting 里的 plug-in velocity 思想，从 velocity space 搬到 Euler MeanFlow 的 $x_1$-prediction space 中。具体做法是，用 batch empirical posterior mean $\hat m_B(x_t)=\sum_i\pi_i x_1^{(i)}$ 替代或融合原始单样本 endpoint target $x_1$，并通过 ESS-adaptive shrinkage、temperature smoothing 和 class-conditional posterior 来控制 bias-variance tradeoff。**

这个方法不是简单的 plug-in velocity 变形，而是利用了 $x_1$-prediction EMF 的输出空间优势，使低方差监督和 image-like endpoint prediction 结合起来。
