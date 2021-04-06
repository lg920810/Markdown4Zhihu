

##  **Unsupervised Visual Representation Learning**

关于Unsupervised visual representation learning，主要总结了8篇文章，如下：

1. [Learning Deep Representations by Mutual Information Estimation and Maximization](https://arxiv.org/pdf/1808.06670.pdf) (Deep InfoMax)
2. [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf) (CPC)
3. [Contrastive Multiview Coding](https://arxiv.org/pdf/1906.05849.pdf) (CMC)
4. [Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://arxiv.org/pdf/1805.01978.pdf) (Memory Bank)
5. [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722.pdf) (MoCo)
6. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf) (SimLR)
7. [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733.pdf) (BYOL)
8. [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) (NCE)



还有一些待读文章

1. [On Mutual Infomation Maximization For Representation Learning](https://arxiv.org/pdf/1907.13625.pdf) 
2. [Run Away From Your Teacher: A New Selfsuperviesed Approach Solving The Puzzle of BYOL](https://openreview.net/pdf?id=tij5dHg5Hk)
3. [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/pdf/2005.10242.pdf)



比较好的总结和综述

1. https://github.com/jason718/awesome-self-supervised-learning
2. [Contrastive Representation Learning: A Framework and Review](https://arxiv.org/ftp/arxiv/papers/2010/2010.05113.pdf)



### Supervised Learning & Unsupervised Learning

下图是监督学习和无监督学习在Imagenet数据集上的结果对比，图中左边蓝色框内是目前在添加额外数据的基础上得到的最好结果，top1为88.5%，top5则达到了98.7%。而未添加额外数据监督学习方法的top1大概在85.8%左右，无监督学习方法为71.7%。从数据可以看出两者的差距较大，接近14%，但相对于早期的无监督方法，现在已经从top1 54%提升到71.7%，有17.7%的提升，可见提升幅度还是很大的。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201013164622334.png" alt="image-20201013164622334" style="zoom:80%;" />



### Deep InfoMax (DIM) — Microsoft Research (ICLR2019)

在介绍DIM方法前，我们先了解一些基本概念(熵，条件熵，互信息)，这个过程可能比较吃力，但是对contrastive learning将会有本质认识。

此外将不再介绍AMDIM方法，因为AMDIM(Augmented Multiscale DIM)和DIM来自一家，其对于DIM做了一些改进：

* 数据增强

* 优化多尺度间的互信息，而不是局部+全局

* 换了更好的ecoder

  

#### **熵和条件熵**

为了引出互信息，先给出熵和条件熵的定义，这样方便对互信息的理解。

在信息论与概率统计中，熵 (entropy) 是表示随机变量不确定性的度量，熵越大，随机变量的不确定就越大。设$X$是个取有限个值的离散随机变量，其概率分布为：
$$
P(X=x_i)=p_i,i=1,2,...,n
$$
则随机变量$X$的熵定义为
$$
H(X)=-\sum_{i=1}^n {p_i \log{p_i}}
$$
设有随机变量$(X,Y)$，其联合概率分布为：
$$
P(X=x_i,Y=y_j)=p_{ij},i=1,2,..,n;j=1,2,...,m
$$
条件熵表示在已知随机变量$Y$的条件下随机变量$X$的不确定性。条件熵 (conditional entropy) $H(X\vert Y)$ 定义为：
$$
\begin{align}
H(X\vert Y)&=\sum_{j=1}^m p_j H(X\vert Y=y_j)
\\&=\sum_{j=1}^m p_j \cdot [-\sum_{i=1}^n (\frac{p_{ij}}{p_j}\log{\frac{p_{ij}}{p_j}})] =-\sum_{j=1}^m \sum_{i=1}^n (p_{ij}\log{\frac{p_{ij}}{p_j}})
\\&=\sum_{j=1}^m \sum_{i=1}^n (p_{ij}\log{ \frac{p_j}{p_{ij}}})
\end{align}
$$


#### **互信息(Mutual Information)**

互信息(Mutual Information)也称为信息增益，当熵和条件熵的概率由数据估计得到时，所对应的熵与条件熵分别称为经验熵和经验条件熵，则信息增益定义为两者之差。也就是：
$$
\begin{align}
I(X,Y)&=H(X)-H(X \vert Y)
\\&=-\sum_{i=1}^n {p_i \log{p_i}}- \sum_{j=1}^m \sum_{i=1}^n (p_{ij}\log{ \frac{p_j}{p_{ij}}})
\\&=-\sum_{i=1}^n {(\sum_{j=1}^m p_{ij}) \log{p_i}}- \sum_{j=1}^m \sum_{i=1}^n (p_{ij}\log{ \frac{p_j}{p_{ij}}})
\\&=-\sum_{i=1}^n {\sum_{j=1}^m p_{ij} \log{p_i}} - \sum_{j=1}^m \sum_{i=1}^n (p_{ij}\log{ \frac{p_j}{p_{ij}}})
\\&= -\sum_{i=1}^n \sum_{j=1}^m (p_{ij}\log{\frac{p_i p_j}{p_{ij}}})
\\&=\sum_{i=1}^n \sum_{j=1}^m (p_{ij}\log{\frac{p_{ij}}{p_i p_j}})
\end{align}
$$
互信息的直观解释就是描述两个随机变量的相关性，假设$X$和$Y$是独立的，也就是不相关，则$I(X,Y)=0$，同时可以得到$H(X\vert Y)=H(X)$；若希望两者的互信息大，则要求$H(X)$越大越好，$H(X\vert Y)$越小越好，就说明给定的 $Y$ 对 $X$ 要有很大影响，这样的 $X$ 才是好特征。



下面我们用图像再重新描述一遍，用 $X$ 表示原始图像的集合，用 $x\in X$ 表示某一原始图像，$Z$表示编码向量的集合，$z\in Z$ 表示某个编码向量，$p(z|x)$ 表示 $x$ 所产生的编码向量的分布，我们设它为高斯分布，或者简单理解它就是我们想要寻找的编码器。那么可以用互信息来表示$ X,Z $ 的相关性，则有：

$$
\begin{align}
I(X,Z)&=\iint p(x,z) \log{p(x,z) \over p(x)p(z)}dxdz
\\&=\iint p(z|x)p(x) \log{p(z|x) \over p(z)}dxdz
\tag{1}
\end{align}
$$

那么一个好的特征编码器，应该要使得互信息尽量地大，即：
$$
p(z|x)={\underset {p(z|x)}{\operatorname {arg\,max} }}\, I(X,Z)
\tag{2}
$$

#### **先验分布**

我们还希望隐变量服从标准正态分布，使得后续学习更加稳定，因此我们需要对隐变量加一个约束。设 $q(z)$ 为标准正态分布，即$q(z)\sim N(0,1)$，我们利用KL散度来最小化 $p(z)$ 与先验分布 $q(z)$。
$$
KL(p(z)||q(z))=\int p(z) \log {p(z) \over q(z)}dz
\tag{3}
$$

#### **优化函数**

结合以上两个式子，我们构造的优化函数为：
$$
\begin{align}
p(z|x)&=\min_{p(z|x)}-I(X,Z)+\lambda{KL(p(z)||q(z))}
\\&=\min_{p(z|x)}\{-\iint p(z|x)p(x) \log{p(z|x) \over p(z)}dxdz + \lambda\int p(z) \log {p(z) \over q(z)}dz\}
\tag{4}
\end{align}
$$

但由于我们不知道$p(z)$，因此也无法计算上式。所以我们需要对上式做一些简化，这里利用全概率公式我们已经知道：


$$
p(z)=\int{p(z|x)p(x)dx}
\tag{5}
$$
因此将(5)代入(4)，则有
$$
\begin{align}
p(z|x)&=\min_{p(z|x)}\{-\iint p(z|x)p(x) \log{p(z|x) \over p(z)}dxdz + \lambda\int p(z) \log {p(z) \over q(z)}dz\}
\\&=\min_{p(z|x)}\{\iint p(z|x)p(x) [{-(1+\lambda)}\log{p(z|x) \over p(z)} + \lambda \log {p(z|x) \over q(z)}]dxdz\}
\tag{6}
\end{align}
$$
注意上式正好是互信息与$\mathbb{E}_{x\sim p(x)}[KL(p(z|x)\Vert q(z))]$的加权求和，而$KL(p(z|x)\Vert q(z))$这一项是可以算出来的，所以我们已经成功地解决了整个loss的一半，可以写为
$$
\begin{align}
p(z|x) =\min_{p(z|x)}\left\{-\beta\cdot I(X,Z)+\gamma\cdot \mathbb{E}_{x\sim{p}(x)}[KL(p(z|x)\Vert q(z))]\right\}
\end{align}
\tag{7}
$$

#### **互信息本质**

我们把互信息定义(1)的$\log$部分上下同乘一个$p(x)$，则有：
$$
\begin{equation}
\begin{aligned}
I(X,Z) =& \iint p(z|x)p(x)\log \frac{p(z|x)p(x)}{p(z)p(x)}dxdz\\ 
=& KL(p(z|x)p(x)\Vert p(z)p(x))
\end{aligned}
\tag{8}
\end{equation}
$$
这个形式反应了互信息的本质含义：$p(z|x)p(x)$描述了两个变量$z,x$的联合分布，$p(z)p(x)$则是随机抽取一个$x$和一个$z$时的分布（假设它们两个不相关时），而互信息则是这两个分布的$KL$散度。所谓最大化互信息，就是要拉大$p(z|x)p(x)$与$p(z)p(x)$之间的距离。

$KL$散度是无上界的且不对称，因此无法最大化，所以需要换一个有上界的度量来描述两个分布间的差异，$JS$散度或者其他的度量如$Hellinger$距离都可以。在这里我们使用$JS$散度，$JS$散度有很好的性质，非负性和对称性。其定义为：
$$
JS(P,Q) = \frac{1}{2}KL\left(P\left\Vert\frac{P+Q}{2}\right.\right)+\frac{1}{2}KL\left(Q\left\Vert\frac{P+Q}{2}\right.\right)
$$
JS散度同样衡量了两个分布的距离，但是它有上界$\frac{1}{2}\log 2$，我们最大化它的时候，同样能起到类似最大化互信息的效果，但是又不用担心无穷大问题。于是我们用下面的目标函数取代式(7)
$$
\begin{align}
p(z|x) =\min_{p(z|x)}\left\{-\beta\cdot JS\big(p(z|x){p}(x), p(z){p}(x)\big)+\gamma\cdot \mathbb{E}_{x\sim{p}(x)}[KL(p(z|x)\Vert q(z))]\right\}
\end{align}
\tag{9}
$$
现在的问题就剩下如何求解$JS$散度了。

#### **求解JS散度**

利用一般的$f$散度(各种散度的统称)的局部变分推断，$p(x),q(x)$为任意两个分布，则
$$
\begin{align}
\mathcal{D}_f(P\Vert Q) = \max_{T}\Big(\mathbb{E}_{x\sim p(x)}[T(x)]-\mathbb{E}_{x\sim q(x)}[g(T(x))]\Big)
\tag{10}
\end{align}
$$
对于$JS$散度，给出的结果是:
$$
\begin{align}
JS(P,Q) = \max_{T}\Big(\mathbb{E}_{x\sim p(x)}[\log \sigma(T(x))] + \mathbb{E}_{x\sim q(x)}[\log(1-\sigma(T(x))]\Big)
\end{align}
$$
代入$p(z|x){p}(x), p(z){p}(x)$就得到
$$
\begin{aligned}&
JS\big(p(z|x)\tilde{p}(x), p(z)\tilde{p}(x)\big)\\=& \max_{T}\Big(\mathbb{E}_{(x,z)\sim p(z|x){p}(x)}[\log \sigma(T(x,z))] + \mathbb{E}_{(x,z)\sim p(z){p}(x)}[\log(1-\sigma(T(x,z))]\Big)
\end{aligned}
\tag{11}
$$
其实(11)式的含义非常简单，它就是"负采样估计"：引入一个判别网络$\sigma(T(x,z))$，$x$及其对应的$z$视为一个正样本对，$x$及随机抽取的$z$则视为负样本，然后最大化似然函数，等价于最小化交叉熵。

#### **与先验分布的KL散度计算**

假设$p(z) \sim N(\mu, \sigma^2)$，则有：
$$
\begin{align}
KL(p(z)||q(z))&=\int {1 \over \sqrt{2\pi}\sigma} \exp{-{(z-\mu)^2}\over{2\sigma^2}} \log {{1 \over \sqrt{2\pi}\sigma} \exp{-{(z-\mu)^2}\over{2\sigma^2}} \over {1 \over \sqrt{2\pi}}\exp{-z^2\over 2}}dz
\\&=\int{1 \over \sqrt{2\pi}\sigma} \exp{-{(z-\mu)^2}\over{2\sigma^2}}[\log ({1 \over \sigma}\exp{-{(z-\mu)^2}\over{2\sigma^2}}) - \log({\exp{-z^2\over 2}})]dz
\\&=\int{1 \over \sqrt{2\pi}\sigma} \exp{-{(z-\mu)^2}\over{2\sigma^2}}[\log {1 \over \sigma} - {{(z-\mu)^2}\over{2\sigma^2}} +{z^2\over 2}]dz
\\&=-\log\sigma-{1\over2}+{1\over2}(\mu^2+\sigma^2)
\\&=-{1\over2}(1+2\log\sigma-\mu^2-\sigma^2)
\end{align}
\tag{12}
$$

#### **DIM 方法**

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201019140017197.png" alt="image-20201019140017197" style="zoom: 50%;" />

Deep Infomax为更好的获取图像的特征采取了全局特征和局部特征共同利用的方式，所以需要两个Discriminator用来生成两个均值和两个方差参与到整个网络的优化当中，网络输出的就是均值和方差，和VAE方法一致。正负样本的选择采用shuffle的方式，所有的参数均保存在内存中，仅适用于小数据集。



###  CPC— Google DeepMind (NIPS2018）

CPC(Contrastive Predictive Coding)首次提出infoNCE，后续很多方法都使用其作为损失函数。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201106150822852.png" alt="image-20201106150822852" style="zoom:50%;" />
$$
I(x,c)=\sum_{x,c}p(x,c)\log{p(x|c) \over p(x)}
\tag 1
$$
(1)式为互信息定义。CPC的流程为观测值$x_t$会通过非线性模型$g_{enc}$被映射为一个序列表达$z_t=g_{enc}(x_t)$，而后利用自回归模型$g_{ar}$生成$z_{\le{t}}$的上下文表达即$c_t=g_{ar}(z_{\le{t}})$

作者不直接使用生成模型$p_k(x_{t+k}\vert c_t)$来预测未来的观测值$x_{t + k}$，而是对$x_{t + k}$与$c_t$之间的互信息密度比进行建模：
$$
f_k(x_{t+k}, c_t)\propto \frac{p(x_{t+k}\vert c_t)}{p(x_{t+k})}
\tag 2
$$
(2)式右侧可以理解为在$c_t$情况下的$x_{t+k}$发生的概率占任意情况下$x_{t+k}$的概率比值，随后利用一个大于零的函数来构造$f$，在这里使用了对数双线性模型:
$$
f_k(x_{t+k}, c_t)=\exp{(z_{t+k}^T W_k c_t)}
\tag 3
$$
InfoNCE被构造以下形式，它表示了正样本占总负样本的比值，也是正确分类正确的交叉熵损失
$$
\mathcal{L}_N=-\mathbb{E}_X \left[{\log{ f_k(x_{t+k},c_t) \over \sum_{x_j \in X} f_k(x_j, c_t)}}\right]
\tag 4
$$
互信息与上式的关系为:
$$
I(x_{t+k,c_t} )\geq  \log(N) - \mathcal{L}_N
\tag 5
$$
随着$N$变大，两者关系变得更紧密，还有最小化InfoNCE等价于最大化互信息的下界。



### **CMC—Google DeepMind** (ECCV2020)

CMC(Contrastive Multiview Coding) 有对CPC中互信息与infoNCE关系的证明

* **Multiview**



<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201104133811981.png" alt="image-20201104133811981" style="zoom:67%;" />



* **互信息下界的证明**

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201104133915751.png" alt="image-20201104133915751" style="zoom:80%;" />



### **Memory Bank**—UC Berkeley (CVPR2018)

在介绍MoCo方法前，不得不提一下[Memory Bank](https://arxiv.org/pdf/1805.01978.pdf)，该方法是CVPR2018的spotlight，由伯克利、香港中文大学、亚马逊联合发表的论文。论文主要论述如何通过非参数的instance discrimination进行无监督的特征学习。主要的思想是将softmax分类转化成非参数化方法，进行每个样本的区分。

#### **非参数化 Non-Parametric**

正常softmax函数形式如下：
$$
P(i|\mathbf{v})={\exp{\mathbf{w}_i^T\mathbf{v}} \over \sum_{j=1}^n{\exp{\mathbf{w}_j^T\mathbf{v}}}}
\tag{1}
$$
替换$\mathbf{w}_j^T\mathbf{v}$为 $\mathbf{v}_j^T\mathbf{v}$，并且限制$\Vert\mathbf{v}\Vert=1$ ，可以得到非参数化的softmax函数：
$$
P(i|\mathbf{v})={\exp{\mathbf{v}_i^T\mathbf{v}} / \tau \over \sum_{j=1}^n{\exp{\mathbf{v}_j^T\mathbf{v}}}/ \tau }
\tag{2}
$$
非参数的softmax主要思路是每个样本特征除了可以作为特征之外，也可以把它作为使用cos距离的KNN分类器，因为$L_2$-norm之后的特征乘积本身就等于cos相似性。
$$
\cos(\mathbf{v}_j,\mathbf{v})=(\mathbf{v}_j^T,\mathbf{v})={(\mathbf{v}_j^T,\mathbf{v}) \over {\Vert \mathbf{v}_j^T \Vert \Vert \mathbf{v} \Vert}}
$$


#### **对比损失(Contrastive loss) vs 交叉熵损失(Cross-entropy)**

假如我们有$n$个样本 $X_i \in R^K, y_i\in\lbrace0,1\rbrace^C, i=1,...,n$，$C$为类别数。

##### **交叉熵损失**

交叉熵损失(Cross-entropy Loss) 是分类问题中常用的损失函数：
$$
L_{CE}=-\sum_{c=1}^C{I(y_i=c)\log{P(y=c|X_i)}}
\tag{3}
$$
这里，由于 $y_i$ 是binary向量，即只有预测标签等于真实标签时，求和项才有意义。在现有的图像分类模型中，最后一层一般是linear layer+softmax，如果将用于特征提取的最后一层输出视为$f(x_i)$，最后一层linear layer的权重视为$\mathbf{w}$ ，则有：
$$
P(y=c|X_i)={\exp(\mathbf{w}_c^T f(X_i)) \over \sum_{c=1}^{C}{\exp(\mathbf{w}_c^T f(X_i))}}
\tag{4}
$$

##### 非参数样本分类损失

$$
P(y=c|X_i)={\exp(f(X_c)^T f(X_i)) \over \sum_{c=1}^C{\exp(f(X_c)^T f(X_i))}}
\tag{5}
$$

##### **对比损失**

区别与以上两种分类损失函数，对比损失常用于无监督学习中，因此是没有真实标签的，但基于同一张图像无论是旋转，色彩变化亦或是其他变换都不改变图像本事的特征，可以把问题转化为如下分类问题：

假如对原始图像$X_i$，分别做不同变换$A$和$B$，得到$A(X_i),B(X_i)$，对比损失期望$A(X_i),B(X_i)$之间特征的距离要小于$A(X_i),X_j,j\neq i$，即$D(A(X_i),B(X_i)) \ll D(A(X_i),X_{j \neq i}) $。在实际操作中，假如我们使用$cosine$距离，假设已经归一化特征值，则优化上式实际上等同于最大化下式中的$softmax$概率，
$$
P(A,B\in {same\_class})={\exp(f(A(X_c))^T f(B(X_i)) \over \sum_{k=0}^K{\exp(f(A(X_{k\ne c}))^T f(X_i))}}
\tag{6}
$$

#### **Memory Bank**

为了减少计算量，这篇文章提出了memory bank的概念：维护一个memory bank $\mathbf{V}=\lbrace\mathbf{v_j}\rbrace$，使用样本特征$\mathbf{f}_i=f_\theta(x_i)$ 来更新memory $\mathbf{f}_i\to\mathbf{v}_i$ 。

![image-20201015140102105](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201015140102105.png)

memory bank 开辟了和训练集数量同样的内存，用于存储每张图像经过网络提取的特征向量。在训练中，每个mini-batch都需要从memory bank中随机采样，memory bank中保存的向量通过momentum的方式更新。

##### **Alias Method 离散分布随机取样**

对于处理离散分布的随机变量采样问题，Alias Method for Sampling 是一种很高效的方式，在初始好之后，每次采样的复杂度为$O(1)$。 

方法待补充 http://shomy.top/2017/05/09/alias-method-sampling/



### **MoCo—Facebook AI Research (CVPR2020)**

#### 1. Introduction

论文第一段介绍了无监督学习在NLP领域卓有成效，但是在计算机视觉领域却发展缓慢，理由如下：

1. 语言任务通常处于离散信号空间，可以很好的构建tokenized dictionaries。
2. 计算机视觉任务由于原始信号是连续的且处于高维度空间导致字典构造困难。

同时引出现在的方法大多数是本质都是利用编码器构建字典，且基于contrastive loss进行优化。作者认为这个思路正确，但需要关注两个问题：

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201014165256299.png" alt="image-20201014165256299" style="zoom: 67%;" />

**Inconsistent 理解** 

文中所指的inconsistent主要说的是Memory Bank的方法，该方法可以构建很大的字典，但是样本的特征表达会在每个step后被更新，也就是说在一轮训练中每个step采样的样本都是来自不同encoder的，因此会导致不一致的问题。对比实验也表明在构建同样大的字典时，该方法比MoCo低2.6个百分点。



#### **PS1. Contrastive Loss (InfoNCE)**

本文给出的损失函数来自于CPC文章使用的infoNCE，形式如下：


$$
\begin{align}
L_q=-\log{\exp(q \cdot k_+ /\tau) \over \sum_{i=0}^K\exp(q\cdot k_i
/\tau)}
\end{align}
$$
where $\tau$ is a temperature hyper-parameter per. The sum is over one positive and $K$ negative samples. Intuitively, this loss is the log loss of a $(K+1)$-way softmax-based classifier that tries to classify q as $k_+$. Contrastive loss functions can also be based on other forms, such as margin-based losses and variants of NCE losses.

我们还是先介绍一下Contrastive Loss，首次被Yann LeCun提出，论文发表于CVPR 2006，Dimensionality Reduction by Learning an Invariant Mapping，目的是增大分类器的类间差异，其形式如下：
$$
L(W,Y,\vec{X_1},\vec{X_2})=(1-Y)\frac{1}{2}(D_W)^2+(Y)\frac{1}{2}[max(0,m-D_W)]^2
$$
TODO: 这部分内容太多太杂了，不同的paper都对contrastive loss有不同的改动，从N-pair loss到NCE loss，但本质都是为了对同类样本拉近距离，不同类样本拉远距离。区别于分类的交叉熵损失，无监督学习本身没有正负样本标签，因此要引入该种损失函数进行评估。

* 需要详细介绍N-pair loss，该损失函数不同于之前的triplet loss仅使用一个负样本，而是使用了多个负样本。



#### **2. Related Work**

文中该部分主要介绍了三部分内容，篇幅不长，分别为Loss function，Pretext tasks 和 Contrastive learning vs pretext tasks。loss function之前已经介绍过了，但是下文所提到的生成对抗网络和NCE的关系还需要进一步研究。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201015112904471.png" alt="image-20201015112904471" style="zoom: 67%;" />

Pretext tasks 可以理解为是一种为达到特定训练任务而设计的间接任务，或者说就是通过self-supervised的方式进行预训练模型的训练，同样的利用contrastive loss进行训练，再fine-tuning到下游任务。通常会利用图像本身的操作构造“标签”，从而学习encoder。其应用有图像复原，图像去噪，图像着色等。

而Contrastive learning和pretext tasks的关系原文有如下描述：

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201015114157944.png" alt="image-20201015114157944" style="zoom: 63%;" />

该段表示不同的pretext task都是基于不同变种的contrastive loss function的。比如Instance Discrimination方法与基于实例的任务Discriminative unsupervised feature learning with convolutional neural networks相关，还有NCE。

#### **3. Method**

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201019165110923.png" alt="image-20201019165110923" style="zoom:50%;" />

总的来说MoCo方法的改进主要相对于基于memory bank的方法而言的，上文已经介绍memory bank的操作方式。MoCo引入一个新的encoder，如上图称为momentum encoder，用于负样本(key)的生成。由于它也是个encoder，因此也需要参数更新，后面我们再介绍更新方式。我们先顺序的解答几个问题：

**A1: 正负样本如何构造？**

Q1: 首先一张图像进入Dataloader，会经过two_crop的操作，会得到两张局部图像且互为正样本。第一张图像会经过各种transfrom的操作，再进入到encoder部分，得到表征向量$(q)$，第二张图像则通过momentum encoder，得到向量$(k_+)$，那么$q \cdot k_+$则为正样本，对应二分类标签 $1$；从采样队列$queue$中获取的样本记为$k$，那么$q \cdot k$则为负样本，对应二分类标签 $0$，由于每次都是以mini-batch的方式进行训练，因此负样本是有多个的。

**A2: encoder和momentum encoder的关系是什么呢？**

Q2: 初始都是一样的，代码中使用了ResNet50。随着更新的过程，二者参数开始变化，下式表达了二者关系：
$$
momentum  \; encoder=0.99 \times encoder + 0.01 \times momentum \; encoder
$$
**A3: Moco与Memory Bank中momentum的区别？**

Q3: Memory Bank的动量更新是在同一样本的表示上，而不是在编码器上。

**A4: 用于采样的队列是如何维护的？**

Q4: 初始化使用随机初始化方式，长度为65536，batch size为256时，一次前向就往队列中入队256个样本表征向量，出队256个。

**A5: Shuffling BN的作用是什么？**

​                                       <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201027212611546.png" alt="image-20201027212611546" style="zoom:75%;" />

Q5: 文章中提到使用BN会使得模型快速收敛，原因是模型学到了规律而不是真的图像表征。这可能是因为一个batch中的样本通信会泄露信息。之前的做法为：当使用多GPU训练的时候，对每个GPU上的mini-batch进行BN操作，然后将样本再分发到每个GPU中；现在的做法为：对于momentum encoder($f_k$)，在分配到其他GPU前打乱当前mini-batch的样本顺序，然后利用$f_k$生成编码向量进行normalize后再还原顺序，而对于encoder($f_q$)不变。这样可以保证用于计算的$query$和$key_+$来自两个不同的子集。



#### **4. Experiments**

##### **Ablation: contrastive loss mechanisms**

end-to-end: 一个mini-batch内选择负样本

memory bank: 一个memory bank内随机选择负样本

MoCo: 一个memory内选择负样本，区别去memory bank，样本本身不同，由于更新机制的不同

下图证明了在Imagenet上采用MoCo方式的有效性，同时作者还在detection任务上做了实验作为验证。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201028222309206.png" alt="image-20201028223010687" style="zoom: 55%;" />

##### **Ablation: momentum**

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201028223010687.png" alt="image-20201028223010687" style="zoom: 50%;" />

##### **Comparison with previous results**

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201028223123599.png" alt="image-20201028223123599" style="zoom:50%;" />

##### **MoCo v2 results**

MoCo v2是在SimCLR提出projection head和data augumentation后，在原模型基础上加了projection head和相应的数据增强，并使用了cosine lrschedule，迭代次数由200增加至800后得到的结果。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201028224144291.png" alt="image-20201028224144291" style="zoom:60%;" />

MoCo做了很多的比对实验，有兴趣的话可以看原文。



### **SimCLR—Google Research** (ICML2020)

论文证明了三个点

* 数据增强在任务中起着至关重要的作用
* 在representation和对比损失之间引入可学习的非线性变换(MLP)，大大提高了学习表示的质量
* 与监督学习相比，对比学习受益于更大的批次数量和更多迭代次数

#### 1. Introduction

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201104103636873.png" alt="image-20201104103636873" style="zoom:60%;" />

作者表示在主流的用于visual representations方法中(包括监督学习和无监督学习)，主要有两种学习机制：生成学习和判别学习，这和机器学习就很一致了。生成模型通过学习$P(X,Y)$的联合概率分布，然后求出条件概率分布$P(Y|X)=P(X,Y)/P(X)$; 而判别模型则直接计算$P(Y|X)$。文中对CV中的生成学习总结成pixel-level generation，但细分的话有Auto-regressive (AR) Model，Flow-based Model，Auto-encoding (AE) Model，Hybrid Generative Models等。而在CV中的判别学习就是指那些基于contrastive learning的方法。

#### **2. Method**

* **Projection Head**

从网络结构上，区别去其他方法作者在encoder后面加了一个MLP (多层感知机)，称为projection head。随后kaiming在MoCo v2中加了projection head后在imagenet上涨了6个点，证实了其有效性。

* **Data augmentation**

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201104133042042.png" alt="image-20201104133042042" style="zoom:67%;" />

* **Training Detail (Large Batchsize)**

With 128 TPU v3 cores, it takes ∼1.5 hours to train our ResNet-50 with a batch size of 4096 for 100 epochs. 后续MoCo v2说不需要large batchsize 也可以。



### **BYOL (Google DeepMind)**

BYOL (Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning) 同样是Google DeepMind团队发出的一篇文章，2020年6月首次挂到arXiv上面，暂时还没查到发表到哪儿了。区别于之前提到的方法，该方法仅使用了正样本对，而舍弃了负样本对。

#### 损失函数

为了清晰说明我们重新写一下contrastive loss的公式：
$$
\begin{align}
L_q&=-\log{\exp(q \cdot k_+ /\tau) \over \sum_{i=0}^K\exp(q\cdot k_i
/\tau)}\\
&=-q \cdot k_+/\tau + \log{[\exp{(q \cdot k_+ /\tau)} + \sum_{i=0}^{K-1}\exp{(q\cdot k_-}/\tau)]}\\
&=\underbrace{-q \cdot k_+/\tau}_{\rm positive} + \underbrace{\log{[\exp{(1/\tau)} + \sum_{i=0}^{K-1}\exp{(q\cdot k_-}/\tau)]}}_{\rm negtive}
\end{align}
$$
可以看到contrastive loss可以写成正负样本损失相加的形式，而BYOL就是去掉了negtive部分，仅使用positive部分作为损失，并且把问题看作回归问题，使用mean squared error作为评估策略。原文并未从上述contrative loss入手构造损失函数，而是直接给出了以下形式：
$$
\begin{align}
L_{BYOL}&={\Vert q-k_+ \Vert}_2^2 =  q^2 + k_+^2 - 2q \cdot k  \\
&=2-2 \cdot \frac{\left\langle q, k_+ \right\rangle}{\Vert q\Vert_2 \cdot \Vert k_+\Vert_2 }
\end{align}
$$

#### 网络结构

下图为该方法的网络结构图，现在通常把MLP也就是projection head的部分单拿出来作为一部分，其实也可以将resnet+MLP作为整体的网络结构来看。其中target network也就是$t^{'}$的更新策略同MoCo一致。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\BYOL_network.jpg" alt="image-20201109112918642" style="zoom: 67%;" />



#### 实验结果

* **Unsupervise learning on ImageNet**

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201109131345064.png" alt="image-20201109131345064" style="zoom: 67%;" />

* **Semi-supervised learning on Imagenet**

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201109131458054.png" alt="image-20201109131458054" style="zoom:67%;" />

* **Ablation study (Batch size and Transformation)**

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201109132035108.png" alt="image-20201109132035108" style="zoom: 67%;" />



* **https://openreview.net/pdf?id=tij5dHg5Hk** 解释了BYOL为什么work，类似于DQN的训练方式？



### **On Mutual Infomation Maximization For Representation Learning (ICLR2020 Google Research)**

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201105171154188.png" alt="image-20201105171154188" style="zoom: 55%;" />

本文的观点是获取良好的特征表达并不仅仅依赖于互信息，而是归纳偏差。后续文章证明了常用的infoNCE损失函数的优化目标并不是MI的下界，之前的CPC文章首次提出infoNCE证明了是优化MI的下界，目前有些凌乱了。



### **NCE (Noise-Contrastive Estimation)**

NCE的做法就是将它转化为二分类问题，将真实样本判为1，从另一个分布采样的样本判为0。NCE的思想很简单，它希望我们将真实的样本和一批“噪声样本”进行对比，从中发现真实样本的规律出来。具体来说，能量还是原来的能量$G(x;θ)$，但这时候我们不直接算概率$p(x)$了，因为归一化因子很难算。我们去算

$$
p(1|\boldsymbol{x})=\sigma\Big(G(\boldsymbol{x};\boldsymbol{\theta})-\gamma\Big)=\frac{1}{1+e^{-G(\boldsymbol{x};\boldsymbol{\theta})+\gamma}}\tag{1}
$$
这里的$\theta$还是原来的待优化参数，而$\gamma$则是新引入的要优化的参数。

然后，NCE的损失函数变为
$$
\mathop{\arg\min}_{\boldsymbol{\theta},\gamma} - \mathbb{E}_{\boldsymbol{x}\sim \tilde{p}(\boldsymbol{x})}\log  p(1|\boldsymbol{x})- \mathbb{E}_{\boldsymbol{x}\sim U(\boldsymbol{x})}\log  p(0|\boldsymbol{x})\tag{2}
$$
其中$\widetilde{p}(x)$是真实样本，$U(x)$是某个“均匀”分布或者其他的、确定的、方便采样的分布。

我们将(7)式中的loss改写为
$$
-\int \tilde{p}(\boldsymbol{x})\log  p(1|\boldsymbol{x}) d\boldsymbol{x}- \int U(\boldsymbol{x})\log  p(0|\boldsymbol{x})d\boldsymbol{x}\tag{3}
$$
因为$\widetilde{p}(x)$和$U(x)$都跟参数$θ,γ$没关，因此将loss改为下面的形式，不会影响优化结果
$$
\begin{aligned}
&\int \big(\tilde{p}(\boldsymbol{x})+U(\boldsymbol{x})\big) \left(\tilde{p}(1|\boldsymbol{x}) \log  \frac{\tilde{p}(1|\boldsymbol{x})}{p(1|\boldsymbol{x})} + \tilde{p}(0|\boldsymbol{x})\log  \frac{\tilde{p}(0|\boldsymbol{x})}{p(0|\boldsymbol{x})}\right)d\boldsymbol{x}\\ 
=&\int \big(\tilde{p}(\boldsymbol{x})+U(\boldsymbol{x})\big) KL\Big(\tilde{p}(y|\boldsymbol{x})\Big\Vert p(y|\boldsymbol{x})\Big) d\boldsymbol{x}
\end{aligned}
\tag{4}
$$
其中
$$
\tilde{p}(1|\boldsymbol{x})=\frac{\tilde{p}(\boldsymbol{x})}{\tilde{p}(\boldsymbol{x})+U(\boldsymbol{x})}
\tag{5}
$$
(11)式是KL散度的积分，而KL散度非负，那么当“假设的分布形式是满足的、并且充分优化”时，(11)式应该为0，从而我们有$\tilde{p}(y|\boldsymbol{x})= p(y|\boldsymbol{x})$，也就是
$$
\frac{\tilde{p}(\boldsymbol{x})}{\tilde{p}(\boldsymbol{x})+U(\boldsymbol{x})}=\tilde{p}(1|\boldsymbol{x})=p(1|\boldsymbol{x})=\sigma\Big(G(\boldsymbol{x};\boldsymbol{\theta})-\gamma\Big)
\tag{6}
$$

从中可以解得
$$
\begin{aligned}\tilde{p}(\boldsymbol{x})=&\frac{p(1|\boldsymbol{x})}{p(0|\boldsymbol{x})}U(\boldsymbol{x})\\ 
=&\exp\Big\{G(\boldsymbol{x};\boldsymbol{\theta})-\gamma\Big\}U(\boldsymbol{x})\\ 
=&\exp\Big\{G(\boldsymbol{x};\boldsymbol{\theta})-\big(\gamma-\log U(\boldsymbol{x})\big)\Big\}\end{aligned}
\tag{7}
$$
如果$U(x)$取均匀分布，那么$U(x)$就只是一个常数，所以最终的效果表明$γ−logU(x)$起到了$\log Z$Z的作用，而分布还是原来的分布(3)，$θ$还是原来的$θ$。

这就表明了NCE就是一种间接优化(3)式的巧妙方案：看似迂回，实则结果等价，并且(5)式的计算量也大大减少，因为计算量就只取决于采样的数目了。

