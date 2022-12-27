Diffusion models are inspired by non-equilibrium thermodynamics(非平衡热力学). 他们定义了一个 Markov 链里的diffusion steps来慢慢在数据里加入随机噪音，然后学习如何从diffusion过程中反向来从噪音中构造想要的数据采样。与 VAE 或 流模型不同，diffusion 是通过固定步骤(fixed procedure)来学习，而 latent 变量是高维的（和原始数据相同）

## Diffusion Models 是什么？
有很多种方案被提出来：diffusion probabilistic models(2015), noise-conditioned score network(NCSN), and denoising diffuion probabilistic models(DDPM)

### Forward diffusion process
xi 代表从实际数据分布中采样出的数据点 x0 ~ q(x)，定义一个 forward 的 diffusion 过程：在T步骤里，逐步加入少量的高斯噪音，产出一系列带了噪音的采样点：x1,...,xT。step size 是通过一个在(0,1)之间的方差调度βt来控制:

![](imgs/data-sample-and-variance-schedule.png)

从上图可见在xt-1采样下xt的采样是属于高斯分布的。而在x0 已知的情况下，剩余xi的分布是前者的乘积

采样的数据 x0 逐步会随着 t 的增大而丢失可辨识的 features。最终当 T -> 无穷，xT 等价于各向同性的高斯分布。

### Reverse diffusion process
我们可以逆向上述过程，从 q(xt-1|xt)里进行采样，可以从一个高斯噪音输入 xT~N(0,I) 里重建出 true sample。当 βt 足够小，q(xt-1|xt) 就也是高斯分布。但不幸的是，无法轻松估计出 q(xt-1|xt)，因为需要知道所有的数据集，所以我们需要学习一个模型 ptheta 来估计这些有条件的概率，以此来运行反向 diffusion 的过程

