ReLU 全称是 Rectified Linear Unit，是深度学习中最常用的 activation function（激活函数）之一。

它的定义非常简单：

$$f(x) = \max(0, x)$$

也就是说：输入如果是正数，原样输出；如果是负数或零，输出 0。

为什么需要它：神经网络如果只做 linear transformation（线性变换），无论叠多少层，本质上还是一个线性模型。加入 ReLU 这样的 non-linear activation（非线性激活）后，网络才能学习复杂的模式。

ReLU 流行的原因是计算极其简单（就是一个 max 操作），而且在实践中训练效果好，比早期的 sigmoid 和 tanh 更不容易出现 vanishing gradient（梯度消失）问题。