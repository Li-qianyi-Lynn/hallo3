MLP 全称是 Multi-Layer Perceptron（多层感知机），是最基础的 feedforward neural network（前馈神经网络）结构。

它的组成很直观：

Input layer — 接收输入数据
一个或多个 hidden layers — 每层由多个 neurons（神经元）组成
Output layer — 输出结果

每一层的每个 neuron 都和下一层的所有 neuron 相连，所以也叫 fully connected layer（全连接层），在 PyTorch 里对应 nn.Linear。

每个 neuron 做的事情就是：接收输入 → 做一次 linear transformation（权重乘以输入再加偏置） → 通过一个 activation function（比如 ReLU） → 输出。

用公式表示一层的计算就是：

$$h = \text{ReLU}(Wx + b)$$

MLP 是深度学习的基础构件。在现代架构里（比如 Transformer），每个 block 内部都有一个 MLP（通常叫 FFN / Feed-Forward Network），负责对每个 token 的 representation 做非线性变换。