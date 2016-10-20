# sgrid_SVR_only

copy of sgrid's SVR module

## 数据准备

* 训练数据

   第一列为样本标签，后面的列为样本的各个特征值。特征值没有顺序要求，但要与测试数据的特征值顺序对应。数据源支持文本和数据库，若为文本形式，每一列数据间用半角逗号分隔。

  例如：

  4.06,90,5,6500,10600,0.61

  其中，4.06 为样本标签，90, 5, 6500, 10600, 0.61 均为样本特征值。


* 测试数据

  没有标签，每一列均为样本特征值，其顺序需与训练数据的特征值顺序对应。仅用于测试准确率，数据源仅支持文本形式。

## 用例

```java
public void regression1() {
  SVMLib svmLib = SVMLib.getInstance().setType(LibConfig.Type.REGRESSION).initDataFromFile("./datasets/train.csv");
  // uncomment this line to do cross validation and utilize the svm_param
  // caution: cost a lot of time
//  svmLib.svm_param = svmLib.updateParam();
  svm_model model = svmLib.train();
  regressionResult(model, "./datasets/test.csv", "./results/result.txt");
}
```

1. 获得svmLib实例，并设置使用类型（regression or classification）
2. 通过文件（本例中为“./datasets/train.csv”）初始化svmLib
3. （可选）通过交叉验证优化训练参数
4. 训练模型
5. 若需求为测试批量数据，则可通过demo中的regressionResult方法，获得并输出测试结果；若为单点预测，调用SVMLib中的predict方法，传入两个参数，一个为需预测的样本的各个特征值组成的数组，另一个为训练产生的模型。输出结果为模型预测该样本的标签。



## Support Vector Regression简介

​	假设训练集 (training data) 为
$$
(x_1, y_1),...,(x_l, y_l)\in\mathbb{R}^d\times\mathbb{R}
$$
​	其中 x 表示輸输入的特征(attributes), y 表示该特征所对应的回归值。令
$$
f(x) = \omega \cdot x + b, \omega \in \mathbb{R}^{d},b\in \mathbb{R}
$$
​	如果对于每个$x_i$，$f(x_i)$和$y_i$的差都很小，则认为这样的$f(x)$能从$x$准确的预测$y$。这个$\omega$即是SVR所要找的平面。用数学语言表达，可将SVR改写为下面问题：
$$
minimize \frac{1}{2}\lVert\omega\rVert^2\\
subject\,to\, \lVert y_i - (\omega \cdot x_i - b)\rVert \le \varepsilon
$$
​	其中$\varepsilon \ge 0$，用来表示SVR预测值与实际值最大的差距，而此算法也因此得名，称为$\varepsilon$-SVR。

​	在大多数情况下，因为存在噪音和误差的关系，通常无法从上式直接求出结果。因此我们要加入额外的项，以容许个别实例落在$\varepsilon$外：
$$
minimize \frac{1}{2} \lVert \omega \rVert^2 + C \sum_{i = 1}^l (\xi_i + \xi_i^*) \\
subject\,to\,
\begin{cases}
y_i - \omega \cdot x_i - b \le \varepsilon + \xi_i \\
\omega \cdot x_i + b - y_i \le \varepsilon + \xi_i^* \\
\xi_i, \xi_i^* \ge 0
\end{cases}
$$
​	在上式中，每一个训练样例都有其对应的$\xi$以及$\xi^*$，用来决定该样例是否可以落在$\varepsilon$的范围之外。而C的作用和SVM中一样，用来调整训练模型是否过拟合或欠拟合。



## LIBSVM

- 简介

  ​	LIBSVM是一个开源的机器学习库。LIBSVM实现了序列最小优化算法，支持分类（C-SVC，nu-SVC）和回归（epsilon-SVR，nu-SVR），同时它也支持多分类算法。

- 数据预处理

  ​	由于西高院数据不存在类别特征，而回归分析也不需要对数据集进行scaling，因此只需将样本按照0:label, 1:特征值1,...序号n:特征值n的顺序排列好即可。

- 模型选择

  ​	LIBSVM支持许多核函数的类型，一般情况下RBF核都会是首选。RBF核会将样本非线性的映射到高维空间，当特征值与label间的关系为非线性时能达到比线性核更高的准确度。此外，线性核也可以考虑成是RBF核的一种特殊情况。

  ​	另一个需要考虑的点是超参数的数量。多项式核的超参数数量比RBF核多得多，这就使得模型训练时调参的复杂度大大上升。

  ​	另一方面，在数据集的特征数量非常大的时候，RBF核的训练时间将会变得非常的长，因此在这种情况下，线性核也许是最好的选择。

- 交叉验证和Grid-search

  ​	RBF核有两个参数：C和$\gamma$。我们事先是无法知道一个特定问题下C和$\gamma$的最优解的，必须通过一定的方法选择一个好的(C, $\gamma$)对，使得回归模型能够准确的预测未知数据（测试集）。一般的衡量方法是将数据集分为两部分，一部分用作训练集，另一部分假装不知道他们的label，用作测试数据。再将预测出的结果与测试数据的真实label比较，以判断预测的准确性。这种验证方法的升级版就是交叉验证（cross-validation）。

  ​	例如v-折交叉验证，先把训练集按照样本数量平均分为v个子集，每次取其中一个子集用作测试集，剩下的v-1个子集用作训练集。因此整个训练集的每一个样本都会被预测一次。将正确预测的样本百分比作为交叉验证的准确度度量值。交叉验证能够减轻过拟合问题。

  ​	Grid-search是用来选择合适的C和$\gamma$的方法。通过实验大量的(C, $\gamma$)对，选择其中交叉验证准确度最高的一对(C, $\gamma$)作为应用参数。通常，C和$\gamma$的步长为几何级增长（例如，$C = 2^-5, 2^-3,…,2^{15}, \gamma = 2^{-15}, 2^{-13},...2^3$）是一种较为实际可行的搜索方式。

test