# Deep Learning Projects

### Project 1: Hand-written characters classification (including pen-digits and Chinese characters)
Pen-digits data is downloaded from [pendigits](https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/).

Chinese character data is downloaded from [CASIA Online and Offline Chinese Handwriting Databases](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html). It is difficult to preprocess this dataset, I used the code from [this blog](https://blog.csdn.net/weixin_39683769/article/details/113050852) to transform the original dataset and save each hand-written Chinese character as .npy file.

Main idea: Apply the signature transform to preprocess hand-written characters and then apply logistic regression. Here are some reference papers for the signature transform:
* "A Primer on the Signature Method in Machine Learning" by Ilya Chevyreva and Andrey Kormilitzina.
* "Calculation of Iterated-Integral Signatures and Log Signatures" by Jeremy Reizenstein.
* "Signature moments to characterize laws of stochastic processes" by Ilya Chevyrev and Harald Oberhauser.
* "Characteristic functions of measures on geometric rough paths" by Ilya Chevyreva and Terry Lyons.
* "Differential Equations Driven by Rough Paths" by Terry J. Lyons, Michael J. Caruana, and Thierry Lévy. 
* "System Control and Rough Paths" by Terry J. Lyons and Zhongmin Qian. 

And the signature transform method is implemented in the **signatory** package which can be accelerated by GPU and works the fastest in all existing packages. Read more about the tutorial [here](https://signatory.readthedocs.io/en/latest/index.html).

I choose this transform because
* there is a one-to-one map between the original time series and the infinite signature vector, as the signature level increases, its elements' norm will decrease factorially, so we can use this transform will only a little information loss;
* we can transform long time series to a much shorter vector to reduce computation cost;
* it can be applied to any irregularly-sampled time series so we do not need to delete or add points to make the observations have the same
length before training models;
* it is shift-invariance and invariant under time reparametrizations;
* it is robust to outliers.

Besides, I added a dimension to augment raw characters which indicates the number of stroke points belonging to.

### Project 2: Sequential data generation
After analyzing sequential data, I also tried to generate new sequential data. One idea is to train GANs based on Wasserstein distance between signature vectors for different sequences. But it failed for the pen-digit data and kept producing random x-y coordinates. Still working in this direction.
