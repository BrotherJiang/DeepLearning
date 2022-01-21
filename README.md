# Deep Learning Projects

### Project 1: Hand-written characters classification (including pen-digits and Chinese characters)
Pen-digits data is downloaded from [https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/](https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/).

Chinese character data is downloaded from [CASIA Online and Offline Chinese Handwriting Databases](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html). It is difficult to preprocess this dataset, I used the code from [this blog](https://blog.csdn.net/weixin_39683769/article/details/113050852) to transform the original dataset and save each hand-written Chinese characters as .npy file.

Main idea: Apply the signature transform to preprocess hand-written characters and then apply logistic regression. Here are some reference papers for the signature transform:
* "A Primer on the Signature Method in Machine Learning" by Ilya Chevyreva and Andrey Kormilitzina.
* "Calculation of Iterated-Integral Signatures and Log Signatures" by Jeremy Reizenstein.
* "Signature moments to characterize laws of stochastic processes" by Ilya Chevyrev and Harald Oberhauser.
* "Characteristic functions of measures on geometric rough paths" by Ilya Chevyreva and Terry Lyons.
* "Differential Equations Driven by Rough Paths" by Terry J. Lyons, Michael J. Caruana and Thierry Lévy. 
* "System Control and Rough Paths" by Terry J. Lyons and Zhongmin Qian. 

And the signature transform method is implemented in the **signatory** package which can be accelrated by GPU and works the fastest in all existing packages. Read more about the tutorial [here]{https://signatory.readthedocs.io/en/latest/index.html}.

We choose to use this transform because
* there is a one-to-one map between the original time series and the infinite signature vector, as the signature level increases, its elements' norm will decrease factorially, so we can use this transform will only a little information loss;
* we can transform long time series to a much shorter vector to reduce computation cost;
*  
