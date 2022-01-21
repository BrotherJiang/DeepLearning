# Deep Learning Projects

### Project1: Hand-written characters classification (including pen-digits and Chinese characters)
Main idea: Apply the signature transform to preprocess hand-written characters and then apply logistic regression. Here are some reference papers for the signature transform:


Pen-digits data is downloaded from [https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/](https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/).

Chinese character data is downloaded from [CASIA Online and Offline Chinese Handwriting Databases](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html). It is difficult to preprocess this dataset, I used the code from [this blog](https://blog.csdn.net/weixin_39683769/article/details/113050852) to transform the original dataset and save each hand-written Chinese characters as .npy file.

