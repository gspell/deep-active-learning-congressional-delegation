# Deep/Active Learning for Congressional Delegation
This project regards the classification of U.S. Congressional bills with respect to political delegation -- i.e., whether legislation delegates authority to federal agencies or not. As a machine learning and deep learning task, it concerns supervised text classification, using a Convolutional Neural Network (CNN).  The supervised learning endeavor has been extended to include active learning, which will hopefully mitigate labeling efforts for Congressional researchers and political scientists.

## Supervised CNN
An implementation of a CNN for text classification.

## Active Learning
The current active learning implementation involves the use of the logits after the softmax layer to determine which (unlabeled) documents would receive the least-certain labels.  These documents are those that should be queried so as to give the model the most useful information for further classification of the dataset. 
