# ICE - Individualized Classifier Ensemble


### Abstract
Multiple classifier system (MCS) has become a successful method for improving classification performance. We believe that the two crucial steps of MCS base classifier generation and multiple classifier combination, need to be designed coordinately to produce robust results. In this work, we show that for different testing instances, better classifiers may be trained from different subdomains of training instances including, for example, neighboring instances of the testing instance, or even instances far away from the testing instance. To utilize this intuition, we propose Individualized Classifier Ensemble (ICE). ICE groups training data into overlapping clusters, builds a classifier for each cluster, and then associates each training instance to the top-performing models while taking into account model types and frequency. In testing, ICE finds the k most similar training instances for a testing instance, then predicts class label of the testing instance by averaging the prediction from models associated with these training instances. Evaluation results on 49 benchmarks show that ICE has a stable improvement on a significant proportion of datasets over existing MCS methods. ICE provides a novel choice of utilizing internal patterns among instances to improve classification, and can be easily combined with various classification models.


### Content
./data/: data folder containing all the evaluation datasets in the manuscript. 

./src/: the source code folder

./supp/: the supplementary files for the manuscript

./manuscript.pdf: the manuscript file of ICE

