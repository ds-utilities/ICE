The datasets are collected from UCI machine learning repository and Kaggle Dataset for binary classification, with number of instance between 100 and 3,000, number of features between 3 and 1500, and percentage of majority class ranging from 50% to 77%. A total of 42 datasets from UCI and 23 datasets from Kaggle meet the criteria (18 of which appeared in both repositories). In addition, we add two cancer-related datasets - breast-cancer-nki [Van De Vijver et al., 2002] and breast-cancer-wang [Wang et al., 2005]. The data preprocessing mainly follows [Ferna ́ndez-Delgado et al., 2014], which includes a Z-Score transformation based normalization. For nine datasets with nominal features we use two different methods to handle nominal features: (i) removing nominal features (denoted with suffix ‘-1’ in Figure 2), (ii) using One-Hot encoding (denoted with suffix ‘-2’ in Figure 2). Since all features in dataset ‘tic-tac-toe’ are nominal, this dataset only has the One-Hot encoding version.

./data_all/final_49_names_and_ix.m and .mat show the dataset names.

./data_all/x/data.mat shows the dataset in matlab format. This file containing 2 variables - X and y. X is the features vectors; y is the class labels.

X.pkl and y.pkl are the pickle format of data array and label array.


