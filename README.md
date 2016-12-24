# minibatch
A class to preprocess and generate mini-batches of data from a set of csv files.
This class loads mini-batches of specific size given a folder including files or an array of file paths and enables processing the data including normalization, etc.

This class is useful for training models with large training data which is scattered across multiple files.

#Requirements:
1. python 3.4
2. numpy (v1.10.*+)
3. pandas (v0.18.1)

#Requirements in samples:
1. sklearn (v0.17.1)
2. tensorflow (v0.7)
