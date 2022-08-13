# MLNaiveBayesAndLogisticRegression

- ***Naive Bayes Classifier***: `src/nbayes.py`
- ***Logistic Regression Classifier***: `src/logreg.py`
- ***Results***: `notebooks/Programming_Two.ipynb`
- ***Research Extension***: `notesbooks/Research_Extension_Assignment2.ipynb`

# MLNaiveBayesAndLogisticRegression

- ***a. Decision Tree Learner***: Our source code for the ID3 decision tree algorithm is contained in `src/` with `src/dtree.py` as the entry point for the program. See the [Usage](#Usage) section for execution instructions.

- ***c. Jupyter Notebook/pdf Writeup***: 
    - Sections a~c are contained in `notebooks/Programming_One_Part_Two.ipynb`
    - Our research extension is contained in `notebooks/research_extension.ipynb`

# Usage
```
python3 src/dtree.py {path-to-dataset} {tree-depth} [--no-cv] [--use-gain-ratio] [--randomize]
```

### Options
- `{path-to-dataset}`: the directory containing `.data` and `.names` files for the dataset to train on
- `{tree-depth}`: Depth limit of the decision tree. `0` generates an unbounded depth limit
- `--no-cv`: Disables cross-validation
- `--use-gain-ratio`: Enables gain ratio instead of information gain as tree split criterion
- `--randomize`: Enables random selection of split attributes

# Project Structure
- `notebooks/`: Project writeup notebooks
    - `Programming_One_Part_Two.ipynb`: Sections a~c of our project writeup. Contains test results and analysis.
    - `research_extension.ipynb`: Research extension of out project. Contains analysis of the ID3 algorithm and hypotheses and test results of possible improvements to ID3.
    - `reference/`: reference notebooks on python libraries (numpy, matplotlib)
- `src/`: Source code for our ID3 algorithm implementation
    - `dtree.py`: Entry point for the program. Contains our DecisionTree class.
    - `util.py`: Contains utility functions for calculating metrics and manipulating data.
    - `split_criteria.py`: Contains methods to select split attributes in the decision tree.

# General Program Design Overview

The class `dtree.DecisionTree` represents a decision tree model.

We use a recursive implementation where a DecisionTree generates subtrees as DecisionTree's fit on partitions of its training data.

Partitions are determined by a `DecisionTree.split_attribute` and `DecisionTree.split_boundary` where the `split_attribute` is the index of the attribute of the sample schema to partition by, and the `split_boundary` is the value at which to partition continuous attributes. If the `split_attribute` is not continuous, `split_boundary` is `None`.

The `split_attribute` and `split_boundary` are determined by a split criterion function passed to the decision tree as a dependency. By default, a split criterion selects the attribute with highest information gain, but can be passed alternative functions to select attributes by gain ratio or to select a random attribute.

Worked on this project with 3 other fellow students at CWRU
