# abc-ribbon: Approximate Bayesian Inference and a Stochastic Model of a Ribbon Synapse

Code accompanying the paper "Approximate Bayesian Inference for a Mechanistic model of Vesicle Release at a Ribbon Synapse" (Schröder, James et al. Preprint: https://doi.org/10.1101/669218 , NeurIPS version: https://papers.nips.cc/paper/8929-approximate-bayesian-inference-for-a-mechanistic-model-of-vesicle-release-at-a-ribbon-synapse )

## File descriptions:

### data - folder:

Contains the stimulus and experimental recordings for the release of two BC in preprocessed version.
Also the dF/F example traces for one cell are stored.
And a notebook for showing the data.

### generalized_method - folder:

Contains a simple example how to generalize the presented method to other problems. Including the essential steps of defining a meaningful loss function and prior distributions.
The essential sampling and updating functions are in abc_method.py and not specific to the model.


### poster - folder:

Contains the poster which will be presented at NeurIPS conference. 

### paper_version - folder:

Contains all files for the presented ABC method. Contains all files to reproduce the plots of the paper.


### standalone_model -folder:
If you are only interested in the stochastic model of the ribbon synapse, look here.

ribbonv2.py: contains the ribbon model

standalone_model.ipynb: running the model in ribbonv2.py with specified stimulus and model parameters. 





