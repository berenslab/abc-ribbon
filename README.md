# abc-ribbon

Code accompanying the paper "Approximate Bayesian Inference for a Mechanistic model of Vesicle Release at a Ribbon Synapse" (Schröder, James et al. Preprint: https://doi.org/10.1101/669218 )

## File descriptions:

ribbonv2.py: contains the ribbon model

standalone_model.ipynb: running the model in ribbonv2.py with specified stimulus and model parameters. 

### data - folder:

Contains the stimulus and experimental recordings for the release of two BC.
And a notebook for showing the data.

### generalized_method - folder:

Contains a simple example how to generalize the presented method to other problems. Including the essential steps of defining a meaningful loss function and prior distributions.
The essential sampling and updating functions are in abc_method.py and not specific to the model.


### poster - folder:

Contains the poster which will be presented at NeurIPS conference. 
