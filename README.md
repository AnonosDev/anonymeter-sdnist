# anonymeter-sdnist

This repository contains the code used to analyze the privacy of de-identified data 
submitted as part of the [NIST Collaborative Reserach Cycle
(CRC)](https://pages.nist.gov/privacy_collaborative_research_cycle/index.html) using
[`Anonymeter`](https://github.com/statice/anonymeter). The results of this analysis 
will be published soon in the form of a tiny paper. 


### Setup

To run this you need to clone the [CRC repository](https://github.com/usnistgov/privacy_collaborative_research_cycle/tree/research-acceleration-bundle) in this repository.


```shell

git clone git@github.com:usnistgov/privacy_collaborative_research_cycle.git

```

In particular put the [`crc_data_and_metric_bundle_1.1`](https://github.com/usnistgov/privacy_collaborative_research_cycle/tree/research-acceleration-bundle/crc_data_and_metric_bundle_1.1)
directory with all the de-identified data submissions in this directory. Then you also need to download the 
[SDNIST](https://github.com/usnistgov/SDNist/tree/main) repository with the [NIST Diverse Communities Data Excerpts](https://github.com/usnistgov/SDNist/tree/main/nist%20diverse%20communities%20data%20excerpts) original and control datasets.

Finally, you need to install the required packages, including `anonymeter`, to be able to run the privacy evaluations:

```shell
pip install -r requirements.txt
```

### Structure

There are a few modules in this repository:

- `anonymeter_evaluation_lib.py`: contains convenience high-level classes to run privacy attacks for different
  hyperparameter settings. 
- `analysis.py`: this is the main analysis script. It will run everything. It is recommended to provision enough
  computing power for a more pleasant experience.
- `analysis_utils.py`: low-level utility functions needed to run the analysis.
- `plot_results.py`: script to produce all the plot in the paper. 
- `plot_functions.py` and `plot_utils.py`: low level plotting functions and utilities. 

