# XEM: An Explainable-by-Design Ensemble Method for Multivariate Time Series Classification
This repository contains the Python implementation of XEM as described in 
the paper [XEM: An Explainable-by-Design Ensemble Method for Multivariate Time Series Classification](https://hal.inria.fr/hal-03599214/document).

<p align="center">
<img src="/images/non_terminating_atrial_fibrilation.png" width="60%">
</p>

## Requirements
XEM has been implemented in Python 3.8 with the following packages:
* lcensemble
* numpy
* pandas
* pyarrow
* pyyaml
* scikit-learn

## Usage
Run `main.py` with the following argument:

* configuration: name of the configuration file (string)

```
python main.py --config configuration/config.yml
```

## Citation
```
@article{Fauvel22XEM,
  author = {Fauvel, K. and E. Fromont and V. Masson and P. Faverdin and A. Termier},
  title = {XEM: An Explainable-by-Design Ensemble Method for Multivariate Time Series Classification},
  journal = {Data Mining and Knowledge Discovery},
  year = {2022}
}
```