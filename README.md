![](https://img.shields.io/badge/tensorflow_probability-0.11.0-yellow) ![](https://img.shields.io/badge/license-GNU%20GPL%20v3.0-blue)

# tfp_helper

A tiny Helper Library for Tensorflow Probability. It helps you to quickly prototype probabilistic models and run automated inference without much hassle.

- **Inference Button** Single function call for running inference with MCMC
- **Plotting** Inference returns an instance of `arviz.InferenceData` which enables user to use `arviz` for plotting
- **Progress Bar** A smart tqdm-based progress bar to track the progress of MCMC ▓▓▓▓▓▓▓▓▓▓░░░░░


## Getting Started

Instructions on getting the repo setup and running on your local machine.

### Prerequisites

Install requirements for using the library.

```bash
pip install -r requirements.txt
```

### Installation

Install library locally using `setup.py`.

```
python setup.py install
```



## Running the tests

Instructions for running tests.

```bash
pytest tests.py
```

## License

This project is licensed under the GNU GPL v3.0 License - see the [LICENSE](LICENSE) file for details.

