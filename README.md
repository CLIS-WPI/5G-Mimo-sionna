# Explainable AI-Driven Beamforming for Adaptive MIMO Systems in 5G

This repository contains the code and simulation setup for the research project **"Explainable AI-Driven Beamforming for Adaptive MIMO Systems in 5G"**. The project integrates Reinforcement Learning (RL) with Explainable AI (XAI) to optimize beamforming parameters and provide interpretable insights into AI-driven decision-making processes for static user scenarios in 5G networks.

## Project Overview

Beamforming plays a critical role in enhancing 5G network performance by dynamically adapting to changing channel conditions and user demands. Despite advancements in AI, the lack of transparency in AI-driven beamforming limits operator trust. This project aims to:

* Use the Soft Actor-Critic (SAC) algorithm for dynamic beamforming optimization
* Incorporate Explainable AI (XAI) methods, such as SHAP, to interpret RL model decisions
* Simulate the physical layer using NVIDIA's Sionna library

## Repository Structure

```
├── simulation_plan.md       # Detailed simulation parameters and configuration
├── src/                     # Source code for dataset generation, training, and evaluation
│   ├── dataset_generator.py # Code for generating channel realizations
│   ├── rl_training.py       # Implementation of SAC algorithm for beamforming
│   ├── xai_analysis.py      # Code for SHAP-based explainability
│   ├── config.py           # Configuration file for parameters
├── data/                    # Folder to store generated datasets
│   ├── training/           # Training dataset
│   ├── validation/         # Validation dataset
│   ├── test/               # Test dataset
├── results/                 # Folder to store evaluation results and visualizations
│   ├── shap_plots/         # XAI analysis results
│   ├── performance_metrics/# Evaluation metrics for trained RL model
|__ utill
|__ |___ utill.py           #
├── README.md               # Project documentation
└── requirements.txt        # Required Python libraries
```

## Getting Started

### Prerequisites

To run this project, you need the following:

* Python 3.8 or above
* NVIDIA GPUs (optional but recommended for training acceleration)

### Install Dependencies

Run the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

### Generate Dataset

To generate the initial dataset for training, run:

```bash
python src/dataset_generator.py
```

### Train RL Model

Train the SAC-based RL model using:

```bash
python src/rl_training.py
```

### Run XAI Analysis

To analyze the trained RL model's decisions using SHAP, run:

```bash
python src/xai_analysis.py
```

## Key Features

* **Dataset Generation**: Generate channel realizations based on the simulation parameters defined in `simulation_plan.md`
* **Reinforcement Learning**: Train the RL agent (SAC algorithm) for adaptive beamforming
* **Explainable AI**: Use SHAP to provide interpretable insights into RL decision-making
* **Performance Evaluation**: Evaluate model performance using metrics like SINR, throughput, and spectral efficiency

## Results

* Improved beamforming adaptability and system performance in static scenarios
* Enhanced transparency for network operators through XAI
* Dataset of 1.32M samples covering diverse channel conditions

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{your_paper,
  title={Explainable AI-Driven Beamforming for Adaptive MIMO Systems in 5G},
  author={Your Name},
  journal={IJCNN 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
