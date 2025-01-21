# Real-Time Zero-Day Intrusion Detection System for Automotive Controller Area Network on FPGAs

## Overview
This repository contains the implementation of a real-time intrusion detection system (IDS) for automotive Controller Area Networks (CAN). The system uses an unsupervised-learning-based convolutional autoencoder to detect zero-day attacks, targeting resource-constrained FPGA platforms. The IDS achieves high classification accuracy for various attack types (DoS, fuzzing, spoofing) and operates at line rate with low energy consumption.

### Key Features
- **Unsupervised Learning Approach**: Detects zero-day attacks by training only on benign CAN messages.
- **FPGA Deployment**: Utilizes AMD/Xilinx Vitis-AI tools for quantization and optimization.
- **High Classification Accuracy**: Greater than 99.5% accuracy on unseen attack types.
- **Real-Time Detection**: Meets the line-rate detection requirement of 0.43 ms per window on high-speed CAN networks.
- **Low Power Consumption**: Ideal for energy-efficient, embedded IDS systems.

---

## Prerequisites

To run the software and deploy on an FPGA, you will need:

- **Hardware**: Zynq Ultrascale platform or compatible FPGA.
- **Tools**:
  - AMD/Xilinx Vitis-AI tools for quantization and deployment.
  - Python 3.7+.
- **Libraries**:
  - TensorFlow 2.x.
  - NumPy.
  - Scikit-learn.
  - Additional dependencies listed in `requirements.txt`.

---

## Usage

The train/validation/testing and deployment scripts for the model are in the scripts in the respective folders.

## Results
- **Detection Accuracy**: >99.5% on DoS, fuzzing, and spoofing attacks from the [CAN-intrusion-dataset](https://example-dataset-link).
- **Real-Time Performance**: 0.43 ms per message window, suitable for high-speed CAN.

---

## Citation
If you use this work in your research, please cite:

```
@inproceedings{khandelwal2023real,
  title={Real-time zero-day intrusion detection system for automotive controller area network on fpgas},
  author={Khandelwal, Shashwat and Shreejith, Shanker},
  booktitle={2023 IEEE 34th International Conference on Application-specific Systems, Architectures and Processors (ASAP)},
  pages={139--146},
  year={2023},
  organization={IEEE}
}

```

---
