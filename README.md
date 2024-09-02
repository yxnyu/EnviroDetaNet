# EnviroDetaNet

EnviroDetaNet is for the paper:EnviroDetaNet: Pretrained E(3)-equivariant Message-Passing Neural Networks with Multi-Level Molecular Representations for Organic Molecule Spectra Prediction. 
This repository contains the code, datasets, and documentation needed to use and understand the model.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Additional Information](#additional-information)
- [Dataset & Checkpoint](#dataset--checkpoint)
- [License](#license)

## Installation

To use EnviroDetaNet, simply clone this repository to your local machine:

```bash
git clone https://github.com/yxnyu/EnviroDetaNet.git

## Usage

To train the EnviroDetaNet model, you can use the `trainer.py` script located in the root directory of the repository.

Here's how you can run the training script:

```bash
python trainer.py


## Additional Information

Due to version conflicts between Uni-Mol and the original DetaNet in terms of PyTorch, we recommend converting Uni-Mol representations to JSON format first. This allows for querying during training, which reduces scheduling overhead and ensures that Uni-Mol weights are compatible with our model across different versions.

Additionally, we provide an integrated mode for `unimol_embedding`, which can be switched as needed. This approach simplifies usage and enhances speed.

We sincerely thank the Uni-Mol team ([Uni-Mol GitHub Repository](https://github.com/deepmodeling/Uni-Mol)) and the original DetaNet team ([DetaNet CodeOcean Capsule](https://codeocean.com/capsule/3259363/tree/v3)) for their support. Our version builds upon the original DetaNet, and many modules can be easily modified for mutual compatibility.

## Dataset & Checkpoint

We have provided the datasets/json and checkpoints used for training, testing, and infrared applications in the paper. You can download them from our Google Drive:

[Google Drive Link](https://drive.google.com/drive/folders/1u2a3MNFDz4XcK0wv-rBPTqfw2pKdsyz8?usp=sharing)

Simply download the files to your local machine and place them in the appropriate directories as specified in the code.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software as long as you include the original license text in any copies or substantial portions of the software.

See the [LICENSE](LICENSE) file for more details.