---

# FloraAI 🌸

FloraAI is an image classification project developed to identify various species of flowers using machine learning techniques. This project was completed as part of the AWS AI & ML Scholarship program in collaboration with Udacity. The model leverages Python and popular machine learning libraries to create an image classifier that can predict flower species from images.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
FloraAI aims to provide an easy-to-use tool for classifying different types of flowers based on their images. The project involves creating and training a neural network using a dataset of flower images. The model can then be used to predict the species of a flower given a new image.

## Features
- Image classification of flower species using a trained neural network.
- Model training using a custom dataset.
- Easy-to-use command-line interface for predictions.
- Detailed accuracy and loss metrics during training.
- Supports transfer learning using pre-trained models.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jennifer-kay/FloraAI.git
   cd FloraAI
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Once you have installed the dependencies, you can use the model to classify images:

1. **Training the model:**
   ```bash
   python train.py --data_dir path_to_data --epochs 10 --gpu
   ```

2. **Making predictions:**
   ```bash
   python predict.py --image_path path_to_image --checkpoint checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
   ```

## Model Training
The model is trained using a dataset of flower images. Training involves using a convolutional neural network (CNN) architecture with transfer learning from pre-trained models such as VGG16 or ResNet.

**Training Example:**
```bash
python train.py --data_dir flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 20 --gpu
```

## Technologies Used
- **Python**: Core language for development.
- **PyTorch**: Deep learning framework for building and training the model.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For visualizing data and results.
- **AWS**: For cloud computing resources.

## Contributing
Contributions are welcome! If you’d like to contribute, please fork the repository and use a feature branch. Pull requests are reviewed regularly.

**Steps to contribute:**
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact
Created by [Jennifer Kay](https://www.linkedin.com/in/jenniferkaydev) - feel free to connect!

---
