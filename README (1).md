
# Progressive GAN with StyleGAN-like Architecture

This repository contains an implementation of a Progressive Growing GAN with a StyleGAN-like architecture for generating stunning space-themed images. The model is trained using PyTorch and employs Wasserstein loss with gradient penalty for stable training. The project includes a Streamlit-based web application for generating space imagery using the trained GAN model.


## Features

- Progressive Growing: The model starts training on low-resolution images and progressively increases resolution to enhance details.

- StyleGAN-like Architecture: Inspired by StyleGAN, featuring adaptive instance normalization and other enhancements.

- Wasserstein Loss with Gradient Penalty: Ensures stable training and better convergence.

- Streamlit Web App: A user-friendly interface for generating space images using the trained model.

- Pretrained Model Support: Ability to load saved models for continued training or inference.


## Installation

Install Transformer

```bash
git clone https://github.com/09-prince/OrboGAN.git
cd OrboGAN
```

```bash

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
    
## Training the GAN Model

- Add a Dataset file in this folder
- Change the location of dataset in the model.py
- Run the model

## Future Enhancements

- Improve the realism of space images using more advanced training techniques.

- Optimize performance for real-time image generation.

- Deploy the app online for public access.
## Contributing

Contributions are always welcome!

Feel free to fork the repository, submit issues, and contribute improvements!.


## License

This project is licensed under the MIT License.

