"""
StyleGAN Implementation

This is an implementation of StyleGAN architecture as described in the paper
"A Style-Based Generator Architecture for Generative Adversarial Networks"

The implementation features:
- Progressive growing of generated images
- Style-based generator with AdaIN
- Mapping network for style transformation
- Noise injection
- Weight scaling
- Minibatch standard deviation
"""
# Import PyTorch and related modules
import torch
import math
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from math import log2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hyperparameters
DATASET = "Dataset"
START_TRAIN_AT_IMG_SIZE = 8  # Start from 8x8 images instead of 4x4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZES = [256, 128, 64, 32, 16, 8]
CHANNELS_IMG = 3
Z_DIM = 256
W_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)

# Channel scaling factors for different resolutions
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


def get_loader(image_size):
    """
    Create a data loader for the specified image size.

    Args:
        image_size: Target image size to resize to

    Returns:
        loader: DataLoader object
        dataset: Dataset object
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return loader, dataset


class WSLinear(nn.Module):
    """
    Weight-scaled Linear layer as per StyleGAN paper.
    Applie weight scaling for equalized learning rate.
    """

    def __init__(
            self, in_features, out_features,
    ):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        # initialize linear layer
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias


class PixelNorm(nn.Module):
    """
    Pixel-wise normalization layer.
    Normalizes each feature vector to unit length.
    """

    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class MappingNetwork(nn.Module):
    """
    Mapping network to transform z-space to w-space as in StyleGAN.
    Consists of 8 fully connected layers with leaky ReLU activations.
    """

    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
        )

    def forward(self, x):
        return self.mapping(x)


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) layer.
    Applies instance normalization and then scales and shifts with style.
    """

    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = WSLinear(w_dim, channels)
        self.style_bias = WSLinear(w_dim, channels)

    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias


class InjectNoise(nn.Module):
    """
    Noise injection layer.
    Adds random noise scaled by a learned parameter.
    """

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight * noise


class WSConv2d(nn.Module):
    """
    Weight-scaled Conv2d layer as per StyleGAN paper.
    Applies weight scaling for equalized learning rate.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class ConvBlock(nn.Module):
    """
    Convolutional block used in the discriminator.
    Contains two conv layers with LeakyReLU activations.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x


class Discriminator(nn.Module):
    """
    StyleGAN discriminator with progressive growing.
    Features minibatch standard deviation and progressive resolution growing.
    """

    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # Work backwards from highest resolution to 4x4
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out))
            self.rgb_layers.append(
                WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0)
            )

        # RGB layer for 4x4 resolution
        self.initial_rgb = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        # down sampling using avg pool
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Final block for 4x4 input resolution
        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(
                in_channels, 1, kernel_size=1, padding=0, stride=1
            ),
        )

    def fade_in(self, alpha, downscaled, out):
        """
        Used to smoothly fade in downscaled using avg pooling and output from CNN.

        Args:
            alpha: Blending factor between 0 and 1
            downscaled: The downscaled image
            out: Output from the CNN

        Returns:
            Blended result
        """
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        """
        Calculates standard deviation of features across the batch
        and concatenates it as an additional feature map.

        Args:
            x: Input tensor

        Returns:
            x with an additional feature map for batch statistics
        """
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        """
        Forward pass of discriminator.

        Args:
            x: Input image
            alpha: Blending factor for fade-in
            steps: Current resolution step

        Returns:
            Discriminator output (realness score)
        """
        # Select the appropriate starting block based on current resolution
        cur_step = len(self.prog_blocks) - steps

        # Convert from RGB
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:  # For 4x4 resolution
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # Handle progressive growing with fade-in
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        # Process through remaining blocks

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


class GenBlock(nn.Module):
    """
    Generator block used in StyleGAN.
    Contains two conv layers, each followed by noise injection and AdaIN.
    """

    def __init__(self, in_channels, out_channels, w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)

    def forward(self, x, w):
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x


class Generator(nn.Module):
    """
    StyleGAN generator with progressive growing, mapping network,
    style injection with AdaIN, and noise injection.
    """

    def __init__(self, z_dim, w_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.starting_constant = nn.Parameter(torch.ones((1, in_channels, 4, 4)))
        self.map = MappingNetwork(z_dim, w_dim)
        self.initial_adain1 = AdaIN(in_channels, w_dim)
        self.initial_adain2 = AdaIN(in_channels, w_dim)
        self.initial_noise1 = InjectNoise(in_channels)
        self.initial_noise2 = InjectNoise(in_channels)
        self.initial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

        # RGB layer for 4x4 resolution
        self.initial_rgb = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        # Create progressive blocks for different resolutions
        for i in range(len(factors) - 1):  # -1 to prevent index error because of factors[i+1]
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(GenBlock(conv_in_c, conv_out_c, w_dim))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        """
        Blends between upscaled and generated images during fade-in phase.

        Args:
            alpha: Blending factor between 0 and 1
            upscaled: Upscaled image from previous resolution
            generated: Generated image at current resolution

        Returns:
            Blended image with tanh activation applied
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, noise, alpha, steps):
        """
        Forward pass of generator.

        Args:
            noise: Input noise vector
            alpha: Blending factor for fade-in
            steps: Current resolution step

        Returns:
            Generated image
        """
        # Map noise to style space
        w = self.map(noise)
        x = self.initial_adain1(self.initial_noise1(self.starting_constant), w)
        x = self.initial_conv(x)
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)

        # For 4x4 resolution, return directly
        if steps == 0:
            return self.initial_rgb(x)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="bilinear")
            out = self.prog_blocks[step](upscaled, w)

        # Generate RGB images and blend with fade-in
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


def generate_examples(gen, steps, n=100):
    """
    Generate and save example images from the generator.

    Args:
        gen: Generator model
        steps: Current resolution step
        n: Number of images to generate
        device: Device to run generation on
        output_dir: Directory to save images
    """
    gen.eval()
    alpha = 1.0

    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIM).to(DEVICE)
            img = gen(noise, alpha, steps)
            if not os.path.exists(f'saved_examples/step{steps}'):
                os.makedirs(f'saved_examples/step{steps}')
            save_image(img * 0.5 + 0.5, f"saved_examples/step{steps}/img_{i}.png")
    gen.train()


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    """
    Calculate the gradient penalty for WGAN-GP.

    Args:
        critic: Discriminator model
        real: Real images
        fake: Generated images
        alpha: Blending factor
        train_step: Current resolution step
        device: Device to run calculation on

    Returns:
        gradient_penalty: Calculated gradient penalty
    """
    BATCH_SIZE, C, H, W = real.shape

    # Random interpolation between real and fake
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen):
    """
    Training function for one epoch.

    Args:
        critic: Discriminator model
        gen: Generator model
        loader: DataLoader
        dataset: Dataset
        step: Current resolution step
        alpha: Fade-in factor
        opt_critic: Discriminator optimizer
        opt_gen: Generator optimizer
        lambda_gp: Gradient penalty weight
        device: Device to train on
        writer: TensorBoard writer (optional)
        epoch: Current epoch

    Returns:
        alpha: Updated alpha value
    """
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)

        fake = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, device=DEVICE)
        loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
        )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
                (PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

    return alpha


if __name__ == "__main__":
    gen = Generator(
        Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG
    ).to(DEVICE)
    critic = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)

    # initialize optimizers
    opt_gen = optim.Adam([{"params": [param for name, param in gen.named_parameters() if "map" not in name]},
                          {"params": gen.map.parameters(), "lr": 1e-5}], lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99)
    )

    gen.train()
    critic.train()

    step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5  # start with very low alpha
        loader, dataset = get_loader(4 * 2 ** step)
        print(f"Current image size: {4 * 2 ** step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen
            )

        generate_examples(gen, step)
        step += 1  # progress to the next img size
