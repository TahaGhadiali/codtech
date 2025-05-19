import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_image(image_path, max_size=None, shape=None):
    """Load and preprocess an image"""
    image = Image.open(image_path).convert('RGB')
    
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.LANCZOS)
    
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    image = transform(image).unsqueeze(0)
    return image.to(device)

def imshow(tensor, title=None):
    """Convert tensor to image and display"""
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
    
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        # Use pretrained VGG19 with features only (no fully connected layers)
        self.features = models.vgg19(pretrained=True).features.to(device).eval()
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad_(False)
        
        # Layers for style and content extraction
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        # Mean and std for normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    
    def forward(self, x):
        # Normalize input
        x = (x - self.mean) / self.std
        
        content_features = []
        style_features = []
        
        # Extract features at specified layers
        for name, layer in self.features.named_children():
            x = layer(x)
            if name in self.content_layers:
                content_features.append(x)
            if name in self.style_layers:
                style_features.append(x)
        
        return content_features, style_features


def content_loss(content_features, target_features):
    """Calculate content loss (MSE between feature representations)"""
    return F.mse_loss(content_features[0], target_features[0])

def gram_matrix(input):
    """Compute Gram matrix for style representation"""
    batch_size, channels, height, width = input.size()
    features = input.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channels * height * width)

def style_loss(style_features, target_features):
    """Calculate style loss (MSE between Gram matrices)"""
    loss = 0
    for style_feat, target_feat in zip(style_features, target_features):
        style_gram = gram_matrix(style_feat)
        target_gram = gram_matrix(target_feat)
        loss += F.mse_loss(style_gram, target_gram)
    return loss


def neural_style_transfer(content_path, style_path, output_path, 
                         max_size=512, iterations=500, 
                         content_weight=1, style_weight=1e6,
                         show_progress=True):
    """
    Perform neural style transfer
    
    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_path: Path to save output image
        max_size: Maximum dimension for images
        iterations: Number of optimization iterations
        content_weight: Weight for content loss
        style_weight: Weight for style loss
        show_progress: Whether to display progress images
    """
    
    # Load images
    content = load_image(content_path, max_size=max_size)
    style = load_image(style_path, shape=content.shape[-2:])
    
    # Initialize target image as content image with requires_grad
    target = content.clone().requires_grad_(True).to(device)
    
    # Initialize model
    model = VGG19()
    
    # Get features for content and style images
    content_features, _ = model(content)
    _, style_features = model(style)
    
    # Optimizer (LBFGS works well for this problem)
    optimizer = optim.LBFGS([target])
    
    # Run style transfer
    for i in range(iterations):
        def closure():
            # Zero gradients
            optimizer.zero_grad()
            
            # Get features for target image
            target_content_features, target_style_features = model(target)
            
            # Calculate losses
            c_loss = content_loss(content_features, target_content_features)
            s_loss = style_loss(style_features, target_style_features)
            total_loss = content_weight * c_loss + style_weight * s_loss
            
            # Backpropagate
            total_loss.backward()
            
            # Print progress
            if i % 50 == 0 and show_progress:
                print(f"Iteration {i}:")
                print(f"Content Loss: {c_loss.item():.4f}")
                print(f"Style Loss: {s_loss.item():.4f}")
                print(f"Total Loss: {total_loss.item():.4f}")
                print("----------------------")
                imshow(target.detach(), f"Iteration {i}")
            
            return total_loss
        
        optimizer.step(closure)
    
    # Save final image
    final_image = target.detach().cpu().squeeze()
    final_image = transforms.ToPILImage()(final_image)
    final_image.save(output_path)
    print(f"Style transfer complete! Image saved to {output_path}")
    
    return final_image


# Example usage
if __name__ == "__main__":
    content_path = "images/content/content.jpg"  # Replace with your content image path
    style_path = "images/style/image.png"     # Replace with your style image path
    output_path = "images/output/output.jpg"   # Output path
    
    # Run style transfer
    result = neural_style_transfer(
        content_path=content_path,
        style_path=style_path,
        output_path=output_path,
        max_size=512,
        iterations=300,
        content_weight=1,
        style_weight=1e6
    )
    
    # Display result
    plt.imshow(result)
    plt.axis('off')
    plt.show()