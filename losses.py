"""
Custom Loss Functions for Color Fidelity in Inpainting.

Includes:
- ColorConsistencyLoss: Perceptual color loss in LAB space
- HistogramMatchingLoss: Differentiable histogram matching
- FrequencySeparatedLoss: Separate low-freq (color) and high-freq (structure) losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to LAB color space for perceptual color comparison.
    
    Args:
        rgb: Tensor [B, 3, H, W] in range [-1, 1]
    
    Returns:
        lab: Tensor [B, 3, H, W] in LAB space (L: 0-100, a/b: -128 to 127)
    """
    # Normalize to [0, 1]
    rgb = (rgb + 1) / 2.0
    rgb = torch.clamp(rgb, 0, 1)
    
    # Apply gamma correction (sRGB to linear)
    mask = rgb > 0.04045
    rgb_linear = torch.where(
        mask,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92
    )
    
    # RGB to XYZ transformation matrix (D65 illuminant)
    # Shape: [3, 3]
    transform = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device, dtype=rgb.dtype)
    
    # Apply transformation: [B, 3, H, W] -> [B, H, W, 3] for matmul
    rgb_flat = rgb_linear.permute(0, 2, 3, 1)  # [B, H, W, 3]
    xyz = torch.matmul(rgb_flat, transform.T)  # [B, H, W, 3]
    
    # XYZ to LAB
    # Reference white D65
    ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device, dtype=rgb.dtype)
    xyz_normalized = xyz / ref_white
    
    # f(t) function for LAB conversion
    delta = 6.0 / 29.0
    delta_cube = delta ** 3
    mask = xyz_normalized > delta_cube
    f_xyz = torch.where(
        mask,
        xyz_normalized ** (1.0 / 3.0),
        xyz_normalized / (3.0 * delta ** 2) + 4.0 / 29.0
    )
    
    # Compute LAB channels
    L = 116.0 * f_xyz[..., 1] - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    
    lab = torch.stack([L, a, b], dim=-1).permute(0, 3, 1, 2)  # [B, 3, H, W]
    return lab


class ColorConsistencyLoss(nn.Module):
    """
    Perceptual color loss in LAB space.
    
    Computes color distance that aligns with human perception.
    Weights chroma (a, b) channels more than lightness for color fidelity.
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        weight_L: float = 1.0,  # Lightness weight
        weight_ab: float = 2.0,  # Chroma weights (color is more important)
    ):
        super().__init__()
        self.reduction = reduction
        self.weight_L = weight_L
        self.weight_ab = weight_ab
    
    def forward(
        self,
        pred: torch.Tensor,  # Predicted image [B, 3, H, W] in [-1, 1]
        target: torch.Tensor,  # Target/reference [B, 3, H, W] in [-1, 1]
        mask: torch.Tensor = None,  # Optional mask [B, 1, H, W]
    ) -> torch.Tensor:
        """Compute weighted LAB distance."""
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)
        
        # Separate L and ab channels
        L_loss = F.mse_loss(pred_lab[:, 0:1], target_lab[:, 0:1], reduction="none")
        ab_loss = F.mse_loss(pred_lab[:, 1:3], target_lab[:, 1:3], reduction="none")
        
        # Weighted combination
        loss = self.weight_L * L_loss + self.weight_ab * ab_loss.mean(dim=1, keepdim=True)
        
        # Apply mask if provided
        if mask is not None:
            # Resize mask to match loss spatial dims if needed
            if mask.shape[-2:] != loss.shape[-2:]:
                mask = F.interpolate(mask, size=loss.shape[-2:], mode="nearest")
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HistogramMatchingLoss(nn.Module):
    """
    Loss that encourages histogram matching between reference and output colors.
    
    Uses soft histogram binning for differentiability.
    Computes Earth Mover's Distance approximation via CDF L1 distance.
    """
    
    def __init__(self, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins
        # Pre-compute bin centers
        self.register_buffer('bins', torch.linspace(0, 1, num_bins))
    
    def compute_histogram(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute soft histogram using differentiable Gaussian binning.
        
        Args:
            x: [B, 3, H, W] in [-1, 1]
            mask: Optional [B, 1, H, W] in [0, 1]
        
        Returns:
            hist: [B, 3, num_bins] normalized histograms
        """
        # Normalize to [0, 1]
        x_normalized = (x + 1) / 2.0
        x_normalized = torch.clamp(x_normalized, 0, 1)
        
        # Flatten spatial dims
        B, C = x.shape[:2]
        x_flat = x_normalized.reshape(B, C, -1)  # [B, 3, HW]
        
        # Get bins on correct device
        bins = self.bins.to(x.device)
        sigma = 1.0 / self.num_bins  # Gaussian width
        
        # Soft assignment to bins via Gaussian
        # x_flat: [B, C, HW], bins: [num_bins]
        diff = x_flat.unsqueeze(-1) - bins  # [B, C, HW, num_bins]
        weights = torch.exp(-0.5 * (diff / sigma) ** 2)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(B, 1, -1, 1)  # [B, 1, HW, 1]
            weights = weights * mask_flat
        
        # Sum over spatial dimension
        hist = weights.sum(dim=2)  # [B, C, num_bins]
        
        # Normalize to probability distribution
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)
        
        return hist
    
    def forward(
        self,
        pred: torch.Tensor,  # [B, 3, H, W] in [-1, 1]
        target: torch.Tensor,  # Reference [B, 3, H, W] in [-1, 1]
        mask: torch.Tensor = None,  # Optional [B, 1, H, W]
    ) -> torch.Tensor:
        """
        Compute histogram matching loss (Earth Mover's Distance approximation).
        """
        pred_hist = self.compute_histogram(pred, mask)
        target_hist = self.compute_histogram(target)  # No mask for reference
        
        # Compute CDFs
        pred_cdf = pred_hist.cumsum(dim=-1)
        target_cdf = target_hist.cumsum(dim=-1)
        
        # EMD approximation via L1 on CDFs
        return F.l1_loss(pred_cdf, target_cdf)


class FrequencySeparatedLoss(nn.Module):
    """
    Separate loss for low-frequency (color) and high-frequency (texture) components.
    
    Low frequencies capture overall color distribution.
    High frequencies capture edges and fine details.
    
    Use this to:
    - Match REFERENCE colors in low frequencies
    - Match TARGET structure in high frequencies
    """
    
    def __init__(
        self,
        low_freq_weight: float = 2.0,  # Emphasize color matching
        high_freq_weight: float = 1.0,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight
        self.kernel_size = kernel_size
        
        # Pre-compute Gaussian kernel
        sigma = kernel_size / 6.0
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = g / g.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        # [1, 1, K, K]
        self.register_buffer('kernel', kernel_2d.unsqueeze(0).unsqueeze(0))
    
    def gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to extract low-frequency component."""
        B, C, H, W = x.shape
        kernel = self.kernel.to(x.device, x.dtype)
        # Expand kernel for all channels
        kernel = kernel.expand(C, 1, -1, -1)
        
        # Apply depthwise convolution
        return F.conv2d(
            x, kernel, padding=self.kernel_size // 2, groups=C
        )
    
    def forward(
        self,
        pred: torch.Tensor,  # Predicted [B, 3, H, W]
        target: torch.Tensor,  # Ground truth [B, 3, H, W]
        reference: torch.Tensor,  # Color reference [B, 3, H, W]
        mask: torch.Tensor = None,  # Optional [B, 1, H, W]
    ) -> torch.Tensor:
        """
        Compute frequency-separated loss.
        
        Args:
            pred: Model prediction
            target: Ground truth (for structure)
            reference: Color reference (for color matching)
            mask: Inpainting mask
        """
        # Separate frequencies
        pred_low = self.gaussian_blur(pred)
        target_low = self.gaussian_blur(target)
        reference_low = self.gaussian_blur(reference)
        
        pred_high = pred - pred_low
        target_high = target - target_low
        
        # Low-frequency loss: match reference colors
        low_loss = F.mse_loss(pred_low, reference_low, reduction="none")
        
        # High-frequency loss: match target structure
        high_loss = F.mse_loss(pred_high, target_high, reduction="none")
        
        # Combine
        total_loss = self.low_freq_weight * low_loss + self.high_freq_weight * high_loss
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape[-2:] != total_loss.shape[-2:]:
                mask = F.interpolate(mask, size=total_loss.shape[-2:], mode="nearest")
            total_loss = total_loss * mask
            return total_loss.sum() / (mask.sum() * total_loss.shape[1] + 1e-8)
        
        return total_loss.mean()


if __name__ == "__main__":
    # Quick test
    print("Testing color losses...")
    
    B, C, H, W = 2, 3, 64, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy tensors in [-1, 1]
    pred = torch.randn(B, C, H, W, device=device) * 0.5
    target = torch.randn(B, C, H, W, device=device) * 0.5
    reference = torch.randn(B, C, H, W, device=device) * 0.5
    mask = torch.ones(B, 1, H, W, device=device)
    
    # Test LAB conversion
    lab = rgb_to_lab(pred)
    print(f"LAB output shape: {lab.shape}")
    print(f"L range: [{lab[:, 0].min():.1f}, {lab[:, 0].max():.1f}]")
    
    # Test ColorConsistencyLoss
    color_loss = ColorConsistencyLoss()
    loss = color_loss(pred, target, mask)
    print(f"ColorConsistencyLoss: {loss.item():.4f}")
    
    # Test HistogramMatchingLoss
    hist_loss = HistogramMatchingLoss()
    loss = hist_loss(pred, target, mask)
    print(f"HistogramMatchingLoss: {loss.item():.4f}")
    
    # Test FrequencySeparatedLoss
    freq_loss = FrequencySeparatedLoss()
    loss = freq_loss(pred, target, reference, mask)
    print(f"FrequencySeparatedLoss: {loss.item():.4f}")
    
    # Test gradients
    pred.requires_grad = True
    loss = color_loss(pred, target, mask)
    loss.backward()
    print(f"Gradient norm: {pred.grad.norm().item():.4f}")
    
    print("✅ All loss tests passed!")
