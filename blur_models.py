import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=13):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return x * avg_out

class SelectiveStateSpaces(nn.Module):
    def __init__(self, in_channels):
        super(SelectiveStateSpaces, self).__init__()
        self.ln = nn.LayerNorm(in_channels)
        self.linear1 = nn.Linear(in_channels, in_channels)
        self.silu = nn.SiLU()
        self.ssm = nn.Linear(in_channels, in_channels)  # Placeholder for SSM layer
        self.linear2 = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        x = self.ln(x)
        x1 = self.silu(self.linear1(x))
        x2 = self.ssm(self.silu(x1))
        out = self.linear2(x1 * x2)
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)
        return out

class ALGBlock(nn.Module):
    def __init__(self, in_channels):
        super(ALGBlock, self).__init__()
        self.local_attention = ChannelAttention(in_channels)
        self.global_attention = SelectiveStateSpaces(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        local = self.local_attention(x)
        global_feat = self.global_attention(x)
        out = local + global_feat
        out = self.conv(out)
        return self.relu(out)

class ALGNet(nn.Module):
    def __init__(self, in_channels=3, num_blocks=9):
        super(ALGNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(*[ALGBlock(64) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.blocks(x)
        x = self.conv2(x)
        return x

class DeblurLoss(nn.Module):
    def __init__(self, epsilon=1e-3, delta=0.05, lam=0.1):
        super(DeblurLoss, self).__init__()
        self.epsilon = epsilon
        self.delta = delta
        self.lam = lam
        self.laplacian = Laplacian()

    def charbonnier_loss(self, pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2) + self.epsilon ** 2)

    def edge_loss(self, pred, target):
        pred_edges = self.laplacian(pred)
        target_edges = self.laplacian(target)
        return torch.sqrt(torch.mean((pred_edges - target_edges) ** 2) + self.epsilon ** 2)

    def frequency_loss(self, pred, target):
        pred_fft = torch.fft.fft2(pred, norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")
        return torch.mean(torch.abs(pred_fft - target_fft))

    def forward(self, pred, target):
        char_loss = self.charbonnier_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        freq_loss = self.frequency_loss(pred, target)
        return char_loss + self.delta * edge_loss + self.lam * freq_loss


class Laplacian(nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()
        kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.filter = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.filter.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size * channels, 1, height, width)
        x = self.filter(x)
        return x.view(batch_size, channels, height, width)


# Example usage
if __name__ == "__main__":
    model = ALGNet()
    input_tensor = torch.randn(1, 3, 256, 256)  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
