import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    """
    FiLM layer: Feature-wise Linear Modulation.
    Generates scaling (gamma) and shifting (beta) coefficients from a conditioning vector and applies them to an input feature map.
    """
    def __init__(self, cond_dim, num_features):
        """
        cond_dim: dimension of conditioning vector (e.g., tissue embedding length).
        num_features: number of feature channels to modulate.
        """
        super(FiLM, self).__init__()
        self.linear = nn.Linear(cond_dim, 2 * num_features)  # outputs [gamma, beta] concatenated

    def forward(self, x, cond):
        """
        x: input feature map tensor of shape [batch, C, ...] (C = num_features).
        cond: conditioning tensor of shape [batch, cond_dim].
        Returns: modulated feature map of same shape as x, where each channel is scaled and shifted.
        """
        # Compute FiLM parameters
        gamma_beta = self.linear(cond.float())  # shape [batch, 2*num_features]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # If x is convolutional feature map (batch, C, L) or (batch, C, H, W), reshape gamma and beta to [batch, C, 1...] to broadcast
        # Here assuming 1D conv outputs (batch, C, L):
        if x.dim() == 3:  # (N, C, L)
            gamma = gamma.unsqueeze(2)  # (N, C, 1)
            beta = beta.unsqueeze(2)    # (N, C, 1)
        elif x.dim() == 4:  # (N, C, H, W) for 2D convs if any
            gamma = gamma.unsqueeze(2).unsqueeze(3)  # (N, C, 1, 1)
            beta = beta.unsqueeze(2).unsqueeze(3)
        # Apply modulation
        return x * gamma + beta

class CNNFiLMModel(nn.Module):
    """
    CNN-based model with optional FiLM modulation and additional features (RBP/tRNA).
    This model processes 5' and 3' UTR sequences through separate CNN branches, applies FiLM using tissue embeddings, 
    and then concatenates their outputs with extra features for final prediction.
    """
    def __init__(self, seq_input_channels=4, conv_channels=[64, 64], conv_kernels=[8, 4], 
                 include_film=True, include_rbp=True, include_trna=True, tissue_embed_dim=None, extra_feat_dim=None):
        """
        seq_input_channels: number of input channels for sequence (e.g., 4 for one-hot ACGT).
        conv_channels: list of output channel counts for each convolutional layer.
        conv_kernels: list of kernel sizes for each conv layer (should match length of conv_channels).
        include_film: if True, use FiLM modulation with tissue embeddings.
        include_rbp: if True, model expects RBP features in extra input.
        include_trna: if True, model expects tRNA features in extra input.
        tissue_embed_dim: dimension of tissue embedding vector (required if include_film=True).
        extra_feat_dim: dimension of extra numeric features vector (if any; 0 if none).
        """
        super(CNNFiLMModel, self).__init__()
        self.include_film = include_film
        self.include_rbp = include_rbp
        self.include_trna = include_trna
        # Define CNN branches for 5' UTR and 3' UTR
        self.conv_layers_5 = nn.ModuleList()
        self.conv_layers_3 = nn.ModuleList()
        self.film_layers_5 = nn.ModuleList() if include_film else None
        self.film_layers_3 = nn.ModuleList() if include_film else None
        in_channels = seq_input_channels
        for i, out_channels in enumerate(conv_channels):
            kernel = conv_kernels[i] if i < len(conv_kernels) else conv_kernels[-1]
            # Conv1d for sequence data
            self.conv_layers_5.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding='valid'))
            self.conv_layers_3.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding='valid'))
            # After first conv, subsequent convs take input_channels = previous out_channels
            in_channels = out_channels
            if include_film:
                # Create a FiLM layer for this conv's output channels
                self.film_layers_5.append(FiLM(tissue_embed_dim, out_channels))
                self.film_layers_3.append(FiLM(tissue_embed_dim, out_channels))
        # After conv layers, we'll apply global max pooling to get a fixed-length vector from each branch
        # Compute final feature dimensions
        conv_out_channels = conv_channels[-1] if len(conv_channels) > 0 else in_channels
        # conv_out_channels is channels from last conv; assume global pooling reduces sequence dim to 1
        # Combined conv output from both 5' and 3' branches = 2 * conv_out_channels
        combined_conv_dim = 2 * conv_out_channels
        # Determine extra feature dim (if not provided explicitly)
        if extra_feat_dim is None:
            extra_feat_dim = 0
            if include_rbp:
                extra_feat_dim += 4  # 4 RBP features
            if include_trna:
                extra_feat_dim += 2  # 2 tRNA features
        self.extra_feat_dim = extra_feat_dim
        # Define fully connected layers
        # Input to first FC is combined_conv_dim + extra_feat_dim
        fc_input_dim = combined_conv_dim + extra_feat_dim
        self.fc1 = nn.Linear(fc_input_dim, 128)  # hidden layer size 128 (can be configured)
        self.fc2 = nn.Linear(128, 1)  # final output (regression)
        # Optionally include dropout or batchnorm if needed (not explicitly requested, so omitted)

    def forward(self, seq5, seq3, tissue_vec, extra_features):
        """
        seq5: tensor of shape [batch, seq_len5, channels] or [batch, channels, seq_len5] (depending on how sequence is formatted).
        seq3: tensor of shape [batch, channels, seq_len3].
        tissue_vec: tensor [batch, tissue_embed_dim] (conditioning vector for FiLM).
        extra_features: tensor [batch, extra_feat_dim] (additional numeric features like RBP/tRNA).
        """
        # If sequences are provided as [batch, seq_len, channels], transpose to [batch, channels, seq_len] for Conv1d
        if seq5.dim() == 2 or seq5.dim() == 3:
            # Assume shape might be [batch, seq_len] with numeric labels or [batch, seq_len, channels]
            # If one-hot as [batch, seq_len, 4], transpose to [batch, 4, seq_len]
            if seq5.dim() == 3:
                seq5_in = seq5.permute(0, 2, 1)  # (N, channels, L5)
                seq3_in = seq3.permute(0, 2, 1)  # (N, channels, L3)
            else:
                # If seq given as indices or so (not one-hot), an embedding layer would be needed (not implemented here).
                seq5_in = seq5.unsqueeze(1)  # as a single-channel sequence
                seq3_in = seq3.unsqueeze(1)
        else:
            seq5_in = seq5
            seq3_in = seq3
        # Pass through conv layers for 5' and 3' sequences
        x5 = seq5_in
        x3 = seq3_in
        for i, conv in enumerate(self.conv_layers_5):
            x5 = conv(x5)            # conv
            x5 = F.relu(x5)          # activation
            if self.include_film:
                x5 = self.film_layers_5[i](x5, tissue_vec)  # FiLM modulation
        for i, conv in enumerate(self.conv_layers_3):
            x3 = conv(x3)
            x3 = F.relu(x3)
            if self.include_film:
                x3 = self.film_layers_3[i](x3, tissue_vec)
        # Global max pooling across sequence length dimension for each branch
        # Assuming conv outputs shape [batch, channels, seq_length]
        x5 = torch.max(x5, dim=-1)[0]  # shape [batch, conv_out_channels]
        x3 = torch.max(x3, dim=-1)[0]  # shape [batch, conv_out_channels]
        # Concatenate 5' and 3' features
        conv_combined = torch.cat([x5, x3], dim=1)  # [batch, 2*conv_out_channels]
        # Concatenate extra features if present
        if self.extra_feat_dim > 0:
            # If extra_features tensor is empty (0-dim), skip concatenation
            if extra_features.shape[1] == 0:
                combined = conv_combined
            else:
                combined = torch.cat([conv_combined, extra_features], dim=1)
        else:
            combined = conv_combined
        # Fully connected layers
        hidden = F.relu(self.fc1(combined))
        output = self.fc2(hidden)
        return output  # shape [batch, 1]
