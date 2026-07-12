"""Autoencoder convoluzionale + LSTM, in 6 varianti (1 o 2 layer conv, con o senza
ReLU, bottleneck a vettore o a sequenza ridotta).

Il decoder e' l'esatto speculare dell'encoder: stessa sequenza di layer letta al
contrario, con ConvTranspose1d che usa un `output_padding` calcolato per
riportare la lunghezza temporale esattamente a quella di input (15000 step),
qualunque sia la combinazione di kernel_size/stride/padding scelta dal trial.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import ArchitectureSpec, ARCHITECTURES

REQUIRED_MODEL_KEYS = ("n_filters", "kernel_size", "stride", "padding", "hidden_units", "latent_dim", "dropout")


def conv1d_output_length(length_in: int, kernel_size: int, stride: int, padding: int) -> int:
    return (length_in + 2 * padding - kernel_size) // stride + 1


def conv_transpose1d_output_padding(
    length_in: int, target_length_out: int, kernel_size: int, stride: int, padding: int
) -> int:
    """Calcola l'output_padding di ConvTranspose1d necessario per ottenere esattamente
    `target_length_out` a partire da `length_in`, con gli stessi kernel/stride/padding
    usati dal Conv1d che si vuole invertire."""
    base = (length_in - 1) * stride - 2 * padding + kernel_size
    output_padding = target_length_out - base
    if output_padding < 0 or output_padding >= stride:
        raise ValueError(
            f"Combinazione di iperparametri non invertibile esattamente "
            f"(length_in={length_in}, target={target_length_out}, kernel={kernel_size}, "
            f"stride={stride}, padding={padding} -> output_padding={output_padding})"
        )
    return output_padding


class Encoder(nn.Module):
    def __init__(self, spec: ArchitectureSpec, hp: dict[str, Any], input_len: int, n_features: int) -> None:
        super().__init__()
        self.spec = spec

        conv_layers: list[nn.Module] = []
        lengths = [input_len]
        for i in range(spec.n_conv_layers):
            in_channels = n_features if i == 0 else hp["n_filters"]
            conv_layers.append(
                nn.Conv1d(in_channels, hp["n_filters"], kernel_size=hp["kernel_size"], stride=hp["stride"], padding=hp["padding"])
            )
            if spec.use_activation:
                conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(hp["dropout"]))

            new_len = conv1d_output_length(lengths[-1], hp["kernel_size"], hp["stride"], hp["padding"])
            if new_len < 1:
                raise ValueError(
                    f"La sequenza si annulla dopo il layer conv {i}: lunghezza risultante {new_len}. "
                    "Ridurre kernel_size/stride o aumentare la lunghezza di input."
                )
            lengths.append(new_len)

        self.conv = nn.Sequential(*conv_layers)
        self.lengths = lengths  # [input_len, len_dopo_conv1, (len_dopo_conv2)]

        self.lstm = nn.LSTM(input_size=hp["n_filters"], hidden_size=hp["hidden_units"], num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(hp["dropout"])
        self.latent_proj = nn.Linear(hp["hidden_units"], hp["latent_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = x.transpose(1, 2)  # (batch, n_features, seq_len)
        x = self.conv(x)  # (batch, n_filters, reduced_len)
        x = x.transpose(1, 2)  # (batch, reduced_len, n_filters)

        out, (h_n, _) = self.lstm(x)
        if self.spec.latent_mode == "vector":
            z = self.latent_proj(self.dropout(h_n[-1]))  # (batch, latent_dim)
        else:
            z = self.latent_proj(self.dropout(out))  # (batch, reduced_len, latent_dim)
        return z


class Decoder(nn.Module):
    def __init__(self, spec: ArchitectureSpec, hp: dict[str, Any], encoder_lengths: list[int], n_features: int) -> None:
        super().__init__()
        self.spec = spec
        self.reduced_len = encoder_lengths[-1]

        self.latent_expand = nn.Linear(hp["latent_dim"], hp["hidden_units"])
        self.lstm = nn.LSTM(input_size=hp["hidden_units"], hidden_size=hp["hidden_units"], num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(hp["dropout"])
        self.pre_conv_proj = nn.Linear(hp["hidden_units"], hp["n_filters"])

        transpose_layers: list[nn.Module] = []
        n_layers = spec.n_conv_layers
        for i in reversed(range(n_layers)):
            length_in = encoder_lengths[i + 1]
            target_len = encoder_lengths[i]
            out_channels = n_features if i == 0 else hp["n_filters"]
            output_padding = conv_transpose1d_output_padding(
                length_in, target_len, hp["kernel_size"], hp["stride"], hp["padding"]
            )
            transpose_layers.append(
                nn.ConvTranspose1d(
                    hp["n_filters"],
                    out_channels,
                    kernel_size=hp["kernel_size"],
                    stride=hp["stride"],
                    padding=hp["padding"],
                    output_padding=output_padding,
                )
            )
            if spec.use_activation and i != 0:
                transpose_layers.append(nn.ReLU())

        self.deconv = nn.Sequential(*transpose_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.spec.latent_mode == "vector":
            h0 = self.latent_expand(z)  # (batch, hidden_units)
            dec_input = h0.unsqueeze(1).expand(-1, self.reduced_len, -1)  # (batch, reduced_len, hidden_units)
        else:
            dec_input = self.latent_expand(z)  # (batch, reduced_len, hidden_units)

        out, _ = self.lstm(dec_input)  # (batch, reduced_len, hidden_units)
        out = self.pre_conv_proj(self.dropout(out))  # (batch, reduced_len, n_filters)
        out = out.transpose(1, 2)  # (batch, n_filters, reduced_len)
        out = self.deconv(out)  # (batch, n_features, seq_len)
        return out.transpose(1, 2)  # (batch, seq_len, n_features)


class Autoencoder(nn.Module):
    def __init__(self, spec: ArchitectureSpec, hp: dict[str, Any], input_len: int, n_features: int) -> None:
        super().__init__()
        missing = [k for k in REQUIRED_MODEL_KEYS if k not in hp]
        if missing:
            raise ValueError(f"Iperparametri mancanti per costruire il modello: {missing}")

        self.encoder = Encoder(spec, hp, input_len, n_features)
        self.decoder = Decoder(spec, hp, self.encoder.lengths, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def build_model(arch_name: str, hp: dict[str, Any], input_len: int, n_features: int) -> Autoencoder:
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Architettura sconosciuta: {arch_name}. Disponibili: {list(ARCHITECTURES)}")
    spec = ARCHITECTURES[arch_name]
    return Autoencoder(spec, hp, input_len, n_features)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
