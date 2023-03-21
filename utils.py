import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x, device="cuda"):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.to(device, non_blocking=True)
    return torch.autograd.Variable(x)

def parse_batch(batch, device):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded, device).long()
        input_lengths = to_gpu(input_lengths, device).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded, device).float()
        gate_padded = to_gpu(gate_padded, device).float()
        output_lengths = to_gpu(output_lengths, device).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))