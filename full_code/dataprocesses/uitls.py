# ==========================================================================
# Copyright (c) 2012-2024 Anonymous Authors

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==========================================================================

from typing import List, Optional, Union

import torch


def transform_flatten(x: torch.Tensor) -> torch.Tensor:
    """ [C,H,W] -> [C,H*W] """
    return torch.reshape(x, [x.shape[0], -1])


def transpose_sequence(x: torch.Tensor) -> torch.Tensor:
    """ [C,T] -> [T, C] """
    return torch.transpose(x, 0, 1)


def pad_sequence(batch: List[torch.Tensor], size: Union[int, None] = None):
    """
    Make all tensors in a batch the same length by padding with zeros.
    Note that padding is often done in DataLoader, not in Dataset.

    # Args
    - batch: List with len=batch_size. Tensors has shape=[T, num_channels].
    - size: If None, sequence length = max length in the batch. If int,
      sequence length = size (size must be larger than the max sequence
      length in the batch). Default is None.
    """
    if size is None:
        raise NotImplementedError(
            "In this case, x_mask is not currently implemetned.")
    duration0, num_channels = batch[0].shape

    # Pad the first seq to the desired length (='size')
    # batch[0].shape becomes [num ch, duration=size].
    if size is not None:
        mask0 = size - duration0
        assert mask0 >= 0
        batch[0] = torch.nn.ConstantPad1d((0, mask0), 0.)(
            batch[0].transpose(0, 1)  # [C, duration0]
        ).transpose(0, 1)  # [C, T=size] -> transpose -> [T=size, C]

    len_mask = [size - item.shape[0] for item in batch]
    len_mask[0] = mask0

    batch_pad: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)
    # This function (with batch_first=True) returns a Tensor of size
    # B x T x * where T is the length of the longest sequence.
    # Input shape = [seq length, *].

    # Generate padding mask
    # mask.shape = [B, T=size, num_channels]
    mask = torch.ones_like(batch_pad, requires_grad=False)
    for i, it_len in enumerate(len_mask):  # len(len_mask)=batch_size
        if it_len == 0:
            continue  # bcz mask0 is it
        mask[i, -it_len:, :] = torch.zeros(it_len, num_channels)

    # Concat batch and mask
    batch_cat = torch.cat([batch_pad, mask], dim=0)  # [2*B, T, num_channes]

    return batch_cat


def collate_fn_SpeechCommands(batch: torch.Tensor, size: Optional[int] = None):
    """
    # Args
    - batch: List with len=batch_size. Tensors has shape=[T, num_channels].
    - size: If None, sequence length = max length in the batch. If int,
      sequence length = size (size must be larger than the max sequence
      length in the batch). Default is None.

    # Returns
    - tensors: A batch of waveforms with shape [batch size, num channels, fixed duration].
    - targets: A batch of integer labels.
    """
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [waveform]  # [T, C]
        targets += [label]  # scalar

    # Group the list of tensors into a batched tensor
    # tensors.shape = [batch_size, duration=size or max duration in a batch, num_channels]
    # targets.shape = [batch_size,]
    tensors = pad_sequence(tensors, size=size)
    targets = torch.stack(targets)  # type: ignore

    return tensors, targets


def label_to_index_SpeechCommands(word: str) -> torch.Tensor:
    """
    Returns the position of the word in labels.
    """
    list_labels = ['_unknown_', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight',
                   'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn',
                   'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
                   'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow',
                   'yes', 'zero']
    return torch.tensor(list_labels.index(word), dtype=torch.int)


def index_to_label_SpeechCommands(index: int) -> str:
    """
    Returns the word corresponding to the index in labels.
    This is the inverse of label_to_index.
    """
    list_labels = ['_unknown_', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight',
                   'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn',
                   'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
                   'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow',
                   'yes', 'zero']
    return list_labels[index]


class L2ISpeechCommands(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, word: str) -> torch.Tensor:
        """
        Returns the position of the word in labels.
        """
        list_labels = ['_unknown_', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight',
                       'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn',
                       'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
                       'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow',
                       'yes', 'zero']
        return torch.tensor(list_labels.index(word), dtype=torch.int64)
