# ==========================================================================
# Copyright (c) 2012-2024 Taiki Miyagawa

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

"""
# Coding rule for collocation point datasets
All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
and Tensor y with shape [batch, 1+dim_output].
In a mini-batch, X and X_bc (and X_data if available) are included.
X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
They have the shape of [dim_input,].
Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from scipy.stats import qmc
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from utils.log_config import get_my_logger
from utils.misc import (latin_hypercube_sampling_with_min_max,
                        rand_with_lengths_mins, rand_with_min_max)

logger = get_my_logger(__name__)

NUM_POINTS_MAX = 100_000
NUM_VAL_MAX = int(1e5)  # num of val data
NUM_TE_MAX = int(1e5)  # max num of test data
# assert NUM_VAL_MAX != NUM_TE_MAX, "generate_grid_data requires this; otherwise val data = test data."
RATIO_BULK_BDY_FOR_VA_TE = 1.0  # 1.0  # num cp bulk vs boundary for val & test set


def random_sampling_cp(
        sizes: Tuple[Tuple[Union[int, float], Union[int, float]], ...],
        boundaries: Tuple[Tuple[Union[int, float, None, List], Union[int, float, None, List]], ...],
        dim_output: int,
        num_points_bulk: int, num_points_one_bdy: int,
        batch_size_bulk: int, batch_size_boundary: int) -> Tuple[Tensor, Tensor, int]:
    """
    20230706: num_points_one_bdy = batch_size_boundary = 0 is now OK.

    # Args
    - sizes: len(sizes) = dim_input. The sizes of the physical system.
    - boundaries: len(boundaries) = num of boundaries. len(boundaries[0]) = dim_input.
        Specifies boundaries, to which initial and/or boundary conditions are to be applied.
    - dim_output: Dimension of u(X).
    - num_points_bulk: Number of training collocation points generated within a single epoch.
      For bulk collocation points.
    - num_points_one_bdy: Number of training collocation points generated within a single epoch.
      For a single boundary collocation points.
    - batch_size_bulk: Batch size of the bulk collocation points.
    - batch_size_boundary: Batch size of the boundary collocation points.

    # Returns
    - ds_tr: Dataset Tesnor. Collocation points.
    - ds_tr_label: Label dataset Tensor.
    - num_points: Number of datapoints in ds_tr and ds_tr_label.

    # Remarks
    - ds: Dataset Tesnor. Bulk collocation points.
    - ds_label: Label dataset Tensor.
    - ds_boundary: Dataset Tesnor. Boundary collocation points.
    - ds_label: Label dataset Tensor.
    """
    if len(boundaries) == 0:
        assert batch_size_boundary == 0

    # 1. Bulk collocation points and labels
    ds_bulk_ls: List = []
    ds_bulk_ls_append = ds_bulk_ls.append
    for s in sizes:  # dim loop
        ds_bulk_ls_append(rand_with_min_max(
            [num_points_bulk, 1], s[0], s[1]))
    ds_bulk = torch.cat(ds_bulk_ls, dim=1)  # [num_points_bulk,dim_input]
    ds_bulk_label = torch.tensor(
        [[0.] + [torch.nan] * dim_output] * num_points_bulk).\
        reshape((-1, 1+dim_output))  # [num_points,1+dim_output]

    # Return bulk dataset if no bdy data is needed
    if batch_size_boundary == 0:
        assert num_points_one_bdy == 0
        ds_tr = ds_bulk
        ds_tr_label = ds_bulk_label
        num_points = ds_tr.shape[0]
        return ds_tr, ds_tr_label, num_points

    # 2. Boundary collocation points and labels
    ds_boundary_ls: List[Tensor] = []
    ds_boundary_ls_append = ds_boundary_ls.append
    for it_b in boundaries:  # boundary loop
        it_ds: List = []
        it_ds_append = it_ds.append
        for s, b in zip(sizes, it_b):  # dim loop
            if b is None:
                it_ds_append(rand_with_min_max(  # type: ignore
                    [num_points_one_bdy, 1], s[0], s[1]))
            elif type(b) in [int, float]:
                it_ds_append(torch.stack(
                    [torch.tensor(b)] * num_points_one_bdy, dim=0).unsqueeze(dim=1))
            elif type(b) in [tuple, list]:
                it_ds_append(rand_with_min_max(
                    [num_points_one_bdy, 1], b[0], b[1]))  # type:ignore
            else:
                raise ValueError(
                    f"Invalid boundaries. Got\nb = {b},\nboundaries = {boundaries}")

        # Shape = [num_points_one_bdy, dim_input]
        it_ds_t = torch.cat(it_ds, dim=1)
        ds_boundary_ls_append(it_ds_t)

    ds_boundary = torch.cat(
        ds_boundary_ls, dim=0)  # [<=num_points_boundary, dim_input]
    ds_boundary_label = torch.tensor(
        [[1.] + [torch.nan]*dim_output] * ds_boundary.shape[0]).\
        reshape((-1, 1+dim_output)
                )  # [<=num_points_boundary, 1+dim_output]
    idx = torch.randperm(ds_boundary.shape[0])
    ds_boundary = ds_boundary[idx]
    ds_boundary_label = ds_boundary_label[idx]
    # Now we have ds, ds_label, ds_boundary, and ds_boundary_label.

    # 3. Batchwise balancing
    num_points_boundary = ds_boundary.shape[0]
    num_iter = int(min(num_points_bulk // batch_size_bulk,
                       num_points_boundary // batch_size_boundary))
    ds_tr_ls: List = []
    ds_tr_label_ls: List = []
    ds_tr_ls_append = ds_tr_ls.append
    ds_tr_label_ls_append = ds_tr_label_ls.append
    for i in range(num_iter):
        ds_tr_ls_append(
            ds_bulk[i * batch_size_bulk: (i + 1) * batch_size_bulk])
        ds_tr_label_ls_append(
            ds_bulk_label[i * batch_size_bulk: (i + 1) * batch_size_bulk])
        ds_tr_ls_append(
            ds_boundary[i * batch_size_boundary: (i + 1) * batch_size_boundary])
        ds_tr_label_ls_append(
            ds_boundary_label[i * batch_size_boundary: (i + 1) * batch_size_boundary])
    ds_tr = torch.cat(ds_tr_ls, dim=0)  # [<=num_points_max, dim_input]
    ds_tr_label = torch.cat(
        ds_tr_label_ls, dim=0)  # [<=num_points_max, 1+dim_output]

    num_points = ds_tr.shape[0]

    return ds_tr, ds_tr_label, num_points


def latin_hypercube_sampling_cp(  # Deprecated
        sizes: Tuple[Tuple[Union[int, float], Union[int, float]], ...],
        boundaries: Tuple[Tuple[Union[int, float, None, List], Union[int, float, None, List]], ...],
        dim_output: int,
        num_points_bulk: int, num_points_one_bdy: int
) -> Tuple[Tensor, Tensor, Union[Tensor, None], Union[Tensor, None]]:
    """
    Latin hypercube sampling for collocation points.
    latin_hypercube_sampling_with_min_max used.

    # Dependencies
    latin_hypercube_sampling -> latin_hypercube_sampling_with_min_max -> latin_hypercube_sampling_cp -> latin_hypercube_sampling_cp_BBBatch

    # Args
    - sizes: len(sizes) = dim_input. The sizes of the physical system.
    - boundaries: len(boundaries) = num of boundaries. len(boundaries[0]) = dim_input.
        Specifies boundaries, to which initial and/or boundary conditions are to be applied.
    - dim_output: Dimension of u(X).
    - num_points_bulk: Number of training collocation points generated within a single epoch.
      For bulk collocation points.
    - num_points_one_bdy: Number of training collocation points generated within a single epoch.
      For a single boundary collocation points.

    # Returns
    - ds_bulk: Dataset Tesnor. Bulk collocation points.
    - ds_bulk_label: Label dataset Tensor.
    - ds_boundary: Dataset Tesnor. Boundary collocation points. Or None if num_points_one_bdy = 0.
    - ds_boundary_label: Label dataset Tensor. Or None if num_points_one_bdy = 0.
    """
    # 1. Bulk collocation points and labels
    ds_bulk_ls: List = []
    ds_bulk_ls_append = ds_bulk_ls.append
    for s in sizes:
        ds_bulk_ls_append(latin_hypercube_sampling_with_min_max(
            num_points_bulk, 1, s[0], s[1]))
    ds_bulk = torch.cat(
        ds_bulk_ls, dim=1)  # [num_points_bulk,dim_input]
    ds_bulk_label = torch.tensor(
        [[0.] + [torch.nan] * dim_output] * num_points_bulk).\
        reshape((-1, 1+dim_output))  # [num_points,1+dim_output]

    # Return bulk dataset if no bdy data is needed
    if num_points_one_bdy == 0:
        ds_tr = ds_bulk
        ds_tr_label = ds_bulk_label
        return ds_tr, ds_tr_label, None, None

    # 2. Boundary collocation points and labels
    ds_boundary_ls: List[Tensor] = []
    ds_boundary_ls_append = ds_boundary_ls.append
    for it_b in boundaries:  # boundary loop
        it_ds: List = []
        it_ds_append = it_ds.append
        for s, b in zip(sizes, it_b):  # dim loop
            if b is None:
                it_ds_append(latin_hypercube_sampling_with_min_max(
                    num_points_one_bdy, 1, s[0], s[1]))
            elif type(b) in [int, float]:
                it_ds_append(torch.stack(
                    [torch.tensor(b)] * num_points_one_bdy, dim=0).unsqueeze(dim=1))
            elif type(b) in [tuple, list]:
                it_ds_append(latin_hypercube_sampling_with_min_max(
                    num_points_one_bdy, 1, b[0], b[1]))  # type:ignore
            else:
                raise ValueError(
                    f"Invalid boundaries. Got\nb = {b},\nboundaries = {boundaries}")

        it_ds_t = torch.cat(
            it_ds, dim=1)  # [num_points_one_bdy, dim_input]
        ds_boundary_ls_append(it_ds_t)

    ds_boundary = torch.cat(
        ds_boundary_ls, dim=0)  # [<=num_points_boundary, dim_input]
    ds_boundary_label = torch.tensor(
        [[1.] + [torch.nan]*dim_output] * ds_boundary.shape[0]).\
        reshape((-1, 1+dim_output)
                )  # [<=num_points_boundary, 1+dim_output]
    idx = torch.randperm(ds_boundary.shape[0])
    ds_boundary = ds_boundary[idx]
    ds_boundary_label = ds_boundary_label[idx]

    return ds_bulk, ds_bulk_label, ds_boundary, ds_boundary_label


def latin_hypercube_sampling_cp_BBBatch(  # Deprecated
        sizes: Tuple[Tuple[Union[int, float], Union[int, float]], ...],
        boundaries: Tuple[Tuple[Union[int, float, None, List], Union[int, float, None, List]], ...],
        dim_output: int,
        num_points_bulk: int, num_boundaries: int,
        batch_size_bulk: int, batch_size_boundary: int) -> Tuple[Tensor, Tensor, int]:
    """
    Batchwise balancing of bulk vs. boundary.
    latin_hypercube_sampling_cp used.

    # Dependencies
    latin_hypercube_sampling -> latin_hypercube_sampling_with_min_max -> latin_hypercube_sampling_cp -> latin_hypercube_sampling_cp_BBBatch

    # Args
    - sizes: len(sizes) = dim_input. The sizes of the physical system.
    - boundaries: len(boundaries) = num of boundaries. len(boundaries[0]) = dim_input.
        Specifies boundaries, to which initial and/or boundary conditions are to be applied.
    - dim_output: Dimension of u(X).
    - num_points_bulk: Number of training collocation points generated within a single epoch.
      For bulk collocation points.
    - num_points_one_bdy: Number of training collocation points generated within a single epoch.
      For a single boundary collocation points.
    - batch_size_bulk: Batch size of the bulk collocation points.
    - batch_size_boundary: Batch size of the boundary collocation points.

    # Returns
    - ds_tr: Dataset Tesnor. Collocation points.
    - ds_tr_label: Label dataset Tensor.
    - num_points: Number of datapoints in ds_tr and ds_tr_label.

    # Remarks
    - ds: Dataset Tesnor. Bulk collocation points.
    - ds_label: Label dataset Tensor.
    - ds_boundary: Dataset Tesnor. Boundary collocation points.
    - ds_label: Label dataset Tensor.
    """

    # Batchwise balancing
    if num_boundaries == 0:
        batch_size_one_bdy = 0
    else:
        batch_size_one_bdy = int(batch_size_boundary // num_boundaries)
    num_iter = int(num_points_bulk // batch_size_bulk)
    assert num_iter > 0

    ds_tr_ls: List = []
    ds_tr_label_ls: List = []
    ds_tr_ls_append = ds_tr_ls.append
    ds_tr_label_ls_append = ds_tr_label_ls.append
    for _ in range(num_iter):
        it_ds_bulk, it_ds_bulk_label, it_ds_bdy, it_ds_bdy_label = latin_hypercube_sampling_cp(
            sizes, boundaries, dim_output, batch_size_bulk, batch_size_one_bdy)
        ds_tr_ls_append(it_ds_bulk)
        ds_tr_label_ls_append(it_ds_bulk_label)
        if batch_size_one_bdy == 0:
            assert it_ds_bdy is None
            continue
        else:
            ds_tr_ls_append(it_ds_bdy)  # type: ignore
            ds_tr_label_ls_append(it_ds_bdy_label)  # type: ignore
    ds_tr = torch.cat(ds_tr_ls, dim=0)  # [<=num_points_max, dim_input]
    ds_tr_label = torch.cat(
        ds_tr_label_ls, dim=0)  # [<=num_points_max, 1+dim_output]

    num_points = ds_tr.shape[0]

    return ds_tr, ds_tr_label, num_points


def generate_sparse_grid_data(
        num_points_grid: int,
        sizes: Tuple[Tuple[Union[int, float], Union[int, float]], ...],
        boundaries: Tuple[Tuple[Union[int, float, List, None], ...], ...],
        dim_output: int, dim_input: int, seed) -> Tuple[Tensor, Tensor]:
    """
    Currently not used. Previously used for val and test sets.
    Generate grid collocation points for validation or test sets.
    len(boundaries) = 0 (no boundary) is supported.

    # Args
    - num_points_grid: Number of validation or test datapoints.
    - sizes: See __init__.
    - boundaries: See __init__.
    - dim_output: An int.
    - dim_input: An int.
    - seed: Seed for numpy.

    # Returns
    - ds: Grid collocation points for validation or test.
            Includes both bulk and boundary collocation points.
            Tensor with shape [num_points_grid, dim_input]
    - ds_label: Labels with shape [num_points_grid, 1 + dim_output].
    """
    # Get random state and fix seed
    current_state_numpy = np.random.get_state()
    np.random.seed(seed)

    # Num of datapoints for bulk (X) and boundary (X_bc)
    if len(boundaries) != 0:
        num_points_bulk = int(RATIO_BULK_BDY_FOR_VA_TE * num_points_grid)
        num_points_boundary = num_points_grid - num_points_bulk
    else:
        num_points_bulk = num_points_grid
        num_points_boundary = 0

    # 1. Bulk collocation points and labels
    ds_bulk = torch.stack([
        torch.linspace(
            sizes[i][0],
            sizes[i][1],
            steps=num_points_bulk)[np.random.permutation(num_points_bulk)] for i in range(dim_input)],
        dim=1)  # on CPU, shape=[num_points_bulk, dim_input,]
    ds_bulk_label = torch.tensor(
        [[0.] + [torch.nan]*dim_output] * num_points_bulk).\
        reshape((-1, 1+dim_output))  # [num_points_bulk ,1+dim_out]

    # Return bulk dataset if no bdy data is needed
    if num_points_boundary == 0:
        ds = ds_bulk
        ds_label = ds_bulk_label
        return ds, ds_label

    # 2. Boundary collocation points and labels
    num_boundaries = len(boundaries)  # num of (straight or point) boundaries
    num_points_one_bdy = num_points_boundary // num_boundaries
    ds_boundary_ls: List[Tensor] = []
    ds_boundary_ls_append = ds_boundary_ls.append
    for it_b in boundaries:  # boundary loop
        it_ds: List = []
        it_ds_append = it_ds.append
        for s, b in zip(sizes, it_b):  # dim loop
            if b is None:
                it_ds_append(torch.linspace(
                    s[0], s[1], steps=num_points_one_bdy)[np.random.permutation(num_points_one_bdy)])
            elif type(b) in [int, float]:
                it_ds_append(torch.stack(
                    [torch.tensor(b)] * num_points_one_bdy, dim=0))
            elif type(b) in [tuple, list]:
                it_ds_append(torch.linspace(
                    b[0], b[1], steps=num_points_one_bdy)[np.random.permutation(num_points_one_bdy)])
            else:
                raise ValueError(
                    f"Invalid boundaries:\nb = {b},\nboundaries = {boundaries}")

        # Shape = [num_points_one_bdy, dim_input]
        it_ds_t = torch.stack(it_ds, dim=1)
        ds_boundary_ls_append(it_ds_t)

    # Shape = [<=num_points_boundary, dim_input]
    ds_boundary = torch.cat(ds_boundary_ls, dim=0)
    ds_boundary_label = torch.tensor(
        [[1.] + [torch.nan]*dim_output] * ds_boundary.shape[0]).\
        reshape((-1, 1+dim_output)
                )  # [<=num_points_boundary, 1+dim_output]

    # 3. Concatenate all and get final collocation points and labels
    # Shape = [<= num_points_grid, dim_input] and [<= num_points_grid, 1+dim_output]
    ds = torch.cat([ds_bulk, ds_boundary], dim=0)
    ds_label = torch.cat([ds_bulk_label, ds_boundary_label], dim=0)

    # Restore random state
    np.random.set_state(current_state_numpy)

    return ds, ds_label


def generate_latin_hypercube_data(
        num_points_grid: int,
        sizes: Tuple[Tuple[Union[int, float], Union[int, float]], ...],
        boundaries: Tuple[Tuple[Union[int, float, List, None], ...], ...],
        dim_output: int, dim_input: int, seed: int) -> Tuple[Tensor, Tensor]:
    """
    Generate collocation points with latin hypercube sampling for validation or test sets.
    len(boundaries) = 0 (no boundary) is supported.

    # Args
    - num_points_grid: Number of validation or test datapoints.
    - sizes: See __init__.
    - boundaries: See __init__.
    - dim_output: An int.
    - dim_input: An int.
    - seed: Seed for numpy.

    # Returns
    - ds: Collocation points for validation or test.
            Includes both bulk and boundary collocation points.
            Tensor with shape [num_points_grid, dim_input]
    - ds_label: Labels with shape [num_points_grid, 1 + dim_output].
    """
    # Get random state and fix seed
    current_state_numpy = np.random.get_state()
    np.random.seed(seed)
    lhsampler = qmc.LatinHypercube(d=dim_input, seed=seed)

    # For bulk cpt
    array_l_bounds_bulk = np.array(
        [i[0] for i in sizes])  # [dim_input,]
    array_u_bounds_bulk = np.array(
        [i[1] for i in sizes])  # [dim_input,]

    # For boundary cpt
    ls_l_bounds_bdy = []
    ls_u_bounds_bdy = []
    ls_mult_bdy = []
    ls_add_bdy = []
    for it_bdy_cond in boundaries:  # bdy cond loop
        it_ls_l_bounds_bdy = []
        it_ls_u_bounds_bdy = []
        it_ls_mult_bdy = []
        it_ls_add_bdy = []
        for it_dim, it_bdy_dim in enumerate(it_bdy_cond):  # dim loop
            if it_bdy_dim is None:
                it_ls_l_bounds_bdy.append(sizes[it_dim][0])
                it_ls_u_bounds_bdy.append(sizes[it_dim][1])
                it_ls_mult_bdy.append(1.)
                it_ls_add_bdy.append(0.)
            elif type(it_bdy_dim) in [int, float]:
                it_ls_l_bounds_bdy.append(it_bdy_dim)  # dummy
                it_ls_u_bounds_bdy.append(it_bdy_dim + 1e-3)  # dummy
                it_ls_mult_bdy.append(0.)
                it_ls_add_bdy.append(it_bdy_dim)
            elif type(it_bdy_dim) in [tuple, list]:
                it_ls_l_bounds_bdy.append(it_bdy_dim[0])
                it_ls_u_bounds_bdy.append(it_bdy_dim[1])
                it_ls_mult_bdy.append(1.)
                it_ls_add_bdy.append(0.)
            else:
                raise ValueError(
                    f"Invalid boundaries. Got\nit_bdy = {it_bdy_dim},\nboundaries = {boundaries}")
        ls_l_bounds_bdy.append(it_ls_l_bounds_bdy)
        ls_u_bounds_bdy.append(it_ls_u_bounds_bdy)
        ls_mult_bdy.append([it_ls_mult_bdy])
        ls_add_bdy.append([it_ls_add_bdy])

    array_l_bounds_bdy = np.array(
        ls_l_bounds_bdy)  # [num bdy, dim_input]
    array_u_bounds_bdy = np.array(
        ls_u_bounds_bdy)  # [num bdy, dim_input]
    array_mult_bdy = np.array(ls_mult_bdy)  # [num bdy, 1, dim_input]
    array_add_bdy = np.array(ls_add_bdy)  # [num bdy, 1, dim_input]

    # Num of datapoints for bulk (X) and boundary (X_bc)
    if len(boundaries) != 0:
        num_points_bulk = int(RATIO_BULK_BDY_FOR_VA_TE * num_points_grid)
        num_points_boundary = num_points_grid - num_points_bulk
    else:
        num_points_bulk = num_points_grid
        num_points_boundary = 0

    # 1. Bulk collocation points and labels
    # on CPU, shape=[num_points_bulk, dim_input]
    data_bulk = lhsampler.random(n=num_points_bulk)
    data_bulk = qmc.scale(
        data_bulk,
        array_l_bounds_bulk,
        array_u_bounds_bulk)
    ds_bulk = torch.tensor(data_bulk, dtype=torch.get_default_dtype())
    ds_bulk_label = torch.tensor(
        [[0.] + [torch.nan]*dim_output] * num_points_bulk).\
        reshape((-1, 1+dim_output))  # [num_points_bulk ,1+dim_out]

    # Return bulk dataset if no bdy data is needed
    if num_points_boundary == 0:
        ds = ds_bulk
        ds_label = ds_bulk_label
        return ds, ds_label

    # 2. Boundary collocation points and labels
    num_boundaries = len(boundaries)  # num of (straight or point) boundaries
    num_points_one_bdy = num_points_boundary // num_boundaries
    ds_boundary_ls: List[Tensor] = []
    for it_idx, _ in enumerate(boundaries):  # boundary loop
        it_data_bdy = lhsampler.random(
            n=num_points_one_bdy)  # Shape = [num_points_one_bdy, dim_input]
        it_data_bdy = qmc.scale(
            it_data_bdy,
            array_l_bounds_bdy[it_idx],
            array_u_bounds_bdy[it_idx]) * array_mult_bdy[it_idx] + array_add_bdy[it_idx]
        ds_boundary_ls.append(it_data_bdy)

    ds_boundary = np.concatenate(
        ds_boundary_ls, axis=0)  # Shape = [<=num_points_boundary, dim_input]
    ds_boundary = torch.tensor(ds_boundary, dtype=torch.get_default_dtype())
    ds_boundary_label = torch.tensor(
        [[1.] + [torch.nan]*dim_output] * ds_boundary.shape[0]).\
        reshape((-1, 1+dim_output)
                )  # [<=num_points_boundary, 1+dim_output]

    # 3. Concatenate all and get final collocation points and labels
    # Shape = [<= num_points_grid, dim_input] and [<= num_points_grid, 1+dim_output]
    ds = torch.cat([ds_bulk, ds_boundary], dim=0)
    ds_label = torch.cat([ds_bulk_label, ds_boundary_label], dim=0)

    # Restore random state
    np.random.set_state(current_state_numpy)

    return ds, ds_label


def calc_mins_maxes_from_sizes(sizes: List[List]) -> Tuple[Tensor, Tensor]:
    """
    # Args
    - sizes: A list. Shape = [dim_input, 2].

    # Returns
    - mins: A Tensor with shape [dim_input,].
    - maxes: A Tensor with shape [dim_input,].
    """
    sizes_ts = torch.tensor(sizes)
    maxes = sizes_ts[:, 1]  # [dim_input,]
    mins = sizes_ts[:, 0]  # [dim_input]
    return mins, maxes


class RCPDv3(Dataset):
    """ Random Collocation Points Dataset: RCPDv3.
    Latin hypercube sampling is default in V3.
    Also, mini-batch is made in Dataset, not in DataLoader.

    Random collocation points and boundary collocation points.
    Generates different samples in different epochs.
    One epoch includes num_points datapoints for each of
    random collocation points and boundary collocation points.
    """

    def __init__(
            self,
            sizes: Tuple[Tuple[Union[int, float], Union[int, float]], ...],
            boundaries: Tuple[Tuple[Union[int, float, None, List], Union[int, float, None, List]], ...],
            batch_size_ratio: List,
            dim_output: int,
            batch_size: int,
            transform: Optional[Callable] = None,
            transform_target: Optional[Callable] = None,
            seed: Optional[int] = None,
            num_points: int = 1_000_000_000,
            flag_radial_sampling: bool = False,
            *args, **kwargs) -> None:
        """
        # Args
        - sizes: len(sizes) = dim_input. The sizes of the physical system.
        - boundaries: len(boundaries) = num of boundaries. len(boundaries[0]) = dim_input.
          Specifies boundaries, to which initial and/or boundary conditions are to be applied.
        - batch_size_ratio: len=2 (3 for dataset with X_data).
          This ratio is equal to X:X_bc in a batch.
        - dim_output: Dimension of u(X).
        - batch_size: Batch size. The batch_size in DataLoader is meaningless.
        - num_points_max: Number of training collocation points generated within a single epoch.
        - transform: Optional transform to be applied to a sample.
        - transorm_target: Optional transform to be applied to to Y_data.
        - seed: An int. Random seed.
        - num_points: An int. Defines the number of collocation points in one epoch.
          This is not used for any computation except for __len__.
          However, note that num_points may affect learning rate scheduling.
        - flag_radial_sampling: A bool. Use radial sampling or not. If False, latin hypercube sampling will be used.

        # What is 'sizes'?
        len(sizes) = dim_input = temporal dimension (=1 or 0) + spatial dimensions
        Domain of definition of the first dimension (time or x)) = from sizes[0][0] to sizes[0][1]
        Domain of definition of the second dimension (x or y)    = from sizes[1][0] to sizes[1][1]
        Domain of definition of the second dimension (y or z)    = from sizes[2][0] to sizes[2][1]
        ...

        # What is 'boundaries'?
        len(boundaries) = num of boundaries.
        len(boundaries[0]) = dim_input = temporal dimension (=1) + spatial dimensions.
        - int, float: A boundary condition is assigned at that point.
        - None: A boundary condition is assigned on the whole of this dimension.
        - [int|float, int|float]: A boundary condition is assigned on the interval.
        """
        super().__init__()

        # Assertion
        assert len(batch_size_ratio) == 2
        if len(boundaries) != 0:
            for v in boundaries:
                assert len(sizes) == len(v)

        # Seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # Initialize
        self.sizes = sizes
        self.boundaries = boundaries
        self.transform = transform
        self.transform_target = transform_target
        self.seed = seed
        self.batch_size_ratio = batch_size_ratio
        self.dim_output = dim_output
        self.num_points = num_points
        self.dim_input = len(sizes)
        self.num_boundaries = len(boundaries)
        self.batch_size = batch_size
        self.flag_radial_sampling = flag_radial_sampling

        # Labels
        self.label_bulk = torch.tensor([0.] + [torch.nan] * dim_output)
        self.label_bulk = self.label_bulk.tile([batch_size,]).reshape(
            [batch_size, dim_output+1])  # [batch, dim_output+1]
        self.label_boundary = torch.tensor([1.] + [torch.nan] * dim_output)
        self.label_boundary = self.label_boundary.tile([batch_size,]).reshape(
            [batch_size, dim_output+1])  # [batch, dim_output+1]
        self.label_cat = torch.cat(
            [self.label_bulk[:self.batch_size//2], self.label_boundary[self.batch_size//2:]], dim=0)

        # For bulk cpt
        self.array_l_bounds_bulk = np.array(
            [i[0] for i in sizes])  # [dim_input,]
        self.array_u_bounds_bulk = np.array(
            [i[1] for i in sizes])  # [dim_input,]

        # For boundary cpt
        ls_l_bounds_bdy = []
        ls_u_bounds_bdy = []
        ls_mult_bdy = []
        ls_add_bdy = []
        for it_bdy_cond in boundaries:  # bdy cond loop
            it_ls_l_bounds_bdy = []
            it_ls_u_bounds_bdy = []
            it_ls_mult_bdy = []
            it_ls_add_bdy = []
            for it_dim, it_bdy_dim in enumerate(it_bdy_cond):  # dim loop
                if it_bdy_dim is None:
                    it_ls_l_bounds_bdy.append(sizes[it_dim][0])
                    it_ls_u_bounds_bdy.append(sizes[it_dim][1])
                    it_ls_mult_bdy.append(1.)
                    it_ls_add_bdy.append(0.)
                elif type(it_bdy_dim) in [int, float]:
                    it_ls_l_bounds_bdy.append(it_bdy_dim)  # dummy
                    it_ls_u_bounds_bdy.append(it_bdy_dim + 1e-3)  # dummy
                    it_ls_mult_bdy.append(0.)
                    it_ls_add_bdy.append(it_bdy_dim)
                elif type(it_bdy_dim) in [tuple, list]:
                    it_ls_l_bounds_bdy.append(it_bdy_dim[0])
                    it_ls_u_bounds_bdy.append(it_bdy_dim[1])
                    it_ls_mult_bdy.append(1.)
                    it_ls_add_bdy.append(0.)
                else:
                    raise ValueError(
                        f"Invalid boundaries. Got\nit_bdy = {it_bdy_dim},\nboundaries = {boundaries}")
            ls_l_bounds_bdy.append(it_ls_l_bounds_bdy)
            ls_u_bounds_bdy.append(it_ls_u_bounds_bdy)
            ls_mult_bdy.append([it_ls_mult_bdy])
            ls_add_bdy.append([it_ls_add_bdy])

        self.array_l_bounds_bdy = np.array(
            ls_l_bounds_bdy)  # [num bdy, dim_input]
        self.array_u_bounds_bdy = np.array(
            ls_u_bounds_bdy)  # [num bdy, dim_input]
        self.array_mult_bdy = np.array(ls_mult_bdy)  # [num bdy, 1, dim_input]
        self.array_add_bdy = np.array(ls_add_bdy)  # [num bdy, 1, dim_input]

        # Define self.ratio (bulk:boundary datapoints ratio)
        if len(boundaries) != 0:
            self.ratio = batch_size_ratio[0] / sum(batch_size_ratio)
        else:
            self.ratio = 1.

        # Define default preprocessings
        self.transform_default = torchvision.transforms.Lambda(
            lambda x: x
            # torchvision.transforms.Lambda(
            #     lambda x: torch.tensor(x))
        )
        self.transform_target_default = torchvision.transforms.Lambda(
            lambda x: x  # torch.tensor(x)
        )

        # Latin hypercube sampling
        self.tensorize = torch.tensor
        self.default_dtype = torch.get_default_dtype()
        self.tmp_flag = True

    def __len__(self) -> int:
        """ Returns the sample size """
        return self.num_points

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        # Output format
        The output is [Tensor(point), Tensor(label)].

        'point' has shape=[dim_input,]. X, X_bc, or X_data.
        X is a collocation point in the bulk (non-boundary),
        X_bc is a collocation point on a boundary,
        X_data is a observed data,
        which is not given in RandomCPDataset (no observation data,
        i.e., forward problem for solving PDEs).

        'label' has shape=[1 + dim_output,]. The first dim is equal to 0.0 for X, 1.0 for X_bc,
        or 2.0 for X_data, and
        the second,... dim is
        float("nan"),... for X and X_bc
        or
        float,... for X_data,
        respectively.

        # Args
        - idx: Not used because training collocation points are
          generated every time when __getitem__ is called.

        # Returns
        - example: A pair of point and label.
        """
        # To avoid "My data loader workers return identical random numbers", create qmc.LatinHypercube in __getitem__.
        # Ref: https://pytorch.org/docs/master/notes/faq.html#my-data-loader-workers-return-identical-random-numbers
        if self.tmp_flag:
            self.lhsampler = qmc.LatinHypercube(d=self.dim_input)
            self.tmp_flag = False

        # Bulk or boundary batch
        # dice = np.random.rand()
        batch_points = self.lhsampler.random(
            n=self.batch_size)  # time consuming

        # Get a bulk collocation point
        # if dice < self.ratio:
        batch_points_bulk = qmc.scale(
            batch_points[:self.batch_size//2],
            self.array_l_bounds_bulk,
            self.array_u_bounds_bulk)  # [batch//2, dim_input]
        # batch_points = self.tensorize(
        #     batch_points, dtype=self.default_dtype) # [batch//2, 1+dim_output,]
        # label_bulk = self.label_bulk[:self.batch_size//2]

        # Get a boundary collocation point
        # else:
        dice2 = np.random.randint(0, self.num_boundaries)
        batch_points_bdy = qmc.scale(
            batch_points[self.batch_size//2:],
            self.array_l_bounds_bdy[dice2],
            self.array_u_bounds_bdy[dice2]) * self.array_mult_bdy[dice2] + self.array_add_bdy[dice2]  # [batch//2, dim_input]

        # Concat
        batch_points_cat = np.concatenate(
            [batch_points_bulk, batch_points_bdy], axis=0)  # [batch, dim_input]
        batch_points_cat = self.tensorize(
            batch_points_cat, dtype=self.default_dtype)  # time consuming
        # label_bdy = self.label_boundary[self.batch_size//2:]# [batch//2, 1+dim_output,]

        # Preprocessing
        return self._preproc(batch_points_cat, self.label_cat)

    def _preproc(self, point, label):
        point = self.transform_default(point)
        if self.transform is not None:
            point = self.transform(point)
        label = self.transform_target_default(label)
        if self.transform_target is not None:
            label = self.transform_target(label)
        return point, label


def get_RCPD_for_va_te(
        sizes: Tuple[Tuple[Union[int, float], Union[int, float]], ...],
        boundaries: Tuple[Tuple[Union[int, float, None, List], Union[int, float, None, List]], ...],
        dim_output: int,
        transform: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        flag_radial_sampling: bool = False) -> Tuple[Dataset, Dataset]:
    """
    # Args
    - flag_radial_sampling: A bool. Radial sampling or not.
    """
    if len(boundaries) != 0:
        for v in boundaries:
            assert len(sizes) == len(v)
    num_points_va_max = NUM_VAL_MAX
    num_points_te_max = NUM_TE_MAX
    # lengths = [j - i for i, j in sizes]
    # lows = [i for i, _ in sizes]
    # lengths_t = torch.tensor(
    #     lengths, requires_grad=False)  # [dim_input,]
    # lows_t = torch.tensor(lows, requires_grad=False)  # [dim_input,]
    dim_input = len(sizes)

    # Get validation and test sets
    if flag_radial_sampling:
        raise NotImplementedError
        ds_va_t, ds_va_label_t = generate_radial_sample_data(
            num_points_va_max, sizes, boundaries, dim_output, dim_input, seed=777)
        ds_te_t, ds_te_label_t = generate_radial_sample_data(
            num_points_te_max, sizes, boundaries, dim_output, dim_input, seed=7)
    else:
        ds_va_t, ds_va_label_t = generate_latin_hypercube_data(
            num_points_va_max, sizes, boundaries, dim_output, dim_input, seed=777)
        ds_te_t, ds_te_label_t = generate_latin_hypercube_data(
            num_points_te_max, sizes, boundaries, dim_output, dim_input, seed=7)

    # Define default preprocessings
    transform_default = torchvision.transforms.Lambda(
        lambda x: x
    )
    transform_target_default = torchvision.transforms.Lambda(
        lambda x: x
    )

    # Preprocessing
    ds_va_t = transform_default(ds_va_t)
    if transform is not None:
        ds_va_t = transform(ds_va_t)
    ds_va_label_t = transform_target_default(ds_va_label_t)
    if transform_target is not None:
        ds_va_label_t = transform_target(ds_va_label_t)

    ds_te_t = transform_default(ds_te_t)
    if transform is not None:
        ds_te_t = transform(ds_te_t)
    ds_te_label_t = transform_target_default(ds_te_label_t)
    if transform_target is not None:
        ds_te_label_t = transform_target(ds_te_label_t)

    # Make TensorDataset
    ds_va = TensorDataset(ds_va_t, ds_va_label_t)
    ds_te = TensorDataset(ds_te_t, ds_te_label_t)

    return ds_va, ds_te
