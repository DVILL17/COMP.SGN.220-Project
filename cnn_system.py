#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from Exercise 2

from typing import Union, Tuple

import torch.nn
from torch import Tensor
from torch.nn import Module, Conv2d, MaxPool2d, \
    BatchNorm2d, ReLU, Linear, Sequential, Dropout2d

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['MyCNNSystem']


class MyCNNSystem(Module):

    def __init__(self,
                 cnn_channels_out_1: int,
                 cnn_kernel_1: Union[Tuple[int, int], int],
                 cnn_stride_1: Union[Tuple[int, int], int],
                 cnn_padding_1: Union[Tuple[int, int], int],
                 pooling_kernel_1: Union[Tuple[int, int], int],
                 pooling_stride_1: Union[Tuple[int, int], int],
                 cnn_channels_out_2: int,
                 cnn_kernel_2: Union[Tuple[int, int], int],
                 cnn_stride_2: Union[Tuple[int, int], int],
                 cnn_padding_2: Union[Tuple[int, int], int],
                 pooling_kernel_2: Union[Tuple[int, int], int],
                 pooling_stride_2: Union[Tuple[int, int], int],
                 classifier_input_features: int,
                 output_classes: int) -> None:
        """MyCNNSystem, using two CNN layers, followed by a ReLU, a batch norm,\
        and a max-pooling process.

        :param cnn_channels_out_1: Output channels of first CNN.
        :type cnn_channels_out_1: int
        :param cnn_kernel_1: Kernel shape of first CNN.
        :type cnn_kernel_1: int|Tuple[int, int]
        :param cnn_stride_1: Strides of first CNN.
        :type cnn_stride_1: int|Tuple[int, int]
        :param cnn_padding_1: Padding of first CNN.
        :type cnn_padding_1: int|Tuple[int, int]
        :param pooling_kernel_1: Kernel shape of first pooling.
        :type pooling_kernel_1: int|Tuple[int, int]
        :param pooling_stride_1: Strides of first pooling.
        :type pooling_stride_1: int|Tuple[int, int]
        :param cnn_channels_out_2: Output channels of second CNN.
        :type cnn_channels_out_2: int
        :param cnn_kernel_2: Kernel shape of second CNN.
        :type cnn_kernel_2: int|Tuple[int, int]
        :param cnn_stride_2: Strides of second CNN.
        :type cnn_stride_2: int|Tuple[int, int]
        :param cnn_padding_2: Padding of second CNN.
        :type cnn_padding_2: int|Tuple[int, int]
        :param pooling_kernel_2: Kernel shape of second pooling.
        :type pooling_kernel_2: int|Tuple[int, int]
        :param pooling_stride_2: Strides of second pooling.
        :type pooling_stride_2: int|Tuple[int, int]
        :param classifier_input_features: Input features to the\
                                          classifier.
        :type classifier_input_features: int
        :param output_classes: Output classes.
        :type output_classes: int
        """
        super().__init__()

        self.block_1 = torch.nn.Sequential(Conv2d(1, cnn_channels_out_1, cnn_kernel_1,
                                                  cnn_stride_1, cnn_padding_1),
                                           ReLU(),
                                           BatchNorm2d(cnn_channels_out_1),
                                           MaxPool2d(pooling_kernel_1, pooling_stride_1))
        self.block_2 = torch.nn.Sequential(Conv2d(cnn_channels_out_1, cnn_channels_out_2,
                                                  cnn_kernel_2, cnn_stride_2, cnn_padding_2),
                                           ReLU(),
                                           BatchNorm2d(cnn_channels_out_2),
                                           MaxPool2d(pooling_kernel_2, pooling_stride_2))
        T = 431
        H_out_cnn_1 = (classifier_input_features + 2 * cnn_padding_1[0] - cnn_kernel_1[0]) // cnn_stride_1[0] + 1
        W_out_cnn_1 = (T + 2 * cnn_padding_1[1] - cnn_kernel_1[1]) // cnn_stride_1[1] + 1
        H_out_pool_1 = (H_out_cnn_1 - pooling_kernel_1[0]) // pooling_stride_1[0] + 1
        W_out_pool_1 = (W_out_cnn_1 - pooling_kernel_1[1]) // pooling_stride_1[1] + 1
        H_out_cnn_2 = (H_out_pool_1 + 2 * cnn_padding_2[0] - cnn_kernel_2[0]) // cnn_stride_2[0] + 1
        W_out_cnn_2 = (W_out_pool_1 + 2 * cnn_padding_2[1] - cnn_kernel_2[1]) // cnn_stride_2[1] + 1
        H_out_pool_2 = (H_out_cnn_2 - pooling_kernel_2[0]) // pooling_stride_2[0] + 1
        W_out_pool_2 = (W_out_cnn_2 - pooling_kernel_2[1]) // pooling_stride_2[1] + 1
        # We'll perform a average pooling along the time axis, so W_out_pool_2 dissapears
        num_el_out = int(H_out_pool_2 * cnn_channels_out_2)

        self.classifier = Linear(num_el_out, 55168)


    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass.

        :param x: Input features\
                  (shape either `batch x time x features` or\
                  `batch x channels x time x features`).
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        # Ensure input is 4D: [batch_size, channels, height, width]
        if x.ndimension() == 3:
            x = x.unsqueeze(1)  # Add a channel dimension
            
        h = x if x.ndimension() == 4 else x.unsqueeze(1)

        # apply block_1 to h
        h = self.block_1(h)

        # apply block_2 to h
        h = self.block_2(h)

        # time average pooling and reshaping
        h = h.mean(-1)
        h = h.view(x.size()[0], -1)

        return self.classifier(h).squeeze(-1)


def main():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 8
    d_time = 646
    d_feature = 40
    x = torch.rand(batch_size, d_time, d_feature)
    y = torch.rand(batch_size, 55168)
    # Instantiate our CNN
    # ..................

    # Define the CNN model and give it the model hyperparameters
    cnn_model = MyCNNSystem(
        cnn_channels_out_1=32,
        cnn_kernel_1=(3, 3),
        cnn_stride_1=(1, 1),
        cnn_padding_1=(0, 0),
        pooling_kernel_1=(3, 3),
        pooling_stride_1=(3, 3),
        cnn_channels_out_2=64,
        cnn_kernel_2=(3, 3),
        cnn_stride_2=(1, 1),
        cnn_padding_2=(0, 0),
        pooling_kernel_2=(3, 3),
        pooling_stride_2=(3, 3),
        classifier_input_features=d_feature,
        output_classes=1)

    # Pass DNN to the available device.
    cnn_model = cnn_model.to(device)

    # Give them to the appropriate device.
    x = x.to(device)
    y = y.to(device)

    # Get the predictions .
    y_hat = cnn_model(x)
    print(y_hat)


if __name__ == '__main__':
    main()