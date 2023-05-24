# TLNets



TLNets: Transformation Learning Networks for long-range time-series prediction

we propose a novel plan for the designing of networks' architecture based on transformations, possessing the potential to achieve an enhanced receptive field in learning which brings benefits to fuse features across scales. In this context, we introduce four different transformation mechanisms as bases to construct the learning model including Fourier Transform (FT), Singular Value Decomposition (SVD), matrix multiplication and Conv block. Hence, we develop four learning models based on the above building blocks, namely, FT-Matrix, FT-SVD, FT-Conv, and Conv-SVD. Note that the FT and SVD blocks are capable of learning global information, while the Conv blocks focus on learning local information. The matrix block is sparsely designed to learn both global and local information simultaneously. The above Transformation Learning Networks (TLNets) have been extensively tested and compared with multiple baseline models based on several real-world datasets and showed clear potential in long-range time-series forecasting.


## Get Started

1. Install Python 3.7, PyTorch 1.8.1.
2. Download data. You can obtain all data from https://github.com/thuml/Autoformer.

