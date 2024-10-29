
## How to Learn More? Exploring the Possibility of Kolmogorov-Arnold Networks for Hyperspectral Image Classification







[Ali Jamali](https://www.researchgate.net/profile/Ali-Jamali), [Swalpa Kumar Roy](https://swalpa.github.io), [Danfeng Hong](https://sites.google.com/view/danfeng-hong), [Bing Lu](https://www.sfu.ca/people/binglu/about.html), and [Pedram Ghamisi](https://www.iarai.ac.at/people/pedramghamisi/)

<img src="HybridKAN.png"/>
<img src="Kan.png"/>
<img src="Kan_operation.png"/>

___________

This PyTorch code is for the paper:
Jamali, A.; Roy, S.K.; Hong, D.; Lu, B.; Ghamisi, P. "[How to Learn More? Exploring Kolmogorov-Arnold Networks for Hyperspectral Image Classification],". Remote Sens. 2024, 16, 4015. https://doi.org/10.3390/rs16214015.



Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

@Article{rs16214015,

AUTHOR = {Jamali, Ali and Roy, Swalpa Kumar and Hong, Danfeng and Lu, Bing and Ghamisi, Pedram},

TITLE = {How to Learn More? Exploring Kolmogorov–Arnold Networks for Hyperspectral Image Classification},

JOURNAL = {Remote Sensing},

VOLUME = {16},

YEAR = {2024},

NUMBER = {21},

ARTICLE-NUMBER = {4015},

URL = {https://www.mdpi.com/2072-4292/16/21/4015},

ISSN = {2072-4292},

ABSTRACT = {Convolutional neural networks (CNNs) and vision transformers (ViTs) have shown excellent capability in complex hyperspectral image (HSI) classification. However, these models require a significant number of training data and are computational resources. On the other hand, modern Multi-Layer Perceptrons (MLPs) have demonstrated a great classification capability. These modern MLP-based models require significantly less training data compared with CNNs and ViTs, achieving state-of-the-art classification accuracy. Recently, Kolmogorov–Arnold networks (KANs) were proposed as viable alternatives for MLPs. Because of their internal similarity to splines and their external similarity to MLPs, KANs are able to optimize learned features with remarkable accuracy, in addition to being able to learn new features. Thus, in this study, we assessed the effectiveness of KANs for complex HSI data classification. Moreover, to enhance the HSI classification accuracy obtained by the KANs, we developed and proposed a hybrid architecture utilizing 1D, 2D, and 3D KANs. To demonstrate the effectiveness of the proposed KAN architecture, we conducted extensive experiments on three newly created HSI benchmark datasets: QUH-Pingan, QUH-Tangdaowan, and QUH-Qingyun. The results underscored the competitive or better capability of the developed hybrid KAN-based model across these benchmark datasets over several other CNN- and ViT-based algorithms, including 1D-CNN, 2DCNN, 3D CNN, VGG-16, ResNet-50, EfficientNet, RNN, and ViT.},

DOI = {10.3390/rs16214015}
}

  
Acknowledgement
---------------------

The Efficient KAN is implementated from [EfficientKAN](https://github.com/Blealtan/efficient-kan).
The 3D KAN is implementated from [3DKAN](https://github.com/FirasBDarwish/ConvKAN3D).

## License

Copyright (c) 2024 Ali Jamali. Released under the MIT License. See [LICENSE](LICENSE) for details.

