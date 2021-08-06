#  Is Depth Really Necessary for Salient Object Detection?

The PyTorch implementation of the [DASNet](https://dl.acm.org/doi/10.1145/3394171.3413855).

Please see details in http://cvteam.net/projects/2020/DASNet/


## Prerequisites
- [Python 3.6](https://www.python.org/)
- [Pytorch 1.0+](http://pytorch.org/)
- [OpenCV 4.0](https://opencv.org/)
- [Numpy](https://numpy.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [Apex](https://github.com/NVIDIA/apex)


## DIR
- res: resnet pre-trained models
- eval: test results
- data: datasets

## Train
```shell script
cd src
python train.py
```

## Test
```shell script
cd src
python test.py
```

## Evaluation
```shell
cd eval
matlab
main
```

## Citation
- If you find this work is helpful, please cite our paper
```
@inproceedings{zhao2020DASNet,
  title={Is depth really necessary for salient object detection?},
  author={Zhao, Jiawei and Zhao, Yifan and Li, Jia and Chen, Xiaowu},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={1745--1754},
  year={2020}
}
```
## Reference
This project is based on the following implementations:
- [https://github.com/weijun88/F3Net](https://github.com/weijun88/F3Net)