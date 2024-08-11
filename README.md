# MGSFformer
This github repository corresponds to our paper accepted by Information fusion (MGSFformer: A Multi-Granularity Spatiotemporal Fusion Transformer for air quality prediction).


The following is the meaning of the core hyperparameter:
- Input_len: Historical length
- out_len: Future length
- num_id: Number of time series 
- IE_dim: Embedding size
- emb_size: Embedding size
- dropout: Droupout
- num_head: Number of multi-head attention


If the code is helpful to you, please cite the following paper:
```bibtex
@article{yu2024mgsfformer,
  title={MGSFformer: A Multi-Granularity Spatiotemporal Fusion Transformer for Air Quality Prediction},
  author={Yu, Chengqing and Wang, Fei and Wang, Yilun and Shao, Zezhi and Sun, Tao and Yao, Di and Xu, Yongjun},
  journal={Information Fusion},
  pages={102607},
  year={2024},
  publisher={Elsevier}
}
```
