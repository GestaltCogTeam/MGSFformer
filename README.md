# MGSFformer
This github repository corresponds to our paper accepted by Information fusion (MGSFformer: A Multi-Granularity Spatiotemporal Fusion Transformer for air quality prediction).

In order for MGSFformer to adapt to classic multivariate time series forecasting tasks, we slightly modified the model's input (In the initial version, the inputs of MGSFformer was data with multiple different granularities. In the modified version, coarse-grained data is obtained by segmenting and averaging fine-grained data.). In this case, MGSFformer achieves satisfactory experimental results on datasets such as ETT.

If you want to evaluate the performance of the model on other public datasets, please use the following links: https://github.com/ChengqingYu/BasicTS

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
  volume = {113},
  pages = {102607},
  year = {2025},
  issn = {1566-2535},
  doi = {https://doi.org/10.1016/j.inffus.2024.102607},
  publisher={Elsevier}
}
```
