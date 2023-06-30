# Optimal-Transport gudied Retinal Image Enhancement 

[IPMI'2023] [OTRE: Where Optimal Transport Guided Unpaired Image-to-Image Translation Meets Regularization by Enhancing](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=7HLmlHMAAAAJ&citation_for_view=7HLmlHMAAAAJ:IjCSPb-OGe4C)

<img src="images/network-final.png"/>

**Abstract** 
Non-mydriatic retinal color fundus photography (CFP) is widely available due to the advantage of not requiring pupillary dilation, however, is prone to poor quality due to operators, systemic imperfections, or patient-related causes. Optimal retinal image quality is mandated for accurate medical diagnoses and automated analyses. Herein, we leveraged the \emph{Optimal Transport (OT)} theory to propose an unpaired image-to-image translation scheme for mapping low-quality retinal CFPs to high-quality counterparts. Furthermore, to improve the flexibility, robustness, and applicability of our image enhancement pipeline in the clinical practice, we generalized a state-of-the-art model-based image reconstruction method, regularization by denoising, by plugging in priors learned by our OT-guided image-to-image translation network. We named it \emph{regularization by enhancing (RE)}. We validated the integrated framework, OTRE, on three publicly available retinal image datasets by assessing the quality after enhancement and their performance on various downstream tasks, including diabetic retinopathy grading, vessel segmentation, and diabetic lesion segmentation. The experimental results demonstrated the superiority of our proposed framework over some state-of-the-art unsupervised competitors and a state-of-the-art supervised method. 

### This repository will be maintained and updated ! Stay Tuned !
We will appreciate any suggestions and comments. If you find this code being helpful, please cite our papers. Thanks ! 
```
@inproceedings{zhu2023otre,
  title={OTRE: Where Optimal Transport Guided Unpaired Image-to-Image Translation Meets Regularization by Enhancing},
  author={Zhu, Wenhui and Qiu, Peijie and Dumitrascu, Oana M and Sobczak, Jacob M and Farazi, Mohammad and Yang, Zhangsihao and Nandakumar, Keshav and Wang, Yalin},
  booktitle={International Conference on Information Processing in Medical Imaging},
  pages={415--427},
  year={2023},
  organization={Springer}
}
```
```
@article{zhu2023optimal,
  title={Optimal Transport Guided Unsupervised Learning for Enhancing low-quality Retinal Images},
  author={Zhu, Wenhui and Qiu, Peijie and Farazi, Mohammad and Nandakumar, Keshav and Dumitrascu, Oana M and Wang, Yalin},
  journal={arXiv preprint arXiv:2302.02991},
  year={2023}
}
```

## 1. Dependencies
## 2. Data Structures 

### License
Released under the [ASU GitHub Project License](https://github.com/Retinotopy-mapping-Research/DRRM/blob/master/LICENSE.txt).
