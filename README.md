# Harmonised segmentation of neonatal brain MRI

#### Author: Irina Grigorescu   |   irina[_dot_]grigorescu[_at_]kcl[_dot_]ac[_dot_]uk

This is the companion code for the following publications:

- Grigorescu, I. et al. (2021). _Harmonized Segmentation of Neonatal Brain MRI_. Frontiers in Neuroscience
	- [doi.org/10.3389/fnins.2021.662005](https://doi.org/10.3389/fnins.2021.662005)

#### Example train/validation/test file
```
t2w,lab,ga,as,gender
subj1_T2w_brain.nii.gz,subj1_lab_tissue.nii.gz,41.0,41.14,Male
```

Also, bare in mind that the code currently expects the suffix ```_T2w_brain.nii.gz``` for the images, and the suffix ```_lab_tissue.nii.gz``` for the segmentation maps.
It also expects ```_lab_all.nii.gz``` for the cortical parcellation network.

