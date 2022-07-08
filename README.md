# Realistic Hair Style Try-On: Face and Hair Image Mapping Using Semantic Maps for SDEdit (ECTI-CARD 2022)
[Sorayut Meeyim](https://github.com/sorayutmild), [Phalapat Tektrakul](https://github.com/phalapat), [Pakkaphong Akkabut](https://github.com/sanviiz), Werapon Chiracharit \
[PROCEEDING ECTI-CARD 2022](https://ecticard2022.ecticard.org/program/PROCEEDING%20ECTI%20CARD2022.pdf), [Paper](https://github.com/sanviiz/hairstyle-try-on/blob/dev-mild/misc/Realistic-Hairstyle-try-on-paper.pdf), [Slide](https://github.com/sanviiz/hairstyle-try-on/blob/dev-mild/misc/Realistic-Hairstyle-try-on-presentation.pdf)

<p align="center">
  <img src="https://github.com/sanviiz/hairstyle-try-on/blob/dev-mild/misc/all_results.png?raw=true" alt="all results"/>
</p>

## Abstract
Machine learning-based image generation can create new person face images with new hair colors or hairstyles. This paper presents synthesis and editing method to modify hairstyles in the images by semantic maps. The face and hair images are mapped and inpainted using fast marching method and Stochastic Differential Editing (SDEdit). The experimental results shows that the proposed method controls both hairstyles and color effectively with single target hairstyle image. Moreover, the method is able to generate hairstyles in case of occluded face images.\
**Keywords:** Realistic hair style try-on, Semantic maps, SDEdit

<p align="center">
  <img src="https://github.com/sanviiz/hairstyle-try-on/blob/dev-mild/misc/overview.jpg?raw=true" alt="overview"/>
</p>

<p align="center">
  <img src="https://github.com/sanviiz/hairstyle-try-on/blob/dev-mild/misc/interpolate.jpg?raw=true" alt="interpolate"/>
</p>
## Requirements
- Python 3.8.5 is used. Basic requirements are listed in the 'requirements.txt'.
```sh
pip install -r requirements.txt
```
- Download face segmentation model from [this link]() and put it in ```image_segmentation/```
- Create ```checkpoints``` folder and Download  ```checkpoints/celeba_hq.ckpt ``` from [this link](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt) than put it in ```checkpoints```

## Demo
You can run
```sh
python inference.py --seg_model_path <image segmentation model> --t <Noise level> --target_image_path <target image path> --source_image_path <source image path>
```
example:
```sh
python inference.py --seg_model_path image_segmentation/face_segment_checkpoints_256.pth.tar --t 500 --target_image_path images/92.jpg --source_image_path images/82.jpg
```
The results will shown in ```exp/image_samples``` folder

