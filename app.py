import streamlit as st
import pandas as pd
import numpy as np
import yaml
import cv2
import os
import logging
import traceback
from image_segmentation.segment_inference import face_segment
from runners.image_editing import Diffusion
from image_landmark_transform.face_landmark import face_landmark_transform
from image_artifact_fill.artifact_fill import face_artifact_fill
from inference import resize_image, dict2namespace

class app:
    def __init__(self):
        self.args = self.create_args()
        self.init_app()
        self.config = self.create_config()
        run = st.button('RUN')
        if self.args['target_image'] and self.args['source_image'] and run: # run pipeline when input images
            self.pipeline()
        
        
    def init_app(self, ):
        st.title('Realistic Hairstyle try on')
        st.subheader('Input images')
        self.args['target_image'] = st.file_uploader('Target image (The person whose FACE you desire)', 
                                                type=['png', 'jpg', 'jpeg'])
        self.args['source_image'] = st.file_uploader('Source image (The person whose HAIR you desire)', 
                                                type=['png', 'jpg', 'jpeg'])
        # read original image
        if self.args['target_image'] and self.args['source_image']:
            self.target_image = self.read_image_from_streamlit(self.args['target_image'])
            self.source_image = self.read_image_from_streamlit(self.args['source_image'])
            
            images= [self.target_image, self.source_image]
            indices_on_page = ['Target image', 'Source image']
            st.image(images, width=300, caption=indices_on_page)

            # st.image(self.target_image, channels="RGB", caption='Target image')
            # st.image(self.source_image, channels="RGB", caption='Source image')
            
        st.sidebar.header('Input some parameters (or using the default is fine)')
        
        st.sidebar.subheader('SDEdit parameters')
        
        self.args['seed'] =  st.sidebar.number_input('Input random seed', min_value=0, 
                                        value=1234, step=1, format ='%d',
                                        )
        self.args['sample_step'] = st.sidebar.number_input('Total sampling steps (Number of generated images)', min_value=0, max_value=5, 
                                        value=1, step=1, format ='%d',
                                        )
        self.args['t'] = st.sidebar.number_input('Sampling noise scale (Too much is slower, but too little results in an unsatisfying.)', min_value=0, max_value=2000,
                                        value=500, step=1, format ='%d',
                                        )
        
        # self.args['is_erode_mask'] = int(st.sidebar.checkbox('erode mask')) # erode mask before pass to SDEdit (1) or not (0)
        self.args['erode_kernel_size'] = st.sidebar.number_input('erode kernel size', min_value=0, max_value=10,
                                        value=7, step=1, format ='%d',
                                        )
        

    def create_args(self):
        args = dict()
        # Image segmentation
        args['seg_model_path'] = os.path.join("image_segmentation", "face_segment_checkpoints_256.pth.tar")
        args['image_size'] = (256,256) # output image size (height, width)
        args['input_image_size'] = (256,256) # input image size before segment (height, width)
        args['label_config'] = os.path.join("image_segmentation", "label.yml") # Path to the label.yml

        # SDEdit
        args['exp'] = 'exp' # Path for saving running related data.
        args['verbose'] = 'info' # 'Verbose level: info | debug | warning | critical'
        args['sample'] = True # Whether to produce samples from the model
        args['image_folder'] = 'images' # The folder name of samples
        args['ni'] = True # No interaction. Suitable for Slurm Job launcher
        args['is_erode_mask'] = True
        return args
        
    @st.cache
    def create_config(self, config_file_path=os.path.join("configs", "celeba.yml")):
        # parse config file
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        new_config = dict2namespace(config)
        return new_config
    
    def read_image_from_streamlit(self, uploaded_file):
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        return opencv_image
    
    def pipeline(self, ):
        segment = face_segment(seg_model_path=self.args['seg_model_path'], 
                            label_config=self.args['label_config'], 
                            input_image_size=self.args['input_image_size'])
        

        # infer image segmentation
        target_mask = segment.segmenting(image=self.target_image)
        source_mask = segment.segmenting(image=self.source_image)

        # resize image and mask
        target_image = resize_image(self.target_image, self.args['image_size'])
        source_image = resize_image(self.source_image, self.args['image_size'])
        target_mask = resize_image(target_mask, self.args['image_size'])
        source_mask = resize_image(source_mask, self.args['image_size'])

        # detect face landmark and transform image
        transform_outputs = face_landmark_transform(target_image, target_mask, source_image, source_mask)
        transformed_image, transformed_mask = transform_outputs["result_image"], transform_outputs["result_mask"]
        transformed_segment = segment.segmenting(image=transformed_image)
        # cv2.imwrite('report_images/transformed_mask.png', cv2.cvtColor(transformed_mask, cv2.COLOR_RGB2BGR))

        # fill artifacts 
        filled_image = face_artifact_fill(target_image, target_mask, transformed_image, transformed_mask, transformed_segment)

        # SDEdit
        sde_mask = transform_outputs['only_fixed_face']
        if self.args['is_erode_mask']:
            kernel_size = self.args['erode_kernel_size']
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            sde_mask = cv2.erode(sde_mask, kernel,iterations = 1)
        

        print(">" * 80)
        logging.info("Exp instance id = {}".format(os.getpid()))
        logging.info("Config =")
        print("<" * 80)

        try:
            runner = Diffusion(image_folder=self.args['image_folder'], 
                            sample_step=self.args['sample_step'], 
                            total_noise_levels=self.args['t'], 
                            config=self.config)
            self.show_images = runner.image_editing_sample_for_streamlit(filled_image, sde_mask)
            images = list(self.show_images.values())
            captions = list(self.show_images.keys())
            st.image(images, width=100, caption=captions, clamp=True)
            for it in range(self.args['sample_step']):
                st.image(self.show_images[f'samples_{it}'], width=300, caption=f'Final image {it+1}', clamp=True)
            
        except Exception:
                logging.error(traceback.format_exc())
        
        return 0

if __name__ == '__main__':
    a = app()
    
