from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import numpy as np
import requests, os, sys
from tqdm import tqdm
import pickle
import torch

# (480, 640, 3)

dataset_root_path = "/home/seungyoun/dataset/CLEVR_v1.0"
feat_save_path    = "/home/seungyoun/spatial-reasoning/CLEVR_feat"

phases = ['train', 'val', 'test']

def get_image_model():
    #feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb8')
    #model = ViTModel.from_pretrained('facebook/dino-vitb8').cuda()
    import clip
    model, feature_extractor = clip.load("ViT-B/32", device='cuda', jit=False)

    return feature_extractor, model

if __name__=="__main__":
    feature_extractor, model = get_image_model()
    for phase in phases:
        print(f"{phase} start !")

        phase_dir_path = os.path.join(dataset_root_path, "images", phase)
        phase_iamges_path = [os.path.join(phase_dir_path, i) for i in os.listdir(phase_dir_path)]

        for img_full_path in tqdm(phase_iamges_path):
            img_name = img_full_path.split("/")[-1][:-4]
            image = Image.open(img_full_path).convert('RGB')

            #inputs = feature_extractor(images=image, return_tensors="pt")
            #inputs['pixel_values'] = inputs['pixel_values'].cuda()
            image = feature_extractor(image).unsqueeze(0).cuda()
            with torch.no_grad():
                #outputs = model(**inputs)      # 224,224
                outputs = model.encode_image(image).cpu() # [1,512]
            #last_hidden_states = outputs.last_hidden_state  # 1, 785(28*28 + 1), 768
            
            save_full_dir  = os.path.join(feat_save_path, phase, img_name)
            save_full_dir += '_clip.pkl'

            #torch.save(last_hidden_states.cpu(), save_full_dir)
            torch.save(outputs, save_full_dir)