import os
import pickle,json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import GPT2Tokenizer
from typing import Tuple, Optional, Union
import sys, random
import json

class ClipCLEVRDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item] # torch.tensor([23, 3, 35, 981, ...])
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        self.question_id[item]
        prefix = torch.load(os.path.join(self.clevr_feat_path, self.phase, self.question_id[item] + "_clip.pkl")).squeeze()
        if self.pooler:
            prefix = prefix[-1,:]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str , prefix_length : int ,gpt2_type: str = "gpt2",
                 normalize_prefix=False, phase = 'train', pooler=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.data_path = data_path
        self.phase = phase
        self.pooler = pooler
        self.clevr_feat_path = "./CLEVR_feat"
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open("/home/seungyoun/dataset/CLEVR_v1.0/questions/CLEVR_train_questions.json", "r") as js:
            self.json_file = json.load(js)
        print(f"Json file loaded.")

        self.max_seq_len, self.tokenized = 0, list()
        self.question_id = list()
        self.captions_tokens = list()

        for qdict in self.json_file['questions']:
            qid = qdict['image_filename'][:-4]
            #exp = qdict['factual_explanation']
            full_caption = f"Question : {qdict['question']} \nAnswer : {qdict['answer']}."
            cap_tok = torch.tensor(self.tokenizer.encode(full_caption),dtype=torch.int64)
            
            self.max_seq_len = max(cap_tok.shape[0], self.max_seq_len)
            self.question_id.append(qid) 
            self.captions_tokens.append(cap_tok)

        print(f"CLEVR init end.")

class ClipCLEVRXDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item] # torch.tensor([23, 3, 35, 981, ...])
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        self.question_id[item]
        prefix = torch.load(os.path.join(self.clevr_feat_path, self.phase, self.question_id[item] + "_clip.pkl")).squeeze()
        if self.pooler:
            prefix = prefix[-1,:]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str , prefix_length : int ,gpt2_type: str = "gpt2",
                 normalize_prefix=False, phase = 'train', pooler=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.data_path = data_path
        self.phase = phase
        self.pooler = pooler
        self.clevr_feat_path = "./CLEVR_feat"
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open("/home/seungyoun/dataset/CLEVR-X/CLEVR_train_explanations_v0.7.10.json", "r") as js:
            self.json_file = json.load(js)
        print(f"Json file loaded.")

        self.max_seq_len, self.tokenized = 0, list()
        self.question_id = list()
        self.captions_tokens = list()

        for qdict in self.json_file['questions']:
            qid = qdict['image_filename'][:-4]
            exp = random.choice(qdict['factual_explanation'])
            full_caption = f"Question : {qdict['question']} \nExplanation : {exp} \nAnswer : {qdict['answer']}."
            cap_tok = torch.tensor(self.tokenizer.encode(full_caption),dtype=torch.int64)
            
            self.max_seq_len = max(cap_tok.shape[0], self.max_seq_len)
            self.question_id.append(qid) 
            self.captions_tokens.append(cap_tok)

        print(f"CLEVR-X init end.")

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import random,torch

    dataset_root_path = "/home/seungyoun/dataset/CLEVR_v1.0"

    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    with open("/home/seungyoun/dataset/CLEVR_v1.0/questions/CLEVR_train_questions.json", "r") as js:
        train_questions = json.load(js)
    # len(train_questions['questions']) # 699989
    # info : split, 'license', 'version', date
    # questions : image_index', 'program', 'question_index', 'image_filename', 'question_family_index', 'split', 'answer', 'question'
    question_dic = train_questions['questions'][random.randint(0,699989)]
    question_id = question_dic['image_filename'][:-4]

    print(question_dic['image_filename'])
    print(question_dic['question'])
    print(question_dic['answer'])
    print(tokenizer.encode(question_dic['question']))

    title_str = question_dic['question'] + " -> " + question_dic['answer']
    img = plt.imread(os.path.join(dataset_root_path, "images", "train", question_dic['image_filename']))

    print(img.shape)
    plt.imshow(img)
    plt.title(title_str)
    plt.savefig("./results/dataset_sample.png")
    """

    #dataset = ClipCLEVRDataset("/home/seungyoun/dataset/CLEVR_v1.0", 28*28+1)
    #out =dataset[0]

    with open('/home/seungyoun/dataset/CLEVR-X/train_images_ids_v0.7.10-recut.pkl', 'rb') as file:
        index2img = pickle.load(file)

    with open("/home/seungyoun/dataset/CLEVR-X/CLEVR_train_explanations_v0.7.10.json", "r") as js:
        json_file = json.load(js)

    qdict = json_file['questions'][45]
    print(qdict)

    img = plt.imread(os.path.join(dataset_root_path, "images", "train", qdict['image_filename']))
    title_str = qdict['question'] + " -> " + qdict['answer'] + "\n exp : " + qdict['factual_explanation'][0]

    plt.imshow(img)
    plt.title(title_str)
    plt.savefig("./results/dataset_sample-x.png")
