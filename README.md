# CLIP_prefix_caption_CLEVER

clipcap on CLEVER (visual reasoning) 

While image captioning with the "clipcap" outperform the ordinary method, the experiments on our repo demonstrate that scene grounding with large language model method has a hard time answering basic spatial reasoning task. 

However, we found that **explanation** added to the prompt slightly improve the accuracy of the task.

## Results

\ | CLEVER-minval |
---- | ---- | 
clipcap(+DINO)+finetuned | 0.54 | 
clipcap(+CLIP) | 0.16 | 
clipcap(+CLIP+explanation) | 0.23 | 

+ explanation correct, but answer incorrect

![failure case](https://github.com/SeungyounShin/CLIP_prefix_caption_CLEVER/blob/main/readme_img/clevr_failure.png?raw=true)
