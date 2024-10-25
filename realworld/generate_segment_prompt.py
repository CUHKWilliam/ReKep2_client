import numpy as np
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
from torchvision.ops import box_convert
import base64
from openai import OpenAI
from utils import exec_safe
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
from tqdm import tqdm

sam = sam_model_registry["vit_h"]("sam_vit_h_4b8939.pth")  # Use vit_h, vit_l, or vit_b based on the model
sam.cuda()
mask_generator = SamAutomaticMaskGenerator(sam, stability_score_thresh=0.62,)

model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth")
model = model.cuda()
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def generate_segm_prompt():
    segm_prompt_root = "./segm_prompts"
    prompt_dirs = os.listdir(segm_prompt_root)
    for prompt_dir in tqdm(prompt_dirs):
        if prompt_dir != "prompt3":
            continue
        prompt_dir_path = os.path.join(segm_prompt_root, prompt_dir)
        image_path = os.path.join(prompt_dir_path, "image.png")
        obj_description_path = os.path.join(prompt_dir_path, "obj_description.txt")
        with open(obj_description_path, "r") as f:
            obj_description = f.read().strip()

        image_source, image = load_image(image_path)
        obj_name = obj_description.split("of")[-1].strip()
        obj_part_name = obj_description.split("of")[-2].strip()
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=obj_name,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        w, h = image_source.shape[1], image_source.shape[0]
        boxes *= torch.tensor([[w, h, w, h]]).to(image.device)
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        obj_image = image_source[int(boxes[0][1]): int(boxes[0][3]), int(boxes[0][0]): int(boxes[0][2]), :]
        import cv2
        cv2.imwrite(os.path.join(prompt_dir_path, "object.png"), obj_image)

        masks_dict = mask_generator.generate(obj_image)
        masks = np.stack([mask_dict['segmentation'] for mask_dict in masks_dict], axis=0)

        masks2 = []
        for mask in masks:
            if mask.sum().astype(np.float32) / (mask > -1).sum() > 0.05:
                masks2.append(mask)
        masks2 = np.stack(masks2, axis=0)
        masks = masks2
        
        masks2 = []
        for idx in range(len(masks)):
            mask2 = np.hstack((obj_image.copy(), np.repeat(((masks[idx].copy() > 0) * 255)[:, :, None], repeats=3, axis=-1)))
            masks2.append(mask2)
            cv2.imwrite(os.path.join(prompt_dir_path, "mask_{}.png".format(idx)), mask2[:, :, ::-1])
        masks2 = np.stack(masks2, axis=0)
        masks = masks2

        segm_prompt1 = '''
            There are totally {} pair of images. 
            For each pair, the left image is the image of {}. The right image is the segmentation mask highlighted in white to represent different parts of {}. These images are named as image i, ... (i=0, 1, 2, ...)
            Please select one of the image and use it to get {}.
            - Output: image i, `part` (i=0,1,2... is the index number).
            - Where `part` is geometry, like the edge, the center, left point, right, point, etc..
            - You can analysis the problem if needed, but please output the final result in a seperate line in the format image i, `part`.
            - At the end, output "<splitter>"
            - "ANSWER:"
            '''.format(len(masks), obj_name, obj_name, obj_description)
        

        segm_prompt3 = '''
            Write a Python function to find out the {} given segmentation image {}, {}. 
            - the input `mask` is a boolean numpy array of a segmentation mask in shapes (H, W)
            - return the mask which is a numpy array. 
            - You can `import numpy as np`, but don't import other packages
            TODO: input your answer here
            ## code start here
            def segment_object(mask):
                ## find out {} of segmentation image
                ...
                return mask
            Please directly output the code without explanations. Complete the comment in the code. Remove import lines since they will be manually imported later.'''
        prompt = "{}".format(segm_prompt1)
        with open(os.path.join(prompt_dir_path, "prompt.txt"), "w") as f:
            f.write(prompt)

if __name__ == "__main__":
    generate_segm_prompt()
    # segment("scene.png", "the cap of the pen")
