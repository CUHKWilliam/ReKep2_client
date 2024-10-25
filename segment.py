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
from utils import *

sam = sam_model_registry["vit_h"]("/data//ReKep2/sam_vit_h_4b8939.pth")  # Use vit_h, vit_l, or vit_b based on the model
sam.cuda()
mask_generator = SamAutomaticMaskGenerator(
    sam,
    # stability_score_thresh=0.62,
    # pred_iou_thresh = 0.7
)

model = load_model("/data/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/data//GroundingDINO/weights/groundingdino_swint_ogc.pth")
model = model.cuda()
BOX_TRESHOLD = 0.30
TEXT_TRESHOLD = 0.20
MARGIN = 3
# MARGIN = 20
TEMPRATURE = 0.1
TOP_P=0.05

SEGM_PROMPT1 = '''
Here are some knowledge about finding the parts given segmentation masks: {}
knowledge end.
There are totally {} pair of images. 
For each pair, the left image is the image of {} with different part highlighted in red. The right image is the segmentation mask highlighted in white to represent different parts of {}. These images are named as image i, ... (i=0, 1, 2, ...)
    Please infer what is highlighted in red for the left image one by one, and then select one of the image of {}.
    - Output: image i, `geometry` (i=0,1,2... is the index number).
    - Where `geometry` is the geometry of object, like the edge, the center, the area, left point, right, point, etc..
    - If the segmentation image does not contain the object part, think about whether we can derive the object part from this image, and select this image. For example, if the image does not correspond to "the tip of the pen", output the mask containing the pen and we can derive the tip later.
    - You can analysis the problem if needed, but please output the final result in a seperate line in the format image i, `part`.
    - For the right image, check if the corresponding object part is in black. If so, it is a background and don't use it !!!!!!!!!
    - Remember that the image index i starts from 0.
    - At the end, output "<splitter>"
    '''

SEGM_PROMPT2 = '''
    Write a Python function to find out the {} given the segmentation of image {}, {}. 
    - the input `mask` is a boolean numpy array of a segmentation mask in shapes (H, W)
    - return the mask which is a numpy array. 
    - You can `import numpy as np`, but don't import other packages
    - mask_output should still be in the shape(H, W)
    ## code start here
    def segment_object(mask):
        ...
        return mask_output
    Please directly output the code without explanations. Complete the comment in the code. Remove import lines since they will be manually imported later.'''

def add_title_to_image_cv2(img, title, font_size=1, font_color=(255, 255, 255), font_thickness=2, padding=50):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_height, img_width = img.shape[:2]
    text_size = cv2.getTextSize(title, font, font_size, font_thickness)[0]
    extra_region = np.zeros((padding, img_width, 3), dtype=np.uint8)
    new_img = np.vstack((extra_region, img)) 
    text_y = text_size[1] + (padding - text_size[1]) // 2 
    text_x = (img_width - text_size[0]) // 2
    cv2.putText(new_img, title, (text_x, text_y), font, font_size, font_color, font_thickness, lineType=cv2.LINE_AA)
    return new_img

def segment_init(obj_decription):
    import utils as utils
    return utils.ENV.part_to_pts_dict_init[obj_decription]

def segment(obj_description, timestamp=-1, image_path=None, rekep_program_dir=None, seed=0):
    import utils as utils
    if utils.ENV is not None:
        if utils.ENV.part_to_pts_dict_simulation is not None:
            ## simulate object moving
            part_to_pts_dict = utils.ENV.part_to_pts_dict_simulation.copy()
        else:
            ## track moving object
            part_to_pts_dict = utils.ENV.get_part_to_pts_dict()
        if obj_description in part_to_pts_dict[0].keys():
            return part_to_pts_dict[timestamp][obj_description]
    if image_path is None:
        import ipdb;ipdb.set_trace()
    try:
        assert image_path is not None
    except:
        print("image path is None!")
        import ipdb;ipdb.set_trace()
    image_source, image = load_image(image_path)
    obj_name = obj_description.split("of")[-1].strip()
    obj_part_name = obj_description.split("of")[-2].strip()
    obj_description = "{} of {}".format(obj_part_name, obj_name)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=obj_name.replace("the", "").strip(),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    w, h = image_source.shape[1], image_source.shape[0]
    boxes *= torch.tensor([[w, h, w, h]]).to(image.device)
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    ## pad the image a little bit for the SAM to work well
    ## TODO: select box
    box = boxes[0]
    obj_image = image_source[max(int(box[1]) - MARGIN, 0): min(int(box[3]) + MARGIN, h - 1), max(int(box[0]) - MARGIN, 0): min(int(box[2]) + MARGIN, w - 1), :]

    import cv2
    # cv2.imwrite("debug.png", obj_image[:, :, ::-1])
    original_object_img_path = os.path.join(rekep_program_dir, "object_{}.png".format(obj_description))
    cv2.imwrite(os.path.join(rekep_program_dir, "object_{}.png".format(obj_description)), obj_image[:, :, ::-1])
    masks_dict = mask_generator.generate(obj_image)
    masks = np.stack([mask_dict['segmentation'] for mask_dict in masks_dict], axis=0)
    ## mask out the padding part
    # pad_mask = np.zeros((obj_image.shape[0], obj_image.shape[1])).astype(np.bool_)
    # pad_mask[MARGIN: -MARGIN, MARGIN : -MARGIN] = True
    # pad_mask = np.logical_not(pad_mask)
    # masks[:, pad_mask] = False

    masks2 = []
    ## mask filtering
    for mask in masks:
        if mask.sum().astype(np.float32) > 20:
            masks2.append(mask)
    masks2 = np.stack(masks2, axis=0)
    masks = masks2
    masks2 = []
    mask_path0 = os.path.join(rekep_program_dir, "mask_{}_{}.png").format(obj_description, 0)
    if not os.path.exists(mask_path0):
        for idx in range(len(masks)):
            mask_path = os.path.join(rekep_program_dir, "mask_{}_{}.png").format(obj_description, idx)
            mask2 = np.hstack((obj_image.copy() * 1 + np.repeat(((masks[idx].copy() > 0) * 255)[:, :, None], repeats=3, axis=-1) * np.array([255, 0, 0]) * 0.5, np.repeat(((masks[idx].copy() > 0) * 255)[:, :, None], repeats=3, axis=-1)))
            cv2.imwrite(mask_path, mask2[:, :, ::-1])
            masks2.append(mask2)
    else:
        for idx in range(50):
            mask_path = os.path.join(rekep_program_dir, "mask_{}_{}.png").format(obj_description, idx)
            if not os.path.exists(mask_path):
                break
            mask2 = cv2.imread(mask_path)
            masks2.append(mask2)
    masks2 = np.stack(masks2, axis=0)
    masks = masks2
    if rekep_program_dir is not None:
        cache_segment_program_path = os.path.join(rekep_program_dir, "program_segm_{}.txt".format("-".join(obj_description.split(" "))))
    else:
        cache_segment_program_path = None
    if rekep_program_dir is None or not os.path.exists(cache_segment_program_path):
        contents = []
        ## TODO: use prompt
        segm_prompt_root = "/data/ReKep2/segm_prompts"
        prompt_dirs = os.listdir(segm_prompt_root)
        example_ind = 0
        for prompt_dir in prompt_dirs:
            prompt_raw_path = os.path.join(segm_prompt_root, prompt_dir, "prompt.txt")
            with open(prompt_raw_path, "r") as f:
                prompt_raw = f.read()
            contents += parse_prompt(os.path.join(segm_prompt_root, prompt_dir), prompt_raw)
        with open("./vlm_query/part_knowledge.txt", "r") as f:
            part_knowdge = f.read()
    
        client = OpenAI()
        base64_image = encode_image(image_path)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        for idx in range(len(masks)):
            base64_image = encode_image(os.path.join(rekep_program_dir, "mask_{}_{}.png".format(obj_description, idx)))
            contents.append(
                {
                "type": "text",
                    "text": "The next image is the image {}.".format(idx)
                }
            )
            contents.append(
                    {
                    "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                )
            contents.append(
                    {
                    "type": "text",
                        "text": "What is highlighted in red ?"
                    }
                )
        contents.append(
        {
                "type": "text",
                "text": SEGM_PROMPT1.format(part_knowdge, len(masks), obj_name, obj_name, obj_description),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": contents
            }
        )

        ## TODO:
        completion = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=messages,
            temperature=TEMPRATURE,
            top_p=TOP_P,
        )
        reply1 = completion.choices[0].message.content
        mask_indice = int(reply1.split("image ")[-1].split(",")[0].strip())
        part = reply1.split("image ")[-1].split(",")[-1].split("\n")[0].replace("*", "").strip()
        segm_prompt2 = SEGM_PROMPT2.format(obj_description, mask_indice, part, part)
        messages.append({
            "role": "system", "content": reply1
        })
        messages.append({
        "role": "user",
        "content":[{ 
                "type": "text",
                "text": segm_prompt2}]
        })
        completion = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=messages,
            temperature=TEMPRATURE,
            top_p=TOP_P,
        )
        reply = completion.choices[0].message.content
        code = reply.split("python\n")[1].replace("import numpy as np", "").replace("```", "")
        gvars_dict = {
            'np': np,
        }
        code = "## mask_indice: {}\n".format(mask_indice) + code
        ## TODO:
        import ipdb;ipdb.set_trace()
        if cache_segment_program_path is not None:
            with open(cache_segment_program_path, "w") as f:
                f.write(code)
    else:
        with open(cache_segment_program_path, "r") as f:
            code = f.read()
        mask_indice = int(code.split("\n")[0].split(":")[1])
        gvars_dict = {
            'np': np,
        }
    lvars = {}
    exec_safe(code, gvars_dict, lvars)
    
    mask = masks[mask_indice]
    H, W = mask.shape[0], mask.shape[1]
    mask = mask[:H, W//2:, 0] > 0

    ## TODO: erode the mask a little bit to prevent projecting to the background
    eroded_mask = cv2.erode(mask.astype(np.uint8), np.ones((2, 2), np.uint8))
    mask = eroded_mask > 0

    segm = lvars['segment_object'](mask)
    
    segm2 = np.zeros((image_source.shape[0], image_source.shape[1]))
    if segm.shape[0] != min(int(box[3]) + MARGIN, h - 1) - max(int(box[1]) - MARGIN, 0) or segm.shape[1] != min(int(box[2]) + MARGIN, w - 1) -  max(int(box[0]) - MARGIN, 0):
        segm = cv2.resize(segm.astype(np.uint8).copy(), (min(int(box[2]) + MARGIN, w - 1) -  max(int(box[0]) - MARGIN, 0), min(int(box[3]) + MARGIN, h - 1) - max(int(box[1]) - MARGIN, 0) ))

    segm2[max(int(box[1]) - MARGIN, 0): min(int(box[3]) + MARGIN, h - 1), max(int(box[0]) - MARGIN, 0): min(int(box[2]) + MARGIN, w - 1)] = segm

    ## TODO: for debug
    import cv2
    cv2.imwrite("debug2.png", (segm2 > 0).astype(np.uint8) * 255)
    # import ipdb;ipdb.set_trace()
    return segm2

if __name__ == "__main__":
    segment(image_path="scene.png", obj_description="the lip of the cup")
    # segment("scene.png", "the end of the pen")
    # segment("scene.png", "the end of the pen")