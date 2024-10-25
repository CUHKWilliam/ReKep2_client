import base64
from openai import OpenAI
import os
import cv2
import json
import parse
import numpy as np
import time
from datetime import datetime
import re

TEMPERATURE = 0.1
TOP_P=0.05
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ConstraintGenerator:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')
        with open(os.path.join(self.base_dir, 'prompt_template.txt'), 'r') as f:
            self.prompt_template = f.read()

    def _build_prompt(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        prompt_text = self.prompt_template.format(instruction)
        # save prompt
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_template.format(instruction=instruction)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages

    def _parse_and_save_constraints(self, output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        for i, line in enumerate(lines):
            if line.startswith("def "):
                start = i
                name = line.split("(")[0].split("def ")[1]
            if line.startswith("    return "):
                end = i
                functions[name] = lines[start:end+1]
        # organize them based on hierarchy in function names
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]  # last one is the constraint idx
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w") as f:
                for name in groupings[key]:
                    ## TODO: process for the target constraint.
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")
    
    def _parse_other_metadata(self, output):
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])
        # find grasp_keypoints
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}"
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        # convert into list of ints
        grasp_keypoints = grasp_keypoints['grasp_keypoints'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints = [int(x.strip()) for x in grasp_keypoints]
        data_dict['grasp_keypoints'] = grasp_keypoints
        # find release_keypoints
        release_keypoints_template = "release_keypoints = {release_keypoints}"
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        # convert into list of ints
        release_keypoints = release_keypoints['release_keypoints'].replace("[", "").replace("]", "").split(",")
        release_keypoints = [int(x.strip()) for x in release_keypoints]
        data_dict['release_keypoints'] = release_keypoints
        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate(self, img, instruction, metadata):
        """
        Args:
            img (np.ndarray): image of the scene (H, W, 3) uint8
            instruction (str): instruction for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        # create a directory for the task
        fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
        self.task_dir = os.path.join(self.base_dir, fname)
        os.makedirs(self.task_dir, exist_ok=True)
        # save query image
        image_path = os.path.join(self.task_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])
        # build prompt
        messages = self._build_prompt(image_path, instruction)
        # stream back the response
        stream = self.client.chat.completions.create(model=self.config['model'],
                                                        messages=messages,
                                                        temperature=TEMPERATURE,
                                                        top_p=TOP_P,
                                                        max_tokens=self.config['max_tokens'],
                                                        stream=True)
        output = ""
        start = time.time()
        for chunk in stream:
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
        print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
        # save raw output
        with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
            f.write(output)
        # parse and save constraints
        self._parse_and_save_constraints(output, self.task_dir)
        # save metadata
        metadata.update(self._parse_other_metadata(output))
        self._save_metadata(metadata)
        return self.task_dir



class ConstraintGenerator2:
    def __init__(self, config, prompt_template_path=None):
        self.config = config
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')

        if prompt_template_path is None:
            prompt_template_path = os.path.join(self.base_dir, 'prompt_template.txt')
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()

    def _build_prompt(self, image_path, instruction, hint="",):
        img_base64 = encode_image(image_path)
        with open("./vlm_query/geometry_knowledge.txt", "r") as f:
            geometry_knowledge = f.read()
        prompt_text = self.prompt_template.format(instruction, geometry_knowledge, geometry_knowledge)
        # save prompt
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        prompt_texts = prompt_text.split("<STEP SPLITTER>")

        messages1 = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_texts[0]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        
        messages2 = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_texts[1]
                    },
                ]
            }
        messages = [messages1, messages2]
        return messages

    def _parse_and_save_constraints(self, output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        meta_data = {}
        max_stage = -1
        objects_to_segment = []
        for i, line in enumerate(lines):
            if line.strip().startswith("def"):
                start = i
                name = line.split("(")[0].split("def ")[1]
                stage = int(name.split("_")[1])
                if stage > max_stage:
                    max_stage = stage
            if line.strip().startswith("return"):
                end = i
                functions[name] = lines[start:end+1]
            # if line.strip().startswith("object_to_segment"):
            #     ret = {}
            #     exec("".join(lines[i:]).replace("`", ""), {}, ret)
            #     objects_to_segment = ret['object_to_segment']
            if line.strip().startswith('"""constraints: <'):
                line = line.split('<')[1].split('>')[0].split(",")
                for idx, obj in enumerate(line):
                    obj = obj.replace("\"", "").strip()
                    if obj == "grasp":
                        grasp_obj = line[-1].replace("\"", "").strip()
                        if grasp_obj != "":
                            objects_to_segment.append(grasp_obj)
                        continue
                    else:
                        if idx == len(line) - 1:
                            continue
                    if "constraints" in obj:
                        continue
                    objects_to_segment.append(obj)
        objects_to_segment = list(set(objects_to_segment))

        meta_data.update({
            "num_stage": max_stage,
            "object_to_segment": objects_to_segment
        })
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(meta_data, f)
        # organize them based on hierarchy in function names
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]  # last one is the constraint idx
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")
        
    def _parse_other_metadata(self, output):
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])
        # find grasp_keypoints
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}"
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        # convert into list of ints
        grasp_keypoints = grasp_keypoints['grasp_keypoints'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints = [int(x.strip()) for x in grasp_keypoints]
        data_dict['grasp_keypoints'] = grasp_keypoints
        # find release_keypoints
        release_keypoints_template = "release_keypoints = {release_keypoints}"
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        # convert into list of ints
        release_keypoints = release_keypoints['release_keypoints'].replace("[", "").replace("]", "").split(",")
        release_keypoints = [int(x.strip()) for x in release_keypoints]
        data_dict['release_keypoints'] = release_keypoints
        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate(self, img, instruction, rekep_program_dir=None, hint="", seed=None, ):
        """
        Args:
            img (np.ndarray): image of the scene (H, W, 3) uint8
            instruction (str): instruction for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        if rekep_program_dir is None:
            # create a directory for the task
            fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
            self.task_dir = os.path.join(self.base_dir, fname)
            os.makedirs(self.task_dir, exist_ok=True)
            rekep_program_dir = self.task_dir
        image_path = os.path.join(rekep_program_dir, 'query_img.png')
        # if not os.path.exists(image_path):
        cv2.imwrite(image_path, img[..., ::-1])
        output_raw_file = os.path.join(rekep_program_dir, "output_raw.txt")
        self.task_dir = rekep_program_dir
        if not os.path.exists(output_raw_file):
            # build prompt
            messages = self._build_prompt(image_path, instruction + ". HINT: {}.".format(hint), hint)
            # conversations = [{"role": "system", "content": "You are a helpful assistant."}]
            conversations = []
            conversations.append(messages[0])
            # stream back the response
            stream = self.client.chat.completions.create(model="chatgpt-4o-latest",
                                                            messages=conversations,
                                                            stream=True,
                                                            temperature=TEMPERATURE,
                                                            top_p =TOP_P,)
            output1 = ""
            start = time.time()
            for chunk in stream:
                print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
                if chunk.choices[0].delta.content is not None:
                    output1 += chunk.choices[0].delta.content
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
            conversations.append(
                {"role": "system", "content": "{}".format(output1)}
            )
            conversations.append(messages[1])
            # stream back the response
            stream = self.client.chat.completions.create(model="chatgpt-4o-latest",
                                                            messages=conversations,
                                                            stream=True,
                                                            temperature=TEMPERATURE,
                                                            top_p = TOP_P,)
            output2 = ""
            start = time.time()
            for chunk in stream:
                print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
                if chunk.choices[0].delta.content is not None:
                    output2 += chunk.choices[0].delta.content
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
            # save raw output
            with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
                f.write(output2)
            output = output2
        else:
            with open(output_raw_file, "r") as f:
                output = f.read()
        # parse and save constraints
        flag_contraint = False
        for file in os.listdir(self.task_dir):
            if "constraints" in file:
                flag_contraint = True
                break
        if not flag_contraint:
            self._parse_and_save_constraints(output, self.task_dir)
        return self.task_dir
