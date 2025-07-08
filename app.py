"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import logging
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from donut import DonutModel
from donut.util import setup_logging


class DonutDemo:
    """Encapsulated Donut demo class to avoid global variables"""
    
    def __init__(self, task_name: str, pretrained_path: str):
        self.logger = setup_logging(Path("./result"), "donut_demo")
        self.task_name = task_name
        self.pretrained_path = pretrained_path
        
        # Setup task prompt
        if task_name == "docvqa":
            self.task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        else:  # rvlcdip, cord, ...
            self.task_prompt = f"<s_{task_name}>"
            
        # Load model
        self.logger.info(f"📥 Loading model: {pretrained_path}")
        self.model = DonutModel.from_pretrained(pretrained_path)

        if torch.cuda.is_available():
            self.logger.info("🎮 Moving model to GPU and converting to half precision")
            self.model.half()
            self.model.to("cuda")
        else:
            self.logger.info("💻 Using CPU for inference")

        self.model.eval()
        self.logger.info("✅ Model loaded successfully")

    def process_vqa(self, input_img, question):
        """Process VQA task"""
        try:
            input_img = Image.fromarray(input_img)
            user_prompt = self.task_prompt.replace("{user_input}", question)
            output = self.model.inference(input_img, prompt=user_prompt)["predictions"][0]
            self.logger.debug(f"VQA processed - Question: {question[:50]}...")
            return output
        except Exception as e:
            self.logger.error(f"Error in VQA processing: {str(e)}")
            return {"error": str(e)}

    def process_general(self, input_img):
        """Process general document understanding task"""
        try:
            input_img = Image.fromarray(input_img)
            output = self.model.inference(image=input_img, prompt=self.task_prompt)["predictions"][0]
            self.logger.debug("General document processed")
            return output
        except Exception as e:
            self.logger.error(f"Error in general processing: {str(e)}")
            return {"error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="docvqa")
    parser.add_argument("--pretrained_path", type=str, default="naver-clova-ix/donut-base-finetuned-docvqa")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--sample_img_path", type=str)
    args, left_argv = parser.parse_known_args()

    try:
        # Initialize demo
        demo_instance = DonutDemo(args.task, args.pretrained_path)
        
        # Setup example samples
        example_sample = []
        if args.sample_img_path:
            example_sample.append(args.sample_img_path)

        # Create Gradio interface
        demo = gr.Interface(
            fn=demo_instance.process_vqa if args.task == "docvqa" else demo_instance.process_general,
            inputs=["image", "text"] if args.task == "docvqa" else "image",
            outputs="json",
            title=f"Donut 🍩 demonstration for `{args.task}` task",
            examples=[example_sample] if example_sample else None,
        )
        
        demo_instance.logger.info(f"🚀 Launching Gradio interface for task: {args.task}")
        demo.launch(server_name=args.url, server_port=args.port)
        
    except Exception as e:
        print(f"❌ Failed to start demo: {str(e)}")
        raise
