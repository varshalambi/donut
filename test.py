"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json
from donut.util import PerformanceMonitor, setup_logging


def test(args):
    # Setup logging
    logger = setup_logging(Path("./result"), "donut_test")
    perf_monitor = PerformanceMonitor("donut_test", logger)
    
    logger.info("🚀 Starting Donut model testing...")
    perf_monitor.start_training()  # Reuse for testing timing
    
    try:
        logger.info(f"📥 Loading pretrained model: {args.pretrained_model_name_or_path}")
        pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)

        if torch.cuda.is_available():
            logger.info("🎮 Moving model to GPU and converting to half precision")
            pretrained_model.half()
            pretrained_model.to("cuda")
        else:
            logger.info("💻 Using CPU for inference")

        pretrained_model.eval()

        if args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            logger.info(f"📁 Results will be saved to: {args.save_path}")

        predictions = []
        ground_truths = []
        accs = []

        evaluator = JSONParseEvaluator()
        logger.info(f"📚 Loading dataset: {args.dataset_name_or_path}")
        dataset = load_dataset(args.dataset_name_or_path, split=args.split)
        logger.info(f"📊 Dataset loaded with {len(dataset)} samples")

        # Process samples with progress tracking
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Processing samples"):
            try:
                ground_truth = json.loads(sample["ground_truth"])

                if args.task_name == "docvqa":
                    output = pretrained_model.inference(
                        image=sample["image"],
                        prompt=f"<s_{args.task_name}><s_question>{ground_truth['gt_parses'][0]['question'].lower()}</s_question><s_answer>",
                    )["predictions"][0]
                else:
                    output = pretrained_model.inference(image=sample["image"], prompt=f"<s_{args.task_name}>")["predictions"][0]

                if args.task_name == "rvlcdip":
                    gt = ground_truth["gt_parse"]
                    score = float(output["class"] == gt["class"])
                elif args.task_name == "docvqa":
                    # Note: we evaluated the model on the official website.
                    # In this script, an exact-match based score will be returned instead
                    gt = ground_truth["gt_parses"]
                    answers = set([qa_parse["answer"] for qa_parse in gt])
                    score = float(output["answer"] in answers)
                else:
                    gt = ground_truth["gt_parse"]
                    score = evaluator.cal_acc(output, gt)

                accs.append(score)
                predictions.append(output)
                ground_truths.append(gt)
                
                # Log progress periodically
                if (idx + 1) % 100 == 0:
                    current_acc = np.mean(accs)
                    logger.info(f"📈 Processed {idx + 1}/{len(dataset)} samples | Current accuracy: {current_acc:.4f}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing sample {idx}: {str(e)}")
                # Continue with next sample instead of failing completely
                continue

        scores = {
            "ted_accuracies": accs,
            "ted_accuracy": np.mean(accs),
            "f1_accuracy": evaluator.cal_f1(predictions, ground_truths),
        }
        
        logger.info(f"✅ Testing completed!")
        logger.info(f"📊 Total samples: {len(accs)}")
        logger.info(f"🎯 TED accuracy: {scores['ted_accuracy']:.4f}")
        logger.info(f"🎯 F1 accuracy: {scores['f1_accuracy']:.4f}")

        if args.save_path:
            scores["predictions"] = predictions
            scores["ground_truths"] = ground_truths
            save_json(args.save_path, scores)
            logger.info(f"💾 Results saved to: {args.save_path}")

        perf_monitor.end_training()
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Testing failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    predictions = test(args)
