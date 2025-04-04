### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# setup class names
class_names = ["pizza", "steak", "sushi"]

### model and transforms prep ###
# create Effent2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=len(class_names), seed=42
)

# load saved model weights
effnetb2.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth",
        map_location=torch.device("cpu"),
    )
)

### predict the function ###


# create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    # Start the timer
    start_time = timer()

    # transform the target image and add batch dim
    img = effnetb2_transforms(img).unsqueeze(0)

    # put model into eval mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    # create prediction label and probability dictionary for each class (required for Gradio's output)
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    # calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


### Gradio app ###
# create title, description and article strings
title = "FoodVision Mini"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created by PyTorch"

# create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# create gradio demo
demo = gr.Interface(
    fn=predict,  # mapping function from input to output
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],  # our fn has two outputs
    examples=example_list,
    title=title,
    description=description,
    article=article,
)

# Launch the demo
demo.launch()
