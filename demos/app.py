### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_pretrained_effnet_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# setup class names
class_names = ["NORMAL", "ARMD", "DR", "MH", "ODC"]

### model and transforms prep ###
# create Effent2 model
effnet, effnet_transforms = create_pretrained_effnet_model(
    version="b3", num_classes=len(class_names), seed=42
)

# load saved model weights
effnet.load_state_dict(
    torch.load(
        f="pretrained_effnetb3_type_classification.pth",
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
    img = effnet_transforms(img).unsqueeze(0)

    # put model into eval mode and turn on inference mode
    effnet.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnet(img), dim=1)

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
title = "Reitnal Disease Classifier"
description = "An EfficientNet feature extractor computer vision model to classify retinal diseases."
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
