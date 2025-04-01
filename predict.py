from PIL import Image
from timeit import default_timer as timer
from tqdm.auto import tqdm
from typing import List, Dict, Tuple
import torch
import torchvision
import pathlib


def pred_and_store(
    paths: List[pathlib.Path],
    model: torch.nn.Module,
    transform: torchvision.transforms,
    class_names: list[str],
    device: str = "cuda",
) -> List[Dict]:
    # create empty list to store prediction dicts
    pred_list = []

    # loop through target paths
    for path in tqdm(paths):
        # create empty dict to store prediction info
        pred_dict = {}

        # get the sample path and true class name
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name

        # start prediciton timer
        start_time = timer()

        # open image path
        img = Image.open(path)

        # transform the image, add batch dim (unsqueeze) and send to correct device
        transform_image = transform(img).unsqueeze(0).to(device)

        # preapre the model for inferenece
        model.to(device)
        model.eval()

        # get prediction prob, label, and class
        with torch.inference_mode():
            pred_logit = model(transform_image)
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1)
            pred_class = class_names[pred_label.cpu()]

            # make sure things in the dictionary are on CPU
            pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_dict["pred_class"] = pred_class

            # end the timer
            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time - start_time, 4)

        # see if prediction matches
        pred_dict["correct"] = class_name == pred_class

        # add the dict to the list
        pred_list.append(pred_dict)

    # return the list of dict
    return pred_list


def predict(img, transforms, model, class_names) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    start_time = timer()

    # transform image and add batch dim
    img = transforms(img).unsqueeze(0)

    # put model in eval mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)

    # create prediction label and prediction prob
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    pred_time = round(timer() - start_time, 5)

    return pred_labels_and_probs, pred_time
