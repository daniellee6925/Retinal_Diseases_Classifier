# Retinal Disease Classifier: Computer Vision Classifier
The Retinal Disease Classifier is a computer vision model aiming to assist in the detection and classification of retinal diseases using fundus images. It performs hierarchical classification — first identifying if disease is present, then classifying the specific type. The currently deployed model is a finetuned EffnetB3 model trained on ~1,600 retinal images. 

The model statistics are as follows 
- **Disease Identification Recall**: 85%  
- **Disease Classification Accuracy**: 82%  
- **Inference Speed**: ~0.5 seconds per image  

The classification types are as follows:
- **Normal Retina**
- **Diabetic Retinopathy**: A diabetes-related condition that damages the blood vessels in the retina, potentially leading to vision loss.  
- **Age-Related Macular Degeneration (ARMD)**: A progressive disease that blurs central vision due to damage to the macula, common in older adults.  
- **Media Haze**: A condition where clouding of the eye’s optical media (like the cornea or lens) reduces the clarity of vision.  
- **Optic Disc Cupping**: A structural change in the optic nerve head, often associated with glaucoma and increased intraocular pressure.  


If you'd like to try the final product, you can access it here: [Retinal Disease Classifier Demo](https://huggingface.co/spaces/daniellee6925/Retinal_Disease) (*it may be sleeping due to inactivity) 

---

## Features
- **Hierarchical classification pipeline**
  - Step 1: Disease presence detection
  - Step 2: Disease type classification
- **Custom Vision Models**
  - EfficientNet (EffNet) and Vision Transformer (ViT) architectures implemented from scratch
- **Model Optimization**
  - Lightweight model design with low inference latency
- **Training & Evaluation**
  - Visualized training metrics and model performance using TensorBoard
- **Deployment**
  - Deployed on Hugging Face Spaces using Gradio for a live demo interface

---
## Installation

Install the required dependencies using pip:
`pip install -r requirements.txt`


# Quick Start: Run program locally

Try running the gradio app locally with final model

`git clone https://github.com/daniellee6925/retinal_disease_classifier.git`

`cd retinal_disease_classifier`

`python demos/app.py`

# Quick Start: Train the model yourself 

## 1. download dataset and prepare model 
1. Download the data set from [IEEE Dataport](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid)

2. Run organize_data to correctly label images and split to train-val-test sets. 

    `python organize_data.py`

3. Run data_setup to create PyTorch dataloaders

    `python data_setup.py`

| Acronym | Full Name                         | Training | Validation | Test |
|---------|-----------------------------------|----------|------------|-------|
| NORMAL  | Normal Retina                     | 516      | 134        | 134   |
| DR      | Diabetic Retinopathy              | 375      | 132        | 124   |
| ARMD    | Age-Related Macular Degeneration  | 100      | 38         | 31    |
| MH      | Media Haze                        | 316      | 92         | 100   |
| ODC     | Optic Disc Cupping                | 281      | 72         | 91    |


## 2. Train the effnet model.
If you have access to a GPU, you can start training by running deploy_model_step2.py with the appropriate hyperparameters.

You can choose between pretrained/untrained Effent or ViT models

Run deploy_model_step2.py to run a training loop. You can also write your own train script using the 'train()' function in engine.py 
`python deploy_model_step2.py`

By default, the script uses:
pretrained - effnetb3 model with parameters starting from stage-7 unfreezed. The model is trained on 10 epochs
- Training takes around 5~10 mins on RTX 4080.
- Model will be saved in `models/`


### Model Details

- **Base Model**: EffnetB3
- **Dataset**: ~1,600 Retinal Images

### Training Configuration

- **Optimizer**: AdamW with learning rate: 1e-4
- **Epochs**: 10
- **Loss Function**: Cross Entropy Loss
- **Hardware**: Trained on NVIDIA RTX 4080


## 3. Predict.
You can predict a custom retinal image using 'pred_and_store()' function in predict.py 


### Example Output
| Disease Type                     | Percentage               |
|----------------------------------|--------------------------|
| Diabetic Retinopathy             | 86%                      |
| Age-Related Macular Degeneration | 12%                      |
| Media Haze                       | 0%                       |
| Optic Disc Cupping               | 2%                       |
| Prediction Time                  | 0.134 sec               |

---

### Evaluation Method
The model's performance was evaluated using two key metrics:

- **Recall (Disease Detection)**: Measures the model’s ability to correctly identify whether any retinal disease is present. This is critical in medical applications where missing a diseased case (false negative) can have serious consequences. The model achieved a **recall of 85%** for disease detection.

- **Accuracy (Disease Classification)**: Evaluates how often the model correctly identifies the specific type of disease when one is present. The model achieved a **classification accuracy of 82%** across all disease categories.

These metrics were computed on the validation set after training.

---

## 6. Deployment
The Retinal Disease Classifier is deployed using [Gradio](https://gradio.app/) on [Hugging Face Spaces](https://huggingface.co/spaces/daniellee6925/Retinal_Disease).

### Setup & Hosting
- **Framework**: [Gradio](https://github.com/gradio-app/gradio) for interactive frontend.
- **Hosting**: [Hugging Face Spaces](https://huggingface.co/spaces) for public deployment.
- **Inference Speed**: ~0.5 seconds per image on GPU.

---
## Limitations
- **Not a Diagnostic Tool**: This model is an educational prototype and should not be used as a standalone diagnostic system.
- **Class Imbalance**: Certain diseases (e.g., Media Haze) have fewer training samples compared to others, which could lead to biased predictions
- **False Positives/Negatives**: The model might misidentify a healthy retina as diseased or miss subtle signs of pathology
- **Limited Dataset Diversity**: The model fails to classify diseases other than the 4 categories that were trained on. In reality, there are around 20-30+ retinal condiions
- **No Multi-label Support**: The classifier assumes one disease per image. In reality, retinal conditions may co-exist which the model doesn't handle 

---

## Future Improvements

- Improve accuracy by training on more data
- Add classification for more disease types
- Add Mult-label support


---

## Citation
If you use the [IEEE Dataport](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid)
 dataset in your work, please cite the following authors:

**Citation Author(s):**

- Samiksha Pachade (Center of Excellence in Signal and Image Processing, Shri Guru Gobind Singhji Institute of Engineering and Technology, Nanded, India)  
- Prasanna Porwal (Center of Excellence in Signal and Image Processing, Shri Guru Gobind Singhji Institute of Engineering and Technology, Nanded, India)  
- Dhanshree Thulkar (Veermata Jijabai Technological Institute, Mumbai, India)  
- Manesh Kokare (Center of Excellence in Signal and Image Processing, Shri Guru Gobind Singhji Institute of Engineering and Technology, Nanded, India)  
- Girish Deshmukh (Eye Clinic, Sushrusha Hospital, Nanded 431601, India)  
- Vivek Sahasrabuddhe (Department of Ophthalmology, Shankarrao Chavan Government Medical College, Nanded 431606, India)  
- Luca Giancardo (Center for Precision Health, School of Biomedical Informatics, University of Texas Health Science Center at Houston (UTHealth), Houston, USA)  
- Gwenolé Quellec (Inserm, UMR 1101, Brest, F-29200, France)  
- Fabrice Mériaudeau (ImViA EA 7535 and ERL VIBOT 6000, Université de Bourgogne, France)

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Contact

Questions, feedback, or collab ideas? Reach out: [daniellee6925@gmail.com](mailto:daniellee6925@gmail.com)

