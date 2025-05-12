# Img_captions
A Streamlit app using Salesforce's BLIP model to generate captions from image uploads or URLs.

Try it out: https://imgcaptions.streamlit.app/
---

## Installation
Clone the github repository.
```
git clone https://github.com/Drone944/Img_captions.git
cd Img_captions
```
Create a python virtual environment.
```
python -m venv .venv
```
Activate the virtual environment.
```
source .venv/bin/activate
```
Install the requirements.
```
pip install -r requirements.txt
```
---

## Usage
Run the application.
```
streamlit run streamlit_app.py
```
---

## Example Usage
![image](https://github.com/user-attachments/assets/7728e224-b3ad-4994-8693-7dbde7c2a5d0)
---

## Acknowledgement
- This project uses a pretrained model (`Salesforce/blip-image-captioning-base`) and the [Transformers](https://github.com/huggingface/transformers) library by Hugging Face.
- The Streamlit library is used to create the web application.
  
