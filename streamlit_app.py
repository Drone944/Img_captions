import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
import torch

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Option 1: Load with CPU offload to avoid meta tensor issues
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float32,  # Use float32 precision
        low_cpu_mem_usage=True      # This enables better memory handling
    )
    
    # Explicitly move to CPU without using to_empty() since that's causing issues
    # We'll avoid meta tensors entirely with this approach
    model = model.to("cpu")
    
    return processor, model

st.title("🖼️ Img_captions")

# Add a warning about potential dependencies
st.warning("""
If you encounter errors about meta tensors or accelerate, you may need to install additional dependencies:
```
pip install accelerate
```
""")

try:
    processor, model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

option = st.radio("Choose input method:", ("Upload an image", "Paste image URL"))

image = None
if option == "Upload an image":
    uploaded_img = st.file_uploader("Upload an image.", type=['png', 'jpg', 'jpeg'])
    if uploaded_img:
        try:
            image = Image.open(uploaded_img).convert('RGB')
        except Exception as e:
            st.error(f"Failed to process uploaded image: {e}")

elif option == "Paste image URL":
    img_url = st.text_input("Enter an Image URL.")
    if img_url:
        try:
            response = requests.get(img_url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert('RGB')
        except Exception as e:
            st.error(f"Failed to load image from URL: {e}")

if image:
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Generating caption..."):
        try:
            inputs = processor(image, return_tensors="pt")
            
            # Keep everything on CPU to avoid device mismatch issues
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            out = model.generate(**inputs)
            txt_out = processor.decode(out[0], skip_special_tokens=True)
            
            st.subheader("📝 Caption:")
            st.success(txt_out)
        except Exception as e:
            st.error(f"Error generating caption: {e}")