import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
import torch
import warnings

# Suppress warnings to make output cleaner
warnings.filterwarnings("ignore")

@st.cache_resource
def load_model():
    try:
        # Step 1: Load the processor normally
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Step 2: First load the model config only (this avoids loading weights)
        model_config = BlipForConditionalGeneration.config_class.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Step 3: Create model instance with config but without weights
        model = BlipForConditionalGeneration(config=model_config)
        
        # Step 4: Load state dict separately
        state_dict = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            return_dict=False,
            state_dict=True  # Only get the state dict
        )
        
        # Step 5: Load the state dict into our model
        model.load_state_dict(state_dict)
        
        return processor, model
    except Exception as e:
        # If the above approach fails, try direct loading with warning
        st.warning(f"Advanced loading failed: {e}\nTrying simpler approach...")
        
        # Direct loading attempt - this might work on some systems
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float32
        )
        return processor, model

st.title("🖼️ Img_captions")

try:
    with st.spinner("Loading model... (this may take a minute)"):
        processor, model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.error("""
    This error usually indicates that the transformers library is trying to use meta tensors 
    and you need the 'accelerate' library. Try installing it with:
    ```
    pip install accelerate
    ```
    Then restart this Streamlit app.
    """)
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
            
            # Make sure inputs are on the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            out = model.generate(**inputs)
            txt_out = processor.decode(out[0], skip_special_tokens=True)
            
            st.subheader("📝 Caption:")
            st.success(txt_out)
        except Exception as e:
            st.error(f"Error generating caption: {e}")