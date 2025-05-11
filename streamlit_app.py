import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to_empty(device="cpu")

    return processor, model

processor, model = load_model()

st.title("🖼️ Img_captions")

option = st.radio("Choose input method:", ("Upload an image", "Paste image URL"))
image = None

if option == "Upload an image":
    uploaded_img = st.file_uploader("Upload an image.", type=['png', 'jpg', 'jpeg'])
    if uploaded_img:
        image = Image.open(uploaded_img).convert('RGB')
elif option == "Paste image URL":
    img_url = st.text_input("Enter an Image URL.") 
    if img_url:
        try:
            image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        except Exception as e:
            st.error(f"Failed to load image from URL:  {e}")

if image:
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        inputs = processor(image, return_tensors="pt").to("cpu")

        out = model.generate(**inputs)
        #txt_out = processor.decode(out[0], skip_special_tokens=True)
        
        print(type(out))

        if isinstance(out, torch.Tensor):
            txt_out = processor.decode(out[0] if out.dim() > 1 else out, skip_special_tokens=True)
        elif hasattr(out, "sequences"):
            txt_out = processor.decode(out.sequences[0], skip_special_tokens=True)

    st.subheader("📝 Caption:")
    st.success(txt_out)