import gdown
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Function to download a folder from Google Drive
def download_folder_from_drive(folder_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", output=output_dir, quiet=False)

from huggingface_hub import login
login("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")

# Download the specified folder
folder_id = '1LUCcx8C8U_19O7dlcjSzE8-NEg_8gDlb'
output_directory = 'downloaded/'
download_folder_from_drive(folder_id, output_directory)

# Load the model and tokenizer
@st.cache_resource
def load_model():
    base_model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        # device_map=torch.device('cuda:0'),
        device_map="auto",
        trust_remote_code=True,
    )
    
    eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
    ft_model = PeftModel.from_pretrained(base_model, "downloaded/checkpoint-5")
    
    return ft_model, eval_tokenizer

# Load the model and tokenizer
ft_model, eval_tokenizer = load_model()

# Streamlit app layout
st.title("Mistral-7B Text Generation")
st.write("Enter a prompt below and generate text using the Mistral model.")

eval_prompt = st.text_area("Prompt", "The following is a note by Eevee the Dog, which doesn't share anything too personal: # ")

if st.button("Generate"):
    if eval_prompt:
        model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        
        ft_model.eval()
        with torch.no_grad():
            output_ids = ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)
            generated_text = eval_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        st.subheader("Generated Text")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt to generate text.")
