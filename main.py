import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

    st.title("Python Code Generation App")

   
    st.subheader("Instructions")
    st.write("Use the following format to enter prompts: Write python code for SBERT vector embedding of a sentence")
    st.write("")    
   

