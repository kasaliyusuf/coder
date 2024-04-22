import streamlit as st
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#import torch

def main():
    st.title("Python Code Generation App")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m-py")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-770m-py").to(device)

    # Get user input
    st.subheader("Instructions")
    st.write("Use the following format to enter prompts: Write python code for SBERT vector embedding of a sentence")
    st.write("")    
    query = st.text_input("Enter a prompt here: ")
    if st.button("Generate Code"):
        if query.strip().lower() == 'exit':
            st.stop()
        else:
            # Generate summary
            inputs = tokenizer(f"summarize:{query}", return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model.generate(**inputs, max_length=750)
            generated_text = tokenizer.decode(output[0]).replace("summarize:", "")

            # Display the generated summary
            st.subheader("Generated Code:")
            st.code(generated_text)

if __name__ == "__main__":
    main()
