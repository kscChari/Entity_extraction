import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, list_repo_files

# Set up clean, professional page layout
st.set_page_config(page_title="Mistral Entity Extractor", layout="wide", page_icon="🛒")

# 1. Dynamically find and cache the GGUF model loading sequence
@st.cache_resource
def load_quantized_model():
    # Your exact public model repository path
    repo_id = "ksckaushal/Mistral-Entity-Extraction"
    
    with st.spinner("Initializing system and loading optimized Mistral GGUF model from Hugging Face Hub..."):
        try:
            # Look up files in your public repo to find the .gguf file dynamically
            files = list_repo_files(repo_id)
            gguf_filename = next((f for f in files if f.endswith('.gguf')), None)
            
            if not gguf_filename:
                st.error("Error: Could not find a file ending in '.gguf' in your repository.")
                return None

            # Download the GGUF file 
            model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)
            
            # Load the model utilizing the free tier's 2 available vCPU threads
            llm = Llama(model_path=model_path, n_ctx=2048, n_threads=2)
            return llm
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None

# Initialize the model 
llm = load_quantized_model()

# 2. Define the Alpaca Prompt Template used during your fine-tuning
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 3. Build a high-quality, recruiter-friendly Dashboard UI
st.title("🛒 Fine-Tuned Mistral Entity Extractor")
st.markdown(
    """
    **Developer Portfolio Project**  
    This production-ready application serves a fine-tuned **Mistral-7B** LLM optimized via **Q5_K_M GGUF quantization** 
    to extract custom attributes and entities from raw text into clean, structured JSON format. 
    *Running 100% serverless on commodity free-tier CPU hardware using Llama.cpp.*
    """
)

st.write("---")

# Split user interface into 2 proportional columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input Dataset")
    review_input = st.text_area(
        label="Paste laptop or product review text below:",
        height=250,
        placeholder="Example: 'The battery life on this Dell Inspiron is spectacular, reaching almost 10 hours, but the cooling fan is terribly loud under load...'"
    )
    
    # Custom engineering settings showcased to recruiters
    with st.expander("⚙️ Inference Parameters (Hyperparameters)"):
        temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, 
                         help="Lower temperatures yield highly deterministic, rigidly structured JSON outputs.")
        max_tokens = st.slider("Max New Tokens", min_value=64, max_value=1024, value=512, step=64)

    submit_btn = st.button("Extract Structured Entities", type="primary", use_container_width=True)

with col2:
    st.subheader("⚡ Structured Output (JSON)")
    
    if submit_btn:
        if review_input.strip() and llm is not None:
            # Set the prompt instruction
            instruction = "Extract entities in the input review in a JSON format."
            formatted_prompt = ALPACA_PROMPT.format(instruction, review_input, "")
            
            with st.spinner("Processing text weights on CPU..."):
                try:
                    # Run inference via Llama.cpp
                    raw_output = llm(
                        formatted_prompt,
                        max_tokens=max_tokens,
                        temperature=temp,
                        stop=["###", "\n\n"]  # Strict generation boundary handling
                    )
                    
                    extracted_json = raw_output['choices'][0]['text'].strip()
                    
                    # Display output beautifully as a code block
                    st.success("Extraction Complete!")
                    st.code(extracted_json, language="json")
                    
                except Exception as e:
                    st.error(f"Inference Engine Error: {str(e)}")
        elif llm is None:
            st.error("Model engine failed to initialize. Review the Hugging Face space logs.")
        else:
            st.warning("Please provide an input review text first!")