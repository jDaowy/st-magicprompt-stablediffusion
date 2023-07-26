from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st

hub_llm = HuggingFaceHub(
    repo_id="Gustavosta/MagicPrompt-Stable-Diffusion",
    model_kwargs={'max_length':500},
    huggingfacehub_api_token = st.secrets["apikey"]
    )

st.title('🦜️🔗 Image Generation Prompt Generator')

prompt = PromptTemplate(
    input_variables=["topic"],
    template="{topic}"
 )

userIn = st.text_input('Plug in your prompt here (ex. A landscape of...)');
hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

if userIn:
    st.write(hub_chain.run(userIn))
