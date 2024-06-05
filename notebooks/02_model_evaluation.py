# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftConfig, PeftModel

# Load the saved model and tokenizer
model = AutoModelForCausalLM.from_pretrained("../model/output/model")
tokenizer = AutoTokenizer.from_pretrained("../model/output/tokenizer")
# %%
medical_text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# %%
prompt = """<s>
<<SYS>>
    You are a medical expert, please answer the following question in a friendly manner.
<</SYS>>
<<QUESTION>>
    How to prevent Urinary Tract Infections in Children ?
<</QUESTION>>
<<ANSWER>>

"""
# %%
generated_text = medical_text_generator(
    prompt, 
    max_length=500,
    num_return_sequences=1)

# %%
print(generated_text)
# %%
