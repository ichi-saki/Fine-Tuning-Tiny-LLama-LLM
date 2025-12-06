import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

'''
Extracting tokens from PDF
'''
def get_text(pdf):
    text = ""
    with open(pdf, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    return text

pdf = 'dataset/cpsc-handbook-2022.pdf'
text = get_text(pdf)


'''
Creating instruction-response pairs manually and automatically
'''



'''
Loading model and tokenizer
'''
TinyLlama = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(TinyLlama)
model = AutoModelForCausalLM.from_pretrained(TinyLlama, load_in_4bit=True, torch_dtype=torch.float16)

