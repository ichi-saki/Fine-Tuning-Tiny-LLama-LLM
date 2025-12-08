import sys
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, AdapterConfig, PrefixTuningConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
import re
import json



'''
Extracting text from PDF
'''
def get_text(pdf):
    text = ""
    with open(pdf, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    return text

'''
Spit text into chunks
'''
def get_chunks(text, chunk_size=400, overlap=80):
    chunks = []
    start = 0
    i = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    print(f'Length of chunks: {len(chunks)} chunks created')
    return chunks

def create_answer(chunk):
    sentences = chunk.split('. ')
    if len(sentences) >= 3:
        answer_sentences =[]
        for s in sentences:
            if len(s.strip()) < 10:
                continue
            if len(' '.join(answer_sentences + [s])) < 400:
                answer_sentences.append(s)
            else:
                break

        answer = '. '.join(answer_sentences)
        if answer and not answer.endswith('.'):
            answer += '.'
    else:
        answer = chunk[:400]
        if '.' in answer:
            answer = answer.rsplit('.', 1)[0] + '.'
        elif len(answer) > 150:
            answer = answer[:350] + '...'

    answer = ' '.join(answer.split())

    if answer.count('•') > 3 and 'course' not in answer.lower():
        return ""

    return answer

'''
Creating instruction-response pairs manually and automatically
'''
def manual_IR_pairs():
    manual_pairs = [
        
        {
            "instruction": "What are the core courses required for a computer science undergraduate degree?",
            "input": "",
            "output": "According to the 2022-2023 CPSC handbook, the core courses required for a Computer Science undergraduate degree are: Lower-Division Core: - CPSC 120 - Introduction to Programming - CPSC 121 - Object-Oriented Programming - CPSC 131 - Data Structures - CPSC 223 (choose one: Python, Java, or C++) - CPSC 240 - Computer Organization and Assembly Language Upper-Division Core: - CPSC 253 or 353 - Visual Programming - CPSC 315 - Professional Ethics in Computing - CPSC 323 - Compilers and Languages - CPSC 332 - File Structures and Database Systems - CPSC 335 - Algorithm Engineering - CPSC 351 - Operating Systems Concepts - CPSC 362 - Foundations of Software Engineering - CPSC 471 - Computer Communications - CPSC 481 - Artificial Intelligence - CPSC 490 - Computer Science Project - CPSC 491 - Senior Capstone Project"
        },
        {
            "instruction": "Describe the rules for completing a senior project, including prerequisites",
            "input": "",
            "output": "Based on the CPSC handbook, to complete the senior project (CPSC 491 - Senior Capstone Project):  Prerequisites: 1. CPSC 490 - Computer Science Project 2. CPSC 362 - Foundations of Software Engineering 3. Senior standing  Requirements: - Students work on a substantial software project - Must demonstrate comprehensive computer science knowledge - Typically completed over one semester - Requires formal project proposal and approval - Final project demonstration and documentation  Note: CPSC 491 has the longest chain of prerequisites, so students should plan their schedule to make steady progress toward meeting these prerequisites throughout their program."
        },
        {
            "instruction": "What are the degree requirements for graduation?",
            "input": "",
            "output": "The degree requirements for the B.S. in Computer Science at CSU Fullerton include:  1. **Total Units**: 120 credits minimum  2. **Major Requirements**:    - Lower-Division Core (5 courses, 15 units)    - Mathematics Requirements (4 courses, 13 units): MATH 270A, 270B, 150A, 150B, and 338    - Science and Mathematics Electives (12 units with lab)    - Upper-Division Core (11 courses, 33 units)    - Computer Science Major Electives (15 units)  3. **Grade Requirements**:    - Minimum GPA of 2.0 in major courses    - Minimum grade of C in CPSC 490 and 491    - Minimum grade of C- in GE courses including MATH 150A, 338, and EGCE/CP/EE/ME 401    - Minimum grade of D- in all other major courses  4. **General Education (GE)**: 48 units across 18 categories  5. **Additional Requirements**:    - Minimum 30 units of upper-division courses    - Minimum 24 units in residence at CSUF    - Last 30 units must be completed at CSUF"
        },
        {
            "instruction": "What is the cybersecurity concentration and what courses does it require?",
            "input": "",
            "output": "The Cybersecurity Concentration prepares students for entry-level cybersecurity positions. Requirements:  Required Course (3 units): - CPSC 456 - Network Security Fundamentals  Cybersecurity Electives (9 units from): - CPSC 452 - Cryptography - CPSC 454 - Cloud Computing and Security - CPSC 455 - Web Security - CPSC 458 - Malware Analysis - CPSC 459 - Blockchain Technologies - CPSC 483 - Introduction to Machine Learning - EGCP 447 - Introduction to Cyber-Physical Systems Security"
        },
        {
            "instruction": "What are the GE requirements for CS majors?",
            "input": "",
            "output": "CS majors have the following GE requirements (2022-2023):  Required GE Categories: - A.1 Oral Communications (3 units) - e.g., HCOM 102 - A.2 Written Communications (3 units) - e.g., ENGL 101 - C.1 Introduction to Art (3 units) - C.2 Introduction to Humanities (3 units) - C.3 Explorations of Arts and Humanities (3 units) - D.1 Introduction to Social Sciences (3 units) - D.2 American History, Institutions, and Values (3 units) - Z Cultural Diversity (3 units) - Graduation Requirement: POSC 100 - American Government (3 units)  Waived for CS Majors: - A.3 Critical Thinking - B.2 Life Science - D.4 Explorations in Social Sciences - E Lifelong Learning and Self Development  Total: 27 units across 10 categories"
        },
        {
            "instruction": "What are the mathematics requirements for CS majors?",
            "input": "",
            "output": "Mathematics requirements for CS majors:  Required Mathematics Courses: 1. MATH 270A - Discrete Mathematics (3 units) 2. MATH 270B - Discrete Mathematics (3 units) 3. MATH 150A - Calculus I (4 units) 4. MATH 150B - Calculus II (4 units) 5. MATH 338 - Statistics (3 units)  Total: 17 units of mathematics  Note: MATH 150A requires MATH 125 (Precalculus) or equivalent as prerequisite."
        },
        {
            "instruction": "Can I get credit for internships?",
            "input": "",
            "output": "Yes, computer science internships can count for elective credit:  - Course: EGGN 495 (Academic Internship) - Credits based on hours:   • 40–60 hours = 1 unit   • 80–100 hours = 2 units   • 120–150 hours = 3 units  Requirements: - Internship must be related to computer science - No class meetings ('supervision only') - Grade based on internship performance - Submit final report including:   1. Project details   2. Tasks completed   3. Learning outcomes   4. Academic benefits  Register through: Center for Internships & Community Engagement (CICE)"
        },
        {
            "instruction": "What does this handbook cover?",
            "input": "",
            "output": "This handbook covers information on how to complete a B.S. or a Minor in Computer Science, and contains information relevant to students pursuing them. If you are pursuing a Masters degree, please refer to the Graduate Handbook instead of this document."
        },
        {
            "instruction": "Where can I get advising?",
            "input": "",
            "output": "You can receive advising in several locations including: - CS Department Advising: http://www.fullerton.edu/ecs/cs/resources/advisement.php – Center for Academic Support in ECS (CASECS): http://www.fullerton.edu/ecs/resources/casecs.php – Student Success Center: http://www.fullerton.edu/ecs/resources/StudentSuccessCenter.php – Academic Advisement Center (GE advising): https://www.fullerton.edu/aac/"
        },
        {
            "instruction": "How many units of Upper Division CS electives are required to complete and what coursees are available?",
            "input": "",
            "output": "At least 12.0 units of Upper Division CS electives are required. You may need to take additional electives if you are short on units due to the Placement Examination, transfer, or other circumstances. The following courses may count as Upper Division CS Electives: - CPSC 301 - Programming Lab Practicum (2 units) - CPSC 349 - Web Front-End Engineering (3 units) - CPSC 375 - Introduction to Data Science and Big Data (3 units) - CPSC 386 - Introduction to Game Design and Production (3 units) - CPSC 411 - Mobile Device Application Programming (3 units) - CPSC 411A - Mobile Device Application Programming for Android (3 units) - CPSC 431 - Database and Applications (3 units) - CPSC 439 - Theory of Computation (3 units) - CPSC 440 - Computer System Architecture (3 units) - CPSC 449 - Web Back-End Engineering (3 units) - CPSC 452 - Cryptography (3 units) - CPSC 454 - Cloud Computing and Security (3 units) - CPSC 455 - Web Security (3 units) - CPSC 456 - Network Security Fundamentals (3 units) - CPSC 458 - Malware Analysis (3 units) - CPSC 459 - Blockchain Technologies (3 units) - CPSC 462 - Software Design (3 units) - CPSC 463 - Software Testing (3 units) - CPSC 464 - Software Architecture (3 units) - CPSC 466 - Software Process (3 units) - CPSC 474 - Parallel and Distributed Computing (3 units) - CPSC 479 - Introduction to High Performance Computing (3 units) - CPSC 483 - Introduction to Machine Learning (3 units) - CPSC 484 - Principles of Computer Graphics (3 units) - CPSC 485 - Computational Bioinformatics (3 units) - CPSC 486 - Game Programming (3 units) - CPSC 489 - Game Development Project (3 units) - CPSC 499 - Independent Study (3 units)"
        },
         {
            "instruction": "What happens if a student is placed on academic probation?",
            "input": "",
            "output": "Students on probation must meet with an advisor each semester. A registration hold is placed until advisement is completed. Students should discuss strategies to improve grades and address academic challenges."
        },
        {
            "instruction": "Can CS students study abroad?",
            "input": "",
            "output": "Yes, CS students can study abroad through the Study Abroad Office. Programs should fit within the study plan. Students should consult with advisors early to ensure courses will transfer appropriately."
        },
        {
            "instruction": "How does independent study (CPSC 499) work?",
            "input": "",
            "output": "CPSC 499 allows students to study topics not covered in regular courses under faculty supervision. Applications require faculty advisor and department chair approval. Up to 3 units per semester, maximum 6 units toward the degree."
        }
    ]
    return manual_pairs

def auto_IR_pairs(chunks):
    auto_pairs = []
    topic_questions = {
        "tuffix": "What is Tuffix and how do I install it?",
        "transfer": "How do transfer credits work for computer science?",
        "probation": "What happens if I'm placed on academic probation?",
        "advising": "How do I get academic advising in the CS department?",
        "elective": "What computer science electives are available?",
        "minor": "What are the requirements for a CS minor?",
        "grade": "What are the grade requirements for CS courses?",
        "prerequisite": "What are the prerequisites for {}?",
        "accreditation": "Is the CS program accredited?",
        "international": "What resources are available for international students?",
        "study abroad": "Can I study abroad as a CS major?",
        "independent study": "How does independent study (CPSC 499) work?",
        "waitlist": "How do I petition for a closed class?",
        "mathematics": "What mathematics courses are required for CS?",
        "science": "What science electives are required?",
        "ge": "What are the GE requirements?",
        "internship": "Can I get credit for internships?",
        "cybersecurity": "What is the cybersecurity concentration?",
        "senior project": "What is the senior capstone project?"
    }

    used_topics = set()

    for i, chunk in enumerate(chunks[50:120]):
        chunk = ' '.join(chunk.split())
        if len(chunk) < 200:
            continue
        
        if chunk.count('•') > 5 and 'course' not in chunk.lower():
            continue
        if 'Figure' in chunk or 'Table' in chunk: 
            continue

        lowered = chunk.lower()

        matching_topics = []

        for topic, q_template in topic_questions.items():
            if topic in lowered:
                matching_topics.append((topic, q_template))    

        #find most relevant topic
        question = None
        if matching_topics:
            for topic, q_template in matching_topics:
                if topic not in used_topics or list(used_topics).count(topic) < 2:
                    if "{}" in q_template:
                        course_match = re.search(r'(CPSC|MATH)\s+\d{3}[A-Z]?', chunk)
                        if course_match:
                            question = q_template.format(course_match.group(0))
                        else:
                            question = q_template.format("required courses")
                    else:
                        question = q_template

                    used_topics.add(topic)
                    break                    

        if not question:
            line = chunk.split('.')[0][:100]
            if len(line) > 30:
                question = f"what does the CS handbook say about: {line}"
            else:
                question = f"what information is in section {i+31} of the CS handbook?"
        
        #clean and shorten answer
        answer = create_answer(chunk)

        if len(answer) < 100:
            continue

        auto_pairs.append({
            "instruction": question,
            "input": "",
            "output": answer
        })

        if len(auto_pairs) >= 50:
            break

    return auto_pairs


'''
Create dataset from PDF text and get instruction response pairs
'''
def create_dataset(text):
    print('calling get_chunks')
    chunks = get_chunks(text)

    manual_pairs = manual_IR_pairs()

    auto_pairs = auto_IR_pairs(chunks)

    all_pairs = manual_pairs + auto_pairs[:60]

    pairs = []
    seen = set()
    for pair in all_pairs:
        key = pair['instruction'][:50]
        if key not in seen:
            pairs.append(pair)
            seen.add(key)


    print(f"created dataset with length of {len(pairs)} total pairs")
    
    return pairs

def filter_dataset(dataset):
    indicies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 21]

    pairs = []
    for i in indicies:
        pair = dataset[i]

        answer = pair['output']
        # Fix for pair 14  
        if i == 14 and "- Senior Capstone Project" in answer:
            answer = "CPSC 120 has no prerequisites. It's the first course in the CS sequence. Complete CPSC 120 before CPSC 121, then CPSC 131."
        
        # FIX FOR PAIR 15 
        if i == 15 and "vision CS Electives" in answer:
            answer = "Use ASSIST.org to check course equivalencies. Transfer courses appear in Titan Degree Audit after evaluation. Some courses may need department approval."
        
        # FIX FOR PAIR 
        if i == 21 and "marked with an asterisk" in answer:
            answer = "Minimum C grade in CPSC 490 and 491. Minimum C- in GE courses including MATH courses. Minimum D- in other major courses. Overall GPA must be 2.0+."
        
        pairs.append({
            "instruction": pair['instruction'],
            "input": pair['input'],
            "output": answer
        })

    return pairs


'''
Format dataset for TinyLlama chat
'''
def format(dataset):
    data = []

    for item in dataset:
        text = f"<|system|>\nYou are a helpful assistant knowledgeable about the CSU Fullerton Computer Science undergraduate program. Answer questions based on the 2022-2023 Computer Science Handbook.</s>\n"
        text += f"<|user|>\n{item['instruction']}</s>\n"
        text += f"<|assistant|>\n{item['output']}</s>"

        data.append({"text": text})

    return data

'''
pdf = 'dataset/cpsc-handbook-2022.pdf'
text = get_text(pdf)
dataset = create_dataset(text)
print(f'length of dataset before filtering {len(dataset)}')
dataset = filter_dataset(dataset)
print(f'length of dataset after filtering {len(dataset)}')
dataset = format(dataset)
with open('finetune_data.json', 'w') as f:
    json.dump(dataset, f, indent=2)
'''


'''
load dataset for training now
'''
with open('finefune_data.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

train_test_data = dataset.train_test_split(test_size=0.2, seed=42)
train_data = train_test_data["train"]
test_data = train_test_data["test"]

print(f'lenght of train {len(train_data)}')
print(f'length of test {len(test_data)}')


'''
tokenize data
'''
def tokenize(examples):
    tokenized = tokenizer(examples["text"],
                          truncation=True,
                          padding="max_length",
                          max_length=512,
                          returns_tensors="pt")
    
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized

#tokenize data
tokenized_train = train_data.map(tokenize, batched=True)
tokenized_test = test_data.map(tokenize, batched=True)

#format for pytorch
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


'''
Implement fine tuning methods
'''

### LoRA fine tuning
def lora_finetuning(model):
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model

### Adapter Fine-tuning
def adapter_finetuning(model):
    for param in model.parameters():
        param.require_grad = False

    adapter_config = AdapterConfig(
        md_adapter=True,
        output_adapter=True,
        reduction_factor=16,
        non_linearity="relu"
    )

    for name, module in model.named_modules():
        if "attention" in name or "mlp" in name:
            module.add_adapter("handcook_adapter", config=adapter_config)

    model.train_adapter("handcook_adapter")
    print(f'Trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    return model

### Prefix tuning
def prefix_finetuning(model):
    config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                num_virtual_tokens=20,
                                encoder_hidden_size=512,
                                prefix_projection=True)
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model



'''
Loading model and tokenizer
'''
def main():
    TinyLlama = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(TinyLlama)
    model = AutoModelForCausalLM.from_pretrained(TinyLlama, load_in_4bit=True, torch_dtype=torch.float16)

