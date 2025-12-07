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
            "output": "- CS Department Advising: http://www.fullerton.edu/ecs/cs/resources/advisement.php – Center for Academic Support in ECS (CASECS): http://www.fullerton.edu/ecs/resources/casecs.php – Student Success Center: http://www.fullerton.edu/ecs/resources/StudentSuccessCenter.php – Academic Advisement Center (GE advising): https://www.fullerton.edu/aac/"
        },
        {
            "instruction": "How many units of Upper Division CS electives are required to complete and what coursees are available?",
            "input": "",
            "output": "At least 12.0 units of Upper Division CS electives are required. You may need to take additional electives if you are short on units due to the Placement Examination, transfer, or other circumstances. The following courses may count as Upper Division CS Electives: - CPSC 301 - Programming Lab Practicum (2 units) - CPSC 349 - Web Front-End Engineering (3 units) - CPSC 375 - Introduction to Data Science and Big Data (3 units) - CPSC 386 - Introduction to Game Design and Production (3 units) - CPSC 411 - Mobile Device Application Programming (3 units) - CPSC 411A - Mobile Device Application Programming for Android (3 units) - CPSC 431 - Database and Applications (3 units) - CPSC 439 - Theory of Computation (3 units) - CPSC 440 - Computer System Architecture (3 units) - CPSC 449 - Web Back-End Engineering (3 units) - CPSC 452 - Cryptography (3 units) - CPSC 454 - Cloud Computing and Security (3 units) - CPSC 455 - Web Security (3 units) - CPSC 456 - Network Security Fundamentals (3 units) - CPSC 458 - Malware Analysis (3 units) - CPSC 459 - Blockchain Technologies (3 units) - CPSC 462 - Software Design (3 units) - CPSC 463 - Software Testing (3 units) - CPSC 464 - Software Architecture (3 units) - CPSC 466 - Software Process (3 units) - CPSC 474 - Parallel and Distributed Computing (3 units) - CPSC 479 - Introduction to High Performance Computing (3 units) - CPSC 483 - Introduction to Machine Learning (3 units) - CPSC 484 - Principles of Computer Graphics (3 units) - CPSC 485 - Computational Bioinformatics (3 units) - CPSC 486 - Game Programming (3 units) - CPSC 489 - Game Development Project (3 units) - CPSC 499 - Independent Study (3 units)"
        },
    ]

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
        "waitlist": "How do I petition for a closed class?"
    }

    for i, chunk in enumerate(chunks[20:70]):
        lowered = chunk.lower()

        #find most relevant topic
        question = None
        for topic, q_template in topic_questions.items():
            if topic in lowered:
                if "{}" in q_template:
                    #get course/program name if possible
                    lines = chunk.split('\n')
                    for line in lines:
                        if "CPSC" in line or "MATH" in line:
                            question = q_template.format(line.split()[0])
                            break
                    if not question:
                        question = q_template.format("upper-division courses")
                else:
                    question = q_template
                break
        #defualt question if there is no topic match
        if not question:
            question = f'what does the CS handbook say about the topic in section {i+1}?'

        #clean and shorten answer
        answer = chunk[:400].strip()
        if len(answer) > 400:
            answer = answer[:397] + ".."

        auto_pairs.append({
            "instruction": question,
            "input": "",
            "output": answer
        })

    return auto_pairs




'''
Loading model and tokenizer
'''
TinyLlama = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(TinyLlama)
model = AutoModelForCausalLM.from_pretrained(TinyLlama, load_in_4bit=True, torch_dtype=torch.float16)

