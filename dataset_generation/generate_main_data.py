"""
SPHERE Sentence Generation & Annotation Engine
===============================================
Generates annotated scientific sentences for the SPHERE dataset using the
Google Gemini API. Requires a GOOGLE_API_KEY in a .env file and the
knowledge graph from generate_concepts.py.

Dependencies: google-generativeai, python-dotenv, tqdm
"""
import google.generativeai as genai
import json
import time
import random
from tqdm import tqdm
import os
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("FATAL: GOOGLE_API_KEY not found. Please create a .env file and set your key.")

MODEL_NAME = "gemini-2.0-flash-lite"

generation_config = { "temperature": 1.0, "max_output_tokens": 8192, "response_mime_type": "application/json" }
safety_settings = [ {"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"] ]
REQUEST_DELAY_SECONDS = 4.5

# --- File Paths for Computer Science ---
KB_FILE = "final_cs_knowledge_graph_4_levels.json"
SENTENCES_FILE = "annotated_sentences_final_2.jsonl"
API_TRACKER_FILE = "api_call_tracker_2.json"
NUM_SENTENCES_TARGET = 250000
SENTENCE_GENERATION_BATCH_SIZE = 30
ANNOTATION_BATCH_SIZE = 30 # smaller batch size for annotation reliability
NUM_WORKERS = 30 

# Thread-safe globals
file_lock = threading.Lock()
api_count_lock = threading.Lock()
existing_texts_lock = threading.Lock()
api_call_count = 0
existing_texts = set()
total_sentences_generated = 0

# ==============================================================================
# SECTION 2: THREAD-SAFE API CALLER & STATE MANAGEMENT
# ==============================================================================
def load_and_increment_api_count():
    global api_call_count
    with api_count_lock:
        if os.path.exists(API_TRACKER_FILE):
            with open(API_TRACKER_FILE, 'r') as f:
                api_call_count = json.load(f).get("total_calls", 0)
        api_call_count += 1
        with open(API_TRACKER_FILE, 'w') as f:
            json.dump({"total_calls": api_call_count}, f)
        return api_call_count

def make_api_call(prompt_text, worker_id, task_type):
    current_count = load_and_increment_api_count()
    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME, generation_config=generation_config, safety_settings=safety_settings)
        response = model.generate_content(prompt_text)
        
        if not response.parts:
            print(f"[Worker {worker_id}] {task_type} - Model returned empty response. Skipping batch.")
            return None

        clean_text = response.text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        
        return json.loads(clean_text)

    except json.JSONDecodeError as e:
        print(f"[Worker {worker_id}] {task_type} - JSON Decode Error: {e}. The model's output was not valid JSON.")
        return None
    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str or "quota" in error_str:
            print(f"[Worker {worker_id}] {task_type} - RATE LIMIT/QUOTA EXCEEDED. Stopping worker.")
            return "QUOTA_EXHAUSTED"
        else:
            if "invalid operation" in error_str and "valid part" in error_str:
                print(f"[Worker {worker_id}] {task_type} - API Error: Empty response due to safety/tokens. Skipping.")
            else:
                print(f"[Worker {worker_id}] {task_type} - Unexpected error: {e}")
            return None
    finally:
        time.sleep(REQUEST_DELAY_SECONDS)

# ==============================================================================
# SECTION 3: THREAD-SAFE FILE OPERATIONS
# ==============================================================================
def save_sentences_thread_safe(sentences_to_save, worker_id):
    global total_sentences_generated
    saved_count = 0
    
    with file_lock:
        with open(SENTENCES_FILE, 'a') as f:
            for sentence_obj in sentences_to_save:
                if isinstance(sentence_obj, dict) and "text" in sentence_obj:
                    with existing_texts_lock:
                        if sentence_obj['text'] not in existing_texts:
                            sentence_obj['sentence_id'] = total_sentences_generated + saved_count + 1
                            f.write(json.dumps(sentence_obj) + '\n')
                            existing_texts.add(sentence_obj['text'])
                            saved_count += 1
            total_sentences_generated += saved_count
    
    return saved_count

def is_text_duplicate(text):
    with existing_texts_lock:
        return text in existing_texts

# ==============================================================================
# SECTION 4: PROMPT ENGINEERING (MODIFIED FOR COMPUTER SCIENCE)
# ==============================================================================
def create_sentence_generation_prompt(seed_concepts, long_sentence_mode=False, dependency_focus_mode=False):
    concept_names = [c['name'] for c in seed_concepts]
    concept_str = ", ".join(f'"{c}"' for c in concept_names)
    
    dependency_instruction = ""
    if dependency_focus_mode:
        dependency_instruction = """
    5.  **RELATIONSHIP FOCUS**: A significant portion of your sentences MUST describe a dependency. Show that one concept is foundational to, a prerequisite for, or derived from another. Use phrases like 'is built upon', 'relies on', 'is an implementation of', 'requires the use of', etc."""

    if long_sentence_mode:
        length_instruction = "between 20 and 40 words, focusing on more detailed interactions"
    else:
        length_instruction = "between 10 and 30 words"
    
    return f"""
    You are an expert technical writer for academic research papers in **computer science**.
    Your task is to generate exactly {SENTENCE_GENERATION_BATCH_SIZE} distinct, complex sentences.
    
    CRITICAL INSTRUCTIONS:
    1.  Style: Write in a formal, dense, scientific style appropriate for CS journals.
    2.  Complexity: Each sentence must naturally incorporate and show a relationship between 2 or 3 of the following concepts: [{concept_str}]
    3.  Length: Keep sentences {length_instruction}.
    4.  Output ONLY a valid JSON list of strings.
    {dependency_instruction}
    """

def create_batch_annotation_prompt(sentences_to_annotate, relevant_concepts):
    concept_json_str = json.dumps(relevant_concepts, indent=2)
    sentences_str = "\n".join([f"- \"{s}\"" for s in sentences_to_annotate])
    return f"""
    You are a precise NER and Relation Extraction system specializing in scientific texts in **computer science**. Your task is to annotate EACH sentence in the provided list.
    
    The relevant concepts for this batch (with their permanent IDs and types) are:
    {concept_json_str}
    
    Sentences to Annotate:
    {sentences_str}
    
    CRITICAL INSTRUCTIONS:
    1.  Your output MUST be a valid JSON list where each item is an object representing one sentence.
    2.  For each entity you find, you MUST include its permanent "id", "name", "span", AND its concept "type" from the provided list.
    3.  For each relation, use the permanent IDs for the "source" and "target".
    4.  **Pay close attention to the `dependent-on` relation.** This is a foundational link where the source concept requires the target concept to be understood or to function (e.g., 'Backpropagation' is dependent-on the 'Chain Rule'). Prioritize identifying this where appropriate.

    Example Output Format:
    [
      {{
        "text": "The efficiency of a B-Tree is dependent-on its balanced structure, which minimizes disk I/O operations.",
        "entities": [
          {{"id": 789, "name": "B-Tree", "span": [23, 29], "type": "Data Structure"}},
          {{"id": 101, "name": "disk I/O operations", "span": [88, 107], "type": "Concept"}}
        ],
        "relations": [
          {{"source": 789, "target": 101, "type": "related-to"}}
        ]
      }}
    ]
    """

# ==============================================================================
# SECTION 5: WORKER FUNCTION
# ==============================================================================
def worker_function(worker_id, knowledge_base, target_sentences_per_worker, progress_bar):
    """Each worker generates and annotates sentences independently"""
    sentences_generated_by_worker = 0
    
    while sentences_generated_by_worker < target_sentences_per_worker:
        with existing_texts_lock:
            if total_sentences_generated >= NUM_SENTENCES_TARGET:
                break
        
        # GENERATION PHASE
        seed_concepts = random.sample(knowledge_base, min(20, len(knowledge_base)))
        dependency_focus_mode = random.random() < 0.4
        long_sentence_mode = random.choice([True, False])
        
        sentence_gen_prompt = create_sentence_generation_prompt(seed_concepts, long_sentence_mode, dependency_focus_mode)
        generated_texts = make_api_call(sentence_gen_prompt, worker_id, "Generation")
        
        if generated_texts == "QUOTA_EXHAUSTED": break
        if not generated_texts or not isinstance(generated_texts, list): continue
        
        unique_new_texts = [text for text in generated_texts if isinstance(text, str) and not is_text_duplicate(text)]
        if not unique_new_texts: continue
        
        # ANNOTATION PHASE (NOW BATCHED)
        for i in range(0, len(unique_new_texts), ANNOTATION_BATCH_SIZE):
            batch_to_annotate = unique_new_texts[i:i + ANNOTATION_BATCH_SIZE]
            
            annotation_prompt = create_batch_annotation_prompt(batch_to_annotate, seed_concepts)
            annotated_data = make_api_call(annotation_prompt, worker_id, "Annotation")
            
            if annotated_data == "QUOTA_EXHAUSTED":
                sentences_generated_by_worker = target_sentences_per_worker 
                break 
            if not annotated_data or not isinstance(annotated_data, list): continue
            
            # SAVING PHASE
            saved_count = save_sentences_thread_safe(annotated_data, worker_id)
            sentences_generated_by_worker += saved_count
            progress_bar.update(saved_count)
        
        time.sleep(1)


# ==============================================================================
# SECTION 6: MAIN ORCHESTRATION
# ==============================================================================
def load_knowledge_base():
    if not os.path.exists(KB_FILE):
        print(f"Error: Knowledge base file '{KB_FILE}' not found.")
        return None
    with open(KB_FILE, 'r') as f:
        kb_concepts = json.load(f)
    
    # Assign a unique ID to each concept for the annotation prompt
    for i, concept in enumerate(kb_concepts):
        concept['id'] = i + 1
        
    return kb_concepts

def load_existing_sentences():
    global existing_texts, total_sentences_generated
    if not os.path.exists(SENTENCES_FILE): return
    
    with open(SENTENCES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'text' in data:
                    existing_texts.add(data['text'])
            except json.JSONDecodeError: continue
    
    total_sentences_generated = len(existing_texts)

if __name__ == "__main__":
    genai.configure(api_key=API_KEY)
    print("--- Parallel Computer Science Sentence Generation Engine ---")
    print(f"Using {NUM_WORKERS} workers")
    
    knowledge_base = load_knowledge_base()
    if not knowledge_base: exit()

    load_existing_sentences()
    
    if os.path.exists(API_TRACKER_FILE):
        with open(API_TRACKER_FILE, 'r') as f:
            api_call_count = json.load(f).get("total_calls", 0)

    print(f"Found {len(knowledge_base)} concepts.")
    print(f"Resuming from {total_sentences_generated} existing sentences.")
    print(f"Target: {NUM_SENTENCES_TARGET} total sentences")
    print(f"Total API calls made so far: {api_call_count}")
    
    remaining_sentences = NUM_SENTENCES_TARGET - total_sentences_generated
    if remaining_sentences <= 0:
        print("Target already reached!")
        exit()
    
    sentences_per_worker = remaining_sentences // NUM_WORKERS + 1
    print(f"Each worker will generate approximately {sentences_per_worker} sentences")
    
    progress_bar = tqdm(total=NUM_SENTENCES_TARGET, desc="Total Sentences", initial=total_sentences_generated)
    
    print(f"\nStarting {NUM_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(worker_function, i, knowledge_base, sentences_per_worker, progress_bar) for i in range(NUM_WORKERS)]
        
        try:
            for future in as_completed(futures):
                try: future.result()
                except Exception as e: print(f"Worker encountered an error: {e}")
        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Shutting down workers...")
            executor.shutdown(wait=False, cancel_futures=True)
    
    progress_bar.close()
    
    with api_count_lock:
        if os.path.exists(API_TRACKER_FILE):
             with open(API_TRACKER_FILE, 'r') as f:
                final_api_count = json.load(f).get("total_calls", 0)
        else:
            final_api_count = 0
            
    print(f"\n✅ Parallel generation complete!")
    print(f"Total sentences in file: {total_sentences_generated}")
    print(f"Total API calls made: {final_api_count}")
    print(f"Sentences saved to: {SENTENCES_FILE}")