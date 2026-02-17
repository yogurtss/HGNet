"""
SPHERE Knowledge Graph Concept Generation
==========================================
Generates a hierarchical CS knowledge graph (L1->L2->L3->L4) using the
Google Gemini API. Requires a GOOGLE_API_KEY in a .env file.

Dependencies: google-generativeai, python-dotenv, tqdm
"""
import google.generativeai as genai
import json
import os
import time
import random
import threading
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Model & API Settings ---
MODEL_NAME = "gemini-1.5-flash"
MAX_WORKERS = 10
REQUEST_DELAY_SECONDS = 1.0

# --- Generation Targets (10x10x10x10 structure) ---
L1_CONCEPTS_TO_GENERATE = 10  # Top-level concepts
L2_CONCEPTS_PER_L1 = 10     # Sub-concepts for each L1
L3_CONCEPTS_PER_L2 = 10     # Sub-sub-concepts for each L2
L4_CONCEPTS_PER_L3 = 10     # Granular concepts for each L3

# --- Data File Paths (for state management) ---
STATE_DIR = "cs_graph_state_4_levels" # New directory for the new structure
L1_CONCEPTS_FILE = os.path.join(STATE_DIR, "l1_concepts.json")
L2_CONCEPTS_FILE = os.path.join(STATE_DIR, "l2_concepts.json")
L3_CONCEPTS_FILE = os.path.join(STATE_DIR, "l3_concepts.json")
L4_CONCEPTS_FILE = os.path.join(STATE_DIR, "l4_concepts.json") # Added L4 file
FINAL_GRAPH_FILE = "final_cs_knowledge_graph_4_levels.json"

# --- Knowledge Graph Definitions ---
VALID_CONCEPT_TYPES = [
    "Algorithm", "Data Structure", "Programming Language", "Architecture",
    "Protocol", "Theory", "Principle", "Paradigm", "Model", "Framework",
    "Operating System", "Database", "File System", "Compiler", "Sub-field",
    "Function", "Library", "Standard", "Component" # Added more granular types
]
VALID_RELATION_TYPES = ["is-a-subconcept-of", "related-to", "dependent-on"]

# ==============================================================================
# SECTION 2: STATE & API HELPERS
# ==============================================================================
data_lock = threading.Lock()

def setup_directories():
    if not os.path.exists(STATE_DIR):
        os.makedirs(STATE_DIR)

def load_json_file(filepath, default_value):
    if not os.path.exists(filepath): return default_value
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except (json.JSONDecodeError, IOError): return default_value

def save_json_file(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)

def make_api_call(prompt):
    """Makes a single API call and handles retries for rate limiting."""
    retries = 3
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(MODEL_NAME, generation_config={"response_mime_type": "application/json"})
            response = model.generate_content(prompt)
            time.sleep(REQUEST_DELAY_SECONDS)
            text_content = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(text_content)
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str and attempt < retries - 1:
                wait_time = (2 ** attempt) * 10 + random.uniform(0, 5)
                tqdm.write(f"[!] Rate limit hit. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            elif "response has no text" in error_str and attempt < retries - 1:
                 tqdm.write(f"[!] Empty response from API. Retrying in 5s...")
                 time.sleep(5)
            else:
                tqdm.write(f"[!] Unhandled API Error: {e}. Giving up on this call.")
                return None
    return None

# ==============================================================================
# SECTION 3: PROMPT ENGINEERING
# ==============================================================================

def create_l1_prompt():
    return f"""
    You are an expert computer scientist and ontologist.
    Generate a diverse list of {L1_CONCEPTS_TO_GENERATE} high-level, foundational fields in Computer Science.
    Examples: "Artificial Intelligence", "Operating Systems", "Computer Networks", "Cryptography".
    CRITICAL: Output ONLY a valid JSON object with a single key "concepts", which is a list of strings.
    """

def create_l2_prompt(l1_parent_name):
    return f"""
    You are a computer science expert specializing in **"{l1_parent_name}"**.
    Generate {L2_CONCEPTS_PER_L1} distinct and important sub-concepts within this field.
    VALID TYPES: {VALID_CONCEPT_TYPES}
    CRITICAL: Output ONLY a valid JSON object with a key "concepts". Each concept must have "name" and "type".
    """

def create_l3_prompt(l1_grandparent_name, l2_parent_name):
    return f"""
    You are a world-class specialist in "{l1_grandparent_name}", focusing specifically on **"{l2_parent_name}"**.
    Generate {L3_CONCEPTS_PER_L2} specific concepts related to "{l2_parent_name}".
    VALID TYPES: {VALID_CONCEPT_TYPES}
    CRITICAL: Output ONLY a valid JSON object with a key "concepts". Each concept must have "name" and "type".
    """

def create_l4_prompt(l1, l2, l3, existing_concept_names):
    sample_size = min(len(existing_concept_names), 100)
    candidate_targets = random.sample(list(existing_concept_names), sample_size)
    candidate_str = ", ".join(f'"{name}"' for name in candidate_targets)

    return f"""
    You are a deep technical expert in the hierarchy of "{l1}" -> "{l2}", focusing on the minute details of **"{l3}"**.
    Generate {L4_CONCEPTS_PER_L3} extremely specific, granular concepts (e.g., specific function names, library components, protocol flags, niche algorithms) related to "{l3}".

    ## CRITICAL INSTRUCTIONS
    1.  **Generate Concepts**: Create {L4_CONCEPTS_PER_L3} unique concept objects, each with a 'name' and a 'type' from: {VALID_CONCEPT_TYPES}.
    2.  **Generate Relations**: For each new concept, create a 'relations' list.
        - **MANDATORY**: Each concept MUST have a relation ` {{"target": "{l3}", "type": "is-a-subconcept-of"}} `.
        - **OPTIONAL**: Add 1-2 `related-to` or `dependent-on` cross-relations to concepts from the candidate list below.
    3.  **AVOID DUPLICATES**: Do not generate concepts from the candidate list.

    ## CANDIDATE CONCEPTS (for cross-relations)
    [{candidate_str}]

    ## OUTPUT FORMAT (JSON ONLY)
    {{
      "concepts": [
        {{
          "name": "TCP FIN Flag",
          "type": "Component",
          "relations": [
            {{"target": "{l3}", "type": "is-a-subconcept-of"}},
            {{"target": "TCP SYN Flag", "type": "related-to"}}
          ]
        }},
        ...
      ]
    }}
    """

# ==============================================================================
# SECTION 4: PARALLEL GENERATION WORKERS
# ==============================================================================

def generate_l2_for_l1(l1_concept, master_concept_set):
    l1_name = l1_concept['name']
    prompt = create_l2_prompt(l1_name)
    response = make_api_call(prompt)
    new_concepts = []
    if response and "concepts" in response:
        for concept_data in response["concepts"]:
            name = concept_data.get("name", "").strip()
            c_type = concept_data.get("type")
            if not name or not c_type or c_type not in VALID_CONCEPT_TYPES: continue
            with data_lock:
                if name.lower() in master_concept_set: continue
                master_concept_set.add(name.lower())
            new_concepts.append({
                "name": name, "type": c_type,
                "parent_l1": l1_name, # Context for L3
                "relations": [{"target": l1_name, "type": "is-a-subconcept-of"}]
            })
    return new_concepts

def generate_l3_for_l2(l2_concept, master_concept_set):
    l2_name = l2_concept['name']
    l1_name = l2_concept['parent_l1']
    prompt = create_l3_prompt(l1_name, l2_name)
    response = make_api_call(prompt)
    new_concepts = []
    if response and "concepts" in response:
        for concept_data in response["concepts"]:
            name = concept_data.get("name", "").strip()
            c_type = concept_data.get("type")
            if not name or not c_type or c_type not in VALID_CONCEPT_TYPES: continue
            with data_lock:
                if name.lower() in master_concept_set: continue
                master_concept_set.add(name.lower())
            new_concepts.append({
                "name": name, "type": c_type,
                "parent_l1": l1_name, "parent_l2": l2_name, # Context for L4
                "relations": [{"target": l2_name, "type": "is-a-subconcept-of"}]
            })
    return new_concepts

def generate_l4_for_l3(l3_concept, master_concept_set):
    l3_name = l3_concept['name']
    l2_name = l3_concept['parent_l2']
    l1_name = l3_concept['parent_l1']
    prompt = create_l4_prompt(l1_name, l2_name, l3_name, master_concept_set)
    response = make_api_call(prompt)
    new_concepts = []
    if response and "concepts" in response:
        for concept_data in response["concepts"]:
            name = concept_data.get("name", "").strip()
            c_type = concept_data.get("type")
            relations = concept_data.get("relations", [])
            if not name or not c_type or c_type not in VALID_CONCEPT_TYPES: continue
            with data_lock:
                if name.lower() in master_concept_set: continue
                master_concept_set.add(name.lower())
            
            valid_relations = []
            has_hierarchy = any(r.get("target") == l3_name and r.get("type") == "is-a-subconcept-of" for r in relations)
            if not has_hierarchy:
                valid_relations.append({"target": l3_name, "type": "is-a-subconcept-of"})
            for rel in relations:
                if rel.get("target") and rel.get("type") in VALID_RELATION_TYPES:
                    valid_relations.append(rel)

            new_concepts.append({"name": name, "type": c_type, "field": l1_name, "relations": valid_relations})
    return new_concepts

# ==============================================================================
# SECTION 5: MAIN ORCHESTRATION SCRIPT
# ==============================================================================

if __name__ == "__main__":
    if not API_KEY: raise ValueError("FATAL: GOOGLE_API_KEY not found in .env file.")
    genai.configure(api_key=API_KEY)
    setup_directories()

    # --- Load existing state ---
    l1 = load_json_file(L1_CONCEPTS_FILE, [])
    l2 = load_json_file(L2_CONCEPTS_FILE, [])
    l3 = load_json_file(L3_CONCEPTS_FILE, [])
    l4 = load_json_file(L4_CONCEPTS_FILE, [])
    
    master_set = {c['name'].lower() for c in l1 + l2 + l3 + l4}
    print(f"✅ Loaded {len(master_set)} existing concepts from state files.")

    # --- PHASE 1: Generate L1 Concepts ---
    print("\n--- Phase 1: Generating L1 (Top-Level Fields) ---")
    if not l1:
        response = make_api_call(create_l1_prompt())
        if response and "concepts" in response:
            for name in response["concepts"]:
                if name.lower() not in master_set:
                    l1.append({"name": name, "type": "Sub-field", "relations": []})
                    master_set.add(name.lower())
            save_json_file(L1_CONCEPTS_FILE, l1)
            print(f"✅ Generated and saved {len(l1)} L1 concepts.")
        else:
            print("❌ Failed to generate L1 concepts. Exiting."); exit()
    else:
        print(f"✅ Found {len(l1)} L1 concepts. Skipping generation.")

    # --- PHASE 2: Generate L2 Concepts ---
    print("\n--- Phase 2: Generating L2 (Sub-Concepts) ---")
    processed_l1 = {c.get('parent_l1') for c in l2}
    l1_to_process = [c for c in l1 if c['name'] not in processed_l1]
    if not l1_to_process: print("✅ All L2 concepts generated. Skipping.")
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = [exe.submit(generate_l2_for_l1, concept, master_set) for concept in l1_to_process]
            for future in tqdm(as_completed(futures), total=len(l1_to_process), desc="Processing L1"):
                if new_concepts := future.result(): l2.extend(new_concepts)
        save_json_file(L2_CONCEPTS_FILE, l2)
        print(f"✅ Phase 2 complete. Total L2 concepts: {len(l2)}")

    # --- PHASE 3: Generate L3 Concepts ---
    print("\n--- Phase 3: Generating L3 (Specific Concepts) ---")
    processed_l2 = {c.get('parent_l2') for c in l3}
    l2_to_process = [c for c in l2 if c['name'] not in processed_l2]
    if not l2_to_process: print("✅ All L3 concepts generated. Skipping.")
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = [exe.submit(generate_l3_for_l2, concept, master_set) for concept in l2_to_process]
            for future in tqdm(as_completed(futures), total=len(l2_to_process), desc="Processing L2"):
                if new_concepts := future.result(): l3.extend(new_concepts)
        save_json_file(L3_CONCEPTS_FILE, l3)
        print(f"✅ Phase 3 complete. Total L3 concepts: {len(l3)}")

    # --- PHASE 4: Generate L4 Concepts ---
    print("\n--- Phase 4: Generating L4 (Granular Concepts) ---")
    processed_l3 = set()
    for concept in l4:
        for rel in concept.get('relations', []):
            if rel.get('type') == 'is-a-subconcept-of': processed_l3.add(rel.get('target'))
    l3_to_process = [c for c in l3 if c['name'] not in processed_l3]
    if not l3_to_process: print("✅ All L4 concepts generated. Skipping.")
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = [exe.submit(generate_l4_for_l3, concept, master_set) for concept in l3_to_process]
            for future in tqdm(as_completed(futures), total=len(l3_to_process), desc="Processing L3"):
                if new_concepts := future.result(): l4.extend(new_concepts)
        save_json_file(L4_CONCEPTS_FILE, l4)
        print(f"✅ Phase 4 complete. Total L4 concepts: {len(l4)}")

    # --- PHASE 5: Final Assembly ---
    print("\n--- Phase 5: Assembling Final Knowledge Graph ---")
    final_l2 = [{k: v for k, v in c.items() if k != 'parent_l1'} for c in l2]
    final_l3 = [{k: v for k, v in c.items() if k not in ['parent_l1', 'parent_l2']} for c in l3]
    
    all_concepts_final = l1 + final_l2 + final_l3 + l4
    save_json_file(FINAL_GRAPH_FILE, all_concepts_final)