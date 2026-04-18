#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import InferenceClient

# In[2]:


train_with_rewrites = False

# In[3]:


patterns = [
    # Generic patterns
    "The EVENT_A event happens before the EVENT_B event.",
    "Whenever the EVENT_A event occurs, the EVENT_B event eventually occurs.",
    "After the EVENT_A event, the EVENT_B event must not occur.",
    "The EVENT_A and EVENT_B events must refer to the same OBJECT_A.",
    "The EVENT_B event happens within TIME TIME_UNIT after the EVENT_A event.",
    # Parent-child pattern
    "Each OBJECT_A must have at least NUM associated OBJECT_B for which the EVENT_A event occurs.",
    # Multi-child pattern
    "For any OBJECT_A with at most NUM associated OBJECT_B, the EVENT_B event must occur within TIME TIME_UNIT after the EVENT_A event.",
    # SAT pattern 
    # "A constraint modeling that for any OBJECT_A, the EVENT_B event should happen fully within TIME TIME_UNIT of the EVENT_A event.",
    "For any OBJECT_A, EVENT_B must be completed within TIME TIME_UNIT after EVENT_A.",
    # Deep nested pattern
    "Ensure that any OBJECT_A has at least NUM associated OBJECT_B where the EVENT_B event has occurred at least NUM2 times.",
    # Exactly once pattern
    # "A constraint modeling that the EVENT_A event should occur exactly once for each OBJECT_A.",
    "EVENT_A must occur exactly once for each OBJECT_A.",
    # At least once pattern
    "After the EVENT_A event occurs for a given OBJECT_A, the EVENT_B event should happen at least once.",
    # CEL Aggregation pattern
    "Find all OBJECT_A whose total ATT_LABEL is OP_TEXT NUM3, where the total is calculated by summing the ATT_NAME of every associated OBJECT_B that had an EVENT_A event."
]

# In[4]:


domains = {
    # Generic domains
    "e-commerce": {
        "events": ["create order", "add item", "pay order", "ship package", "cancel order"],
        "objects": ["customers", "employees", "items", "orders", "packages", "products"]
    },
    "recruitment": {
        "events": ["submit application", "review application", "approve application", "reject application"],
        "objects": ["applications", "candidates"]
    },
    "software_dev": {
        "events": ["open issue", "submit pull request", "review pull request", "merge pull request", "close issue"],
        "objects": ["issues", "pull requests", "commits"]
    },

    # Domains from actual log files
    "distribution": {
        "events": ["Apply Discount", "Apply Payment", "Approve Order", "Approve Return", "Generate Invoice", 
                   "Generate Pick List", "Generate Shipping Label", "Handle Payment Discrepancy", 
                   "Issue Credit Note", "Issue Payment Reminder", "Pack Items", "Perform Credit Check", 
                   "Pick Items", "Process Return Request", "Receive Customer Order", "Receive Payment", 
                   "Reject Order", "Schedule Order Fulfillment", "Send Invoice to Customer", "Ship Order", 
                   "Update Inventory", "Validate Customer Order"],
        "objects": ["CreditNote", "Customer", "Dispute", "Invoice", "Order", "Payment", "Product", "Return", "Shipment"]
    },
    "procurement": {
        "events": ["Approve PO", "Approve PR", "Create PO", "Create PO (Batch)", "Create PR", "Make Payment", 
                   "Receive Goods", "Receive Invoice", "Reject PR", "Send PO to Supplier", 
                   "Three-way Match - Mismatch Found", "Three-way Match - Successful"],
        "objects": ["Goods Receipt", "Purchase Requisition", "Payment", "Invoice", "Purchase Order", "Supplier"]
    },
    "02_p2p": {
        "events": ["Approve PO", "Approve PR", "Create PO", "Create PO (Batch)", "Create PR", "Make Payment", 
                   "Receive Goods", "Receive Invoice", "Reject PR", "Send PO to Supplier", "Three-way Match - Mismatch Found", 
                   "Three-way Match - Successful"],
        "objects": ["Goods Receipt", "Invoice", "Payment", "Purchase Order", "Purchase Requisition", "Supplier"]
    },
    "hiring": {
        "events": ['Approve Job Requisition', 'Attend Orientation Training', 'Candidate Accept Offer', 'Candidate Reject Offer', 
                   'Complete Assessment', 'Complete Background Check', 'Complete HR Paperwork', 'Complete Onboarding', 
                   'Conduct Initial Interview', 'Conduct Panel Interview', 'Create Job Requisition', 'Evaluate Interview', 
                   'Initiate Background Check', 'Post Job Opening', 'Prepare Offer Letter', 'Review Assessment', 
                   'Schedule Initial Interview', 'Schedule Panel Interview', 'Screen Resume', 'Send Assessment', 
                   'Send Offer Letter', 'Send Rejection Letter', 'Start Onboarding', 'Submit Application'],
        "objects": ['Interviewer', 'HiringManager', 'Recruiter', 'Assessment', 'Candidate', 'JobRequisition']
    },
    "hospital": {
        "events": ['Appointment Scheduling', 'Assign Bed', 'Diagnostic Test Order', 'Initial Evaluation', 
                   'Insurance Verification', 'Patient Arrival', 'Perform Diagnostic Test', 'Pre-Admission Consultation', 
                   'Registration'],
        "objects": ['Patient', 'InsuranceCompany', 'Physician', 'LabTechnician', 'Nurse', 'DiagnosticEquipment', 'Room', 'Bed']
    },
    "container_logistics": {
        "events": ['Book Vehicles', 'Bring to Loading Bay', 'Collect Goods', 'Create Transport Document', 'Depart', 
                   'Drive to Terminal', 'Load Truck', 'Load to Vehicle', 'Order Empty Containers', 'Pick Up Empty Container', 
                   'Place in Stock', 'Register Customer Order', 'Reschedule Container', 'Weigh'],
        "objects": ['Container', 'Customer Order', 'Forklift', 'Handling Unit', 'Transport Document', 'Truck', 'Vehicle']
    },
    "lending": {
        "events": ['Application Rejected', 'Appointment Denied', 'Hand In Credit Appliaction', 'Loan Denied', 
                   'Make Visit to Assess Colatteral', 'Request Appointment', 'Request Co-Signer On Loan', 
                   'Set Appointment', 'Sign Loan Agreement', 'Submit File to Underwriter', 'Verify Borrowers Information', 
                   'skipped_examination'],
        "objects": ['Person']
    },
    "renting": {
        "events": ["Accept Late Payment", "Apply for Viewing Appointment", "Check Paperwork", "Evict Tenant", "Extensive Screening", 
                   "Hand In The Paperwork", "Issue Warning", "Miss Rent Payment", "Move In", "Pay Rent", "Reject Prospective Tenant", 
                   "Screen Prospective Tenant", "Set Appointment", "Sign Contract", "Tenant Cancels Appartment", "View The Property"],
        "objects": ["Person"]
    }
}

# In[5]:


from transformers import pipeline, BitsAndBytesConfig

import torch

import os
import re

# In[6]:


if train_with_rewrites:
    # Create 4-bit config to compress the ~15GB model to ~5-6GB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Initialize the local pipeline with 4-bit quantization
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"quantization_config": bnb_config},
        device_map="auto",
        token=os.getenv("HF_TOKEN")
    )

# In[7]:


def rewrite_sentences_batch(sentences, batch_size=4):
    # Determine padding token if needed for processing batches
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        
    # CRITICAL FIX: Causal LMs require left-padding for batched generation!
    pipe.tokenizer.padding_side = "left"

    messages_batch = []
    for sentence in sentences:
        prompt = f"""
Rewrite the following sentence to make it sound natural and grammatically correct.
Provide TWO variations:
1. "nl_clean_1": A slightly more formal, straight-forward rewrite.
2. "nl_clean_2": A more varied and naturally flowing rewrite.

Rules:
- Keep the exact meaning.
- Do not add or remove information.
- Keep technical terms unchanged.
- Output ONLY valid JSON containing the keys "nl_clean_1" and "nl_clean_2".
- Do not include any explanations, greetings, or backticks.

Sentence:
"{sentence}"
"""
        messages_batch.append([
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, with no markdown formatting, no backticks, and no conversational text."},
            {"role": "user", "content": prompt}
        ])
    
    # Run the pipeline in batches (Processes `batch_size` sentences concurrently)
    outputs = pipe(
        messages_batch,
        max_new_tokens=500,
        temperature=0.5,
        do_sample=True,
        batch_size=batch_size
    )
    
    results = []
    for i, out in enumerate(outputs):
        content = out[0]["generated_text"][-1]["content"]
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            clean_content = match.group(0) if match else content.strip()
            
            # Auto-heal: If the generation randomly cut off just the closing bracket, fix it
            if not clean_content.endswith("}"):
                clean_content += "\n}"
                
            data = json.loads(clean_content)
            results.append(data)
        except Exception as e:
            print(f"Failed to parse JSON for sample {i}:\nRaw Output: {content}")
            results.append({"nl_clean_1": sentences[i], "nl_clean_2": sentences[i]})
            
    return results

# In[8]:


def generate_example():
    pattern = random.choice(patterns)
    
    # Pick a domain first to ensure semantic consistency
    domain = random.choice(list(domains.keys()))
    domain_data = domains[domain]

    A = random.choice(domain_data["events"])
    # Ensure event A and B are different for better logical examples
    B = random.choice([e for e in domain_data["events"] if e != A])
    
    obj_a = random.choice(domain_data["objects"])
    
    # Fetch a second object if the domain has more than one
    obj_b = random.choice([o for o in domain_data["objects"] if o != obj_a]) if len(domain_data["objects"]) > 1 else obj_a

    # Randomly select a time unit for patterns that require it
    time_unit = random.choice(["hour(s)", "week(s)", "day(s)", "minute(s)", "second(s)", "month(s)", "year(s)"])
    
    # Calculate time scaling based on the time unit
    if time_unit == "year(s)":
        t = random.choice([1, 1.5, 2, 3])
        t_sec = int(t * 31536000) # Years to seconds
    elif time_unit == "month(s)":
        t = random.choice([1, 1.5, 2, 3, 6])
        t_sec = int(t * 2592000) # Months to seconds (approx 30 days)
    elif time_unit == "week(s)":
        t = random.choice([1, 1.5, 2, 3, 3.5, 4])
        t_sec = int(t * 604800) # Weeks to seconds
    elif time_unit == "day(s)":
        t = random.choice([1, 5, 10, 30])
        t_sec = int(t * 86400) # Days to seconds
    elif time_unit == "minute(s)":
        t = random.choice([5, 10, 30, 45])
        t_sec = int(t * 60) # Minutes to seconds
    elif time_unit == "second(s)":
        t = random.choice([10, 30, 60, 500])
        t_sec = int(t) # Seconds
    else: # hours
        t = random.choice([1, 6, 24, 72])
        t_sec = int(t * 3600) # Hours to seconds
    
    # Generate random numbers for pluralization and quantity-based patterns
    num = random.choice([1, 2, 5, 10, 50, 100])
    num2 = random.choice([1, 2, 3, 4, 5])

    # Generate random attributes for CEL scripts
    att_label = random.choice(["order_volume", "total_amount", "total_price"])
    att_name = random.choice(["price", "amount", "value"])
    op_text = random.choice(["below", "above", "exactly", "at least"])
    
    # Map operator language to CEL math operators
    op_map = {"below": "<", "above": ">", "exactly": "==", "at least": ">="}
    op_num = op_map[op_text]
    num3 = random.choice([500, 1000, 5000, 10000, 50000, 300000])

    # Replace placeholders in the pattern with the generated values
    text = pattern.replace("EVENT_A", A)\
                  .replace("EVENT_B", B)\
                  .replace("OBJECT_A", obj_a)\
                  .replace("OBJECT_B", obj_b)\
                  .replace("TIME_UNIT", time_unit)\
                  .replace("TIME", str(t))\
                  .replace("NUM2", str(num2))\
                  .replace("NUM3", str(num3))\
                  .replace("NUM", str(num))\
                  .replace("ATT_LABEL", att_label)\
                  .replace("ATT_NAME", att_name)\
                  .replace("OP_TEXT", op_text)

    # Generate reasoning for Least-to-Most prompting
    is_single_event = "exactly once" in pattern or "at least once" in pattern and "the EVENT_B event should happen" not in pattern

    if "exactly once" in pattern:
        reasoning = f"Step 1: The query involves the event '{A}' and the object '{obj_a}'. "
        reasoning += f"Step 2: The constraint specifies that '{A}' must occur exactly once for '{obj_a}'. "
        reasoning += f"Step 3: To prevent unbounded Cartesian products, we connect the logic structurally. We place '{obj_a}' in the root node, and link to a connected child node containing '{A}'. We bind them with an explicit `O2E` filter in the child node, and enforce the exactly 1 constraint via a SizeFilter in the parent."
    elif "at least once" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and object '{obj_a}'. "
        reasoning += f"Step 2: The constraint specifies that after '{A}' occurs for '{obj_a}', '{B}' must also occur at least once. "
        reasoning += f"Step 3: To prevent unbounded searches, we structure this as a connected tree. The parent node binds '{A}' and '{obj_a}' explicitly with an `O2E` filter, and links to a connected child node for '{B}'. We bind '{B}' back to the object using an `O2E` filter strictly inside this child node."
    elif "total is calculated" in pattern:
        reasoning = f"Step 1: The query involves event '{A}', objects '{obj_a}', '{obj_b}' and attributes '{att_label}', '{att_name}'. "
        reasoning += f"Step 2: The constraint limits '{obj_a}' based on the sum of '{att_name}' for all its '{obj_b}' linked to '{A}'. The sum must be {op_text} {num3}. "
        reasoning += f"Step 3: We establish a 3-node connected hierarchy. Node 0 is for '{obj_a}' with an AdvancedCEL SizeFilter mapping the sum to '{att_label}'. Node 1 is an object node for '{obj_b}' linking to Node 0. Node 2 is an event node for '{A}' connecting to Node 1 via O2E."
    elif "at least NUM2" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and objects '{obj_a}' and '{obj_b}'. "
        reasoning += f"Step 2: The constraint expects '{obj_a}' to have at least {num} '{obj_b}' where '{B}' occurs at least {num2} times after '{A}'. "
        reasoning += f"Step 3: To prevent slow queries, we construct a fully connected 3-node structure: '{obj_a}' links to '{obj_b}'+'{A}' (bound explicitly with `O2E`), which then connects to a child node for '{B}' (also bound with `O2E`). Filters are embedded directly inside these connected child nodes to safely bound the search."
    elif "completed within" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and objects '{obj_a}' and '{obj_b}' (if applicable). "
        reasoning += f"Step 2: '{B}' must be completed within {t} {time_unit} after '{A}' for any '{obj_a}'. "
        reasoning += f"Step 3: To ensure safe query bounds, we use a connected parent-child structure. The SAT constraint is placed in the parent for '{A}', linking to a connected child node for '{B}'. Inside the connected child node, we place the `O2E` bindings and the TimeBetweenEvents filter ({t_sec} seconds) so they safely constrain the search path."
    elif "at most NUM" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and objects '{obj_a}' and '{obj_b}'. "
        reasoning += f"Step 2: For any '{obj_a}' with at most {num} '{obj_b}', '{B}' must occur within {t} {time_unit} after '{A}'. "
        reasoning += f"Step 3: To avoid query explosions, we connect the nodes structurally. The root node for '{A}' and '{obj_a}' (bound via `O2E`) branches to two connected child nodes: one for '{obj_b}' (capped by SizeFilter max {num}) and one for '{B}'. Inside the '{B}' child node, we explicitly place the `O2E` and TimeBetweenEvents ({t_sec} secs) filters to constrain execution."
    elif "for which the EVENT_A" in pattern:
        reasoning = f"Step 1: The query involves event '{A}' and objects '{obj_a}' and '{obj_b}'. "
        reasoning += f"Step 2: Each '{obj_a}' must have at least {num} associated '{obj_b}' where '{A}' occurs. "
        reasoning += f"Step 3: To prevent unbounded searches, we establish a connected hierarchy. A parent node for '{obj_a}' with a SizeFilter (at least {num}) links to a connected child node containing '{A}' and '{obj_b}'. We explicitly bind them using `O2O` and `O2E` filters strictly inside this child node."
    elif "before" in pattern or "eventually" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}'. "
        reasoning += f"Step 2: '{A}' occurs before or eventually leads to '{B}'. "
        reasoning += f"Step 3: To inherently prevent isolated events from forming a Cartesian product, we map them into a single connected node with a TimeBetweenEvents filter from '{A}' to '{B}' with no maximum time."
    elif "within TIME" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}'. "
        reasoning += f"Step 2: '{B}' happens within {t} {time_unit} after '{A}'. "
        reasoning += f"Step 3: To inherently prevent isolated events from forming a Cartesian product, we map them into a single connected node with a TimeBetweenEvents filter from '{A}' to '{B}' up to {t_sec} seconds."
    elif "must not occur" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}'. "
        reasoning += f"Step 2: '{B}' must not occur after '{A}'. "
        reasoning += f"Step 3: A single node referencing a child edge inside a NOT constraint."
    elif "refer to the same" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and object '{obj_a}'. "
        reasoning += f"Step 2: '{A}' and '{B}' must refer to the same '{obj_a}'. "
        reasoning += f"Step 3: To explicitly correlate them and avoid unbounded Cartesian searches, we place '{A}', '{B}', and '{obj_a}' into a single node and add exact `O2E` link filters for both events."
    else:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and objects '{obj_a}' and '{obj_b}' (if applicable). "
        reasoning += "Step 2: Relate the identified elements. Step 3: Map to generic OCPQ single node."

    # Pattern handling: multi-node vs single-node trees
    if "exactly once" in pattern:
        # Two-node exactly once mapping (min: 1, max: 1)
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [],
                            "sizeFilters": [],
                            "constraints": [{"type": "SizeFilter", "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": 1}}],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Connects to child A
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}
        
    elif "at least once" in pattern:
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [{"type": "SizeFilter", "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": None}}],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Connects to child A
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "1": [B] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 1, "qualifier": None},
                                {"type": "TimeBetweenEvents", "from_event": 0, "to_event": 1, "min_seconds": 0, "max_seconds": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}

    elif "total is calculated" in pattern:
        # AdvancedCEL 3-node structure
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [],
                            "sizeFilters": [
                                {
                                    "type": "AdvancedCEL",
                                    "cel": f"childs.map(c, c.{att_label}).sum().int() {op_num} {num3}",
                                    "child_name": "A",
                                    "labels": [
                                        {
                                            "name": att_label,
                                            "cel": f"childs.map(c, c.{att_name}).sum().int()",
                                            "child_name": "B"
                                        }
                                    ]
                                }
                            ],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Contains child 1 (O2O)
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "1": [obj_b] },
                            "filters": [
                                {"type": "O2O", "object": 0, "other_object": 1, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [2] # Contains child 2 (O2E)
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 1, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"], [[1, 2], "B"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}

    elif "at least NUM2" in pattern:
        # Deep nested tree structure (A -> B -> C)
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [],
                            "sizeFilters": [],
                            "constraints": [{"type": "SizeFilter", "filter": {"type": "NumChilds", "child_name": "A", "min": None, "max": num}}],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Connects to child A
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": { "1": [obj_b] },
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 0, "qualifier": None},
                                {"type": "O2E", "object": 1, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [{"type": "NumChilds", "child_name": "B", "min": num2, "max": None}],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [2] # Connects to child B
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "1": [B] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 1, "event": 1, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"], [[1, 2], "B"]] # Ordered hierarchical edges
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}

    elif "completed within" in pattern:
        # Parent-child structure using the SAT constraint
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [],
                            "sizeFilters": [],
                            "constraints": [{"type": "SAT", "child_names": ["A"]}],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Connects to child A
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A], "1": [B] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 0, "qualifier": None},
                                {"type": "O2E", "object": 0, "event": 1, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [
                                {
                                    "type": "Filter",
                                    "filter": {"type": "TimeBetweenEvents", "from_event": 0, "to_event": 1, "min_seconds": 0, "max_seconds": t_sec}
                                }
                            ],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}
        
    elif "at most NUM" in pattern:
        # Multi-node structure with two distinct child branches
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [{"type": "NumChilds", "child_name": "A", "min": None, "max": num}],
                            "constraints": [{"type": "SizeFilter", "filter": {"type": "NumChilds", "child_name": "B", "min": 1, "max": None}}],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1, 2] # Connects to child A and B
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "1": [obj_b] },
                            "filters": [
                                {"type": "O2O", "object": 0, "other_object": 1, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "1": [B] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 1, "qualifier": None},
                                {"type": "TimeBetweenEvents", "from_event": 0, "to_event": 1, "min_seconds": 0, "max_seconds": t_sec}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"], [[0, 2], "B"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}
    
    elif "for which the EVENT_A" in pattern:
        # Construct the OCPQ tree representing the parent-child node logic 
        # (e.g. Customers with at least 50 orders involving payment reminder)
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [],
                            "sizeFilters": [{"type": "NumChilds", "child_name": "A", "min": num, "max": None}],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Connects to child node at index 1
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": { "1": [obj_b] },
                            "filters": [
                                {"type": "O2O", "object": 0, "other_object": 1, "qualifier": None},
                                {"type": "O2E", "object": 1, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}

    # Dynamically inject constraints based on single-node patterns
    constraints = []
    filters = []
    if "before" in pattern or "eventually" in pattern:
        constraints = [{
            "type": "Filter",
            "filter": { "type": "TimeBetweenEvents", "from_event": 0, "to_event": 1, "min_seconds": 0, "max_seconds": None }
        }]
    elif "within TIME" in pattern:
        constraints = [{
            "type": "Filter",
            "filter": { "type": "TimeBetweenEvents", "from_event": 0, "to_event": 1, "min_seconds": 0, "max_seconds": t_sec }
        }]
    elif "must not occur" in pattern:
        # Requires a child edge with a NOT constraint to be completely valid OCPQ
        constraints = [{
            "type": "NOT",
            "child_names": ["edge_1"]
        }]
    elif "refer to the same" in pattern:
        filters = [
            { "type": "O2E", "object": 0, "event": 0, "qualifier": None },
            { "type": "O2E", "object": 0, "event": 1, "qualifier": None }
        ]

    ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A], "1": [B] },
                            "newObjectVars": { "0": [obj_a] } if "refer to the same" in pattern else {},
                            "filters": filters,
                            "sizeFilters": [],
                            "constraints": constraints,
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # IDs of any connected child nodes in the tree
                    ]
                }
            ],
            "edgeNames": []
    }

    return {"nl": text, 
            "events": domain_data["events"], 
            "objects": domain_data["objects"], 
            "query": ocpq_tree, 
            "reasoning": reasoning}

# In[9]:


def generate_dataset(num_samples=500, batch_size=4, original_sentence=True, first_rewrite=True, second_rewrite=True, output_file="ocpq_dataset.json"):
    dataset = []
    seen = set()

    # 1. First, instantly generate all unique logical sentences locally
    print("Generating pure logical combinations...")
    raw_examples = []
    while len(raw_examples) < num_samples:
        ex = generate_example()
        if ex["nl"] not in seen:
            seen.add(ex["nl"])
            raw_examples.append(ex)

    # 2. Process the LLM rewrites in parallel batches
    print(f"Batch processing text inference on GPU (batch_size={batch_size})...")
    for i in range(0, len(raw_examples), batch_size):
        batch = raw_examples[i:i+batch_size]
        sentences = [ex["nl"] for ex in batch]
        
        if first_rewrite or second_rewrite:
            # Send 'batch_size' queries to the GPU at once instead of 1
            rewrites_batch = rewrite_sentences_batch(sentences, batch_size=batch_size)
        else:
            rewrites_batch = [{} for _ in sentences]
            
        for j, ex in enumerate(batch):
            updated_example = {}
            if original_sentence:
                updated_example["nl"] = ex["nl"]
                
            if first_rewrite:
                updated_example["nl_clean_1"] = rewrites_batch[j].get("nl_clean_1", ex["nl"])
            if second_rewrite:
                updated_example["nl_clean_2"] = rewrites_batch[j].get("nl_clean_2", ex["nl"])
            
            updated_example["events"] = ex["events"]
            updated_example["objects"] = ex["objects"]
            updated_example["query"] = ex["query"]
            updated_example["reasoning"] = ex["reasoning"]
            dataset.append(updated_example)
            
        print(f"Processed {len(dataset)} / {num_samples} examples", end="\r")

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print("\nDataset generation completed!")
    return dataset

# In[10]:


dataset = generate_dataset(num_samples=10000, first_rewrite=False, second_rewrite=False)

# In[11]:


dataset[:5]
