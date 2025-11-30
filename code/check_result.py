import json
from typing import Dict, Any, Tuple, List
from utils import load_json

def validate_plan_json(plan_json: Any) -> Tuple[bool, str]:
    """
    Validates the structure and internal consistency of the Plan Agent's output JSON.
    
    Checks include:
    1. Presence and type of all required top-level keys.
    2. Presence and type of all required keys within each evaluation dimension.
    3. Alignment between dimension dependencies and the evaluation graph edges.
    
    Returns:
        A tuple (is_valid, message), where is_valid is True or False.
    """
    errors: List[str] = []
    total_weight: float = 0.0
    
    # Handle JSON string input (User Request)
    if isinstance(plan_json, str):
        try:
            plan_json = json.loads(plan_json)
        except json.JSONDecodeError as e:
            return False, f"JSON parsing failed: Input is a string but not valid JSON. Error: {e}"

    # 1. Top-Level Structure Checks
    if not isinstance(plan_json, dict):
        return False, f"Plan result must be a dictionary after processing, but found {type(plan_json).__name__}."

    required_keys = {
        "task_type": str, 
        "evaluation_dimensions": list, 
        "evaluation_graph": dict
    }

    for key, expected_type in required_keys.items():
        if key not in plan_json:
            errors.append(f"Missing required key: '{key}'.")
        elif not isinstance(plan_json[key], expected_type):
            errors.append(f"Key '{key}' must be of type {expected_type.__name__}, but found {type(plan_json[key]).__name__}.")

    if errors:
        return False, "Top-Level Structure Errors: \n{}".format('\n'.join(errors))

    # Prepare for cross-field consistency checks
    dims = plan_json["evaluation_dimensions"]
    graph = plan_json["evaluation_graph"]
    dimension_names = {d.get("name") for d in dims if isinstance(d, dict) and "name" in d}
    
    # 2. Dimension-Level and Nested Structure Checks
    dimension_required_keys = {
        "name": str, "definition": str, "rationale": str, 
        "scoring_scale": str, "high_score_indicator": str, 
        "low_score_indicator": str, "dependencies": list, "assigned_agent": dict
    }
    
    agent_required_keys = {
        "role_name": str, "role_description": str, "evaluation_task": str, 
        "evaluation_steps": list
    }
    
    for i, dim in enumerate(dims):
        if not isinstance(dim, dict):
            errors.append(f"Dimension at index {i} must be a dictionary.")
            continue
            
        dim_name = dim.get("name", f"Dimension_{i}")
        
        for key, expected_type in dimension_required_keys.items():
            if key not in dim:
                errors.append(f"Dimension '{dim_name}' is missing key '{key}'.")
            elif not isinstance(dim[key], expected_type):
                errors.append(f"Key '{key}' in dimension '{dim_name}' must be {expected_type.__name__}.")

        # --- Weight Validation ---
        if "weight" not in dim:
            errors.append(f"Dimension '{dim_name}' is missing required key 'weight'.")
        else:
            weight_value = dim["weight"]
            try:
                # 1. Must be convertible to float
                weight_float = float(weight_value)
                
                # 2. Must be between 0 and 1
                if not (0.0 <= weight_float <= 1.0):
                    errors.append(f"Weight for dimension '{dim_name}' must be between 0 and 1, but found {weight_float}.")
                
                # Accumulate for total sum check
                total_weight += weight_float
                
            except (ValueError, TypeError):
                errors.append(f"Weight for dimension '{dim_name}' is not a valid number (string or float), found '{weight_value}'.")
        # --- End Weight Validation ---
        
        # Check assigned_agent structure
        agent = dim.get("assigned_agent", {})
        for key, expected_type in agent_required_keys.items():
            if key not in agent:
                errors.append(f"Agent for '{dim_name}' is missing key '{key}'.")
            elif not isinstance(agent[key], expected_type):
                errors.append(f"Key '{key}' in agent for '{dim_name}' must be {expected_type.__name__}.")

    # 3. Evaluation Graph Consistency Checks
    # Check graph keys
    if "nodes" not in graph or not isinstance(graph["nodes"], list):
        errors.append("evaluation_graph is missing 'nodes' or 'nodes' is not a list.")
    if "edges" not in graph or not isinstance(graph["edges"], list):
        errors.append("evaluation_graph is missing 'edges' or 'edges' is not a list.")
        
    if "nodes" in graph and "edges" in graph:
        # Check nodes alignment
        graph_nodes = set(graph["nodes"])
        if graph_nodes != dimension_names:
            missing_in_nodes = dimension_names - graph_nodes
            extra_in_nodes = graph_nodes - dimension_names
            if missing_in_nodes:
                errors.append(f"Dimensions missing from graph['nodes']: {missing_in_nodes}.")
            if extra_in_nodes:
                errors.append(f"Extra nodes in graph['nodes'] not defined as dimensions: {extra_in_nodes}.")

        # Dependency and Edge Alignment
        required_dependencies_by_dim: Dict[str, set] = {name: set() for name in dimension_names}
        
        for edge_index, edge in enumerate(graph["edges"]):
            if not isinstance(edge, dict) or "from" not in edge or "to" not in edge:
                errors.append(f"Edge at index {edge_index} is malformed.")
                continue
                
            source, target = edge["from"], edge["to"]
            
            if source not in dimension_names:
                errors.append(f"Edge source '{source}' is not a defined dimension name.")
            if target not in dimension_names:
                errors.append(f"Edge target '{target}' is not a defined dimension name.")
                
            if target in required_dependencies_by_dim:
                required_dependencies_by_dim[target].add(source)

        # Compare dependencies list in dimensions against required dependencies from graph edges
        for dim in dims:
            dim_name = dim.get("name")
            if dim_name in required_dependencies_by_dim:
                declared_deps = set(dim.get("dependencies", []))
                required_deps = required_dependencies_by_dim[dim_name]
                
                # Every declared dependency must be reflected as an incoming edge
                if not declared_deps.issubset(required_deps):
                    extra_deps = declared_deps - required_deps
                    errors.append(f"Dimension '{dim_name}' declares dependencies {extra_deps}, but no corresponding incoming edge was found in evaluation_graph.")
                
                # Every incoming edge must be reflected as a declared dependency (Alignment)
                if not required_deps.issubset(declared_deps):
                    missing_deps = required_deps - declared_deps
                    errors.append(f"Dimension '{dim_name}' is missing declared dependencies {missing_deps} required by incoming edges in evaluation_graph.")
                    
                # Ensure all dependency names are valid dimension names
                invalid_deps = declared_deps - dimension_names
                if invalid_deps:
                     errors.append(f"Dimension '{dim_name}' declares invalid dependency names: {invalid_deps}.")
    
    if errors:
        return False, "Validation Failed. Errors found: \n{}".format('\n'.join(errors))
    
    return True, "Plan JSON structure and internal consistency validated successfully."


if __name__ == "__main__":

    # plan_json_valid = {
    #     "task_type": "writing",
    #     "evaluation_dimensions": [
    #     {
    #         "name": "Fluency and Coherence",
    #         "definition": "Compare writing fluency, grammar, and logical flow of the two summaries.",
    #         "rationale": "A fluent and coherent summary improves readability and comprehension.",
    #         "weight": 0.2,
    #         "scoring_scale": "binary preference",
    #         "high_score_indicator": "The preferred summary is smoother, grammatically correct, and logically organized.",
    #         "low_score_indicator": "The disfavored summary contains awkward phrasing or unclear structure.",
    #         "dependencies": [],
    #         "assigned_agent": {
    #         "role_name": "Language Judge",
    #         "role_description": "Evaluate which summary reads more fluently and coherently.",
    #         "evaluation_task": "Compare grammatical correctness, sentence flow, and logical structure between the two summaries.",
    #         "evaluation_steps": [
    #             "Step 1: Read both summaries carefully.",
    #             "Step 2: Identify grammatical or stylistic errors in each.",
    #             "Step 3: Evaluate clarity and logical transitions.",
    #             "Step 4: Choose which summary has better fluency, or mark as Tie.",
    #             "Step 5: Provide a short rationale."
    #         ],
    #         "expected_output": {"result":"[Score 1-5]", "confidence":"0-1", "rationale":"[Reasoning]", "evidence":[]}
    #         }
    #     },
    #     {
    #         "name": "Content Coverage",
    #         "definition": "Compare which summary better covers the key information in the source context, including both benefits and concerns of AI across industries.",
    #         "rationale": "A high-quality summary should fully represent essential ideas such as AI's applications, benefits, and risks.",
    #         "weight": 0.4,
    #         "scoring_scale": "binary preference",               
    #         "high_score_indicator": "The preferred summary includes more key points with balanced detail and completeness.",
    #         "low_score_indicator": "The disfavored summary omits or distorts key points from the source context.",
    #         "dependencies": ["Factual Consistency"],
    #         "assigned_agent": {
    #         "role_name": "Content Judge",
    #         "role_description": "Compare the two summaries based on how well they capture the essential ideas of the source text.",
    #         "evaluation_task": "Determine which summary provides more comprehensive and balanced coverage of the context.",
    #         "evaluation_steps": [
    #             "Step 1: Identify key ideas in the context (industries affected, benefits, and concerns).",
    #             "Step 2: Check which of the two summaries includes more of these key points with proper emphasis.",
    #             "Step 3: Evaluate balance between benefits and concerns.",
    #             "Step 4: Decide which model's output better covers the content, or mark as Tie if equal.",
    #             "Step 5: Provide a short textual rationale."
    #         ],
    #         "expected_output": {"result":"[Score 1-5]", "confidence":"0-1", "rationale":"[Reasoning]", "evidence":[]}
    #         }
    #     },
    #     {
    #         "name": "Factual Consistency",
    #         "definition": "Compare factual alignment of the two summaries against the source text.",
    #         "rationale": "An accurate summary should not introduce errors or alter the meaning of the original content.",
    #         "weight": 0.4,
    #         "scoring_scale": "binary preference",
    #         "high_score_indicator": "The preferred summary accurately reflects facts and relationships from the context.",
    #         "low_score_indicator": "The disfavored summary contains factual inaccuracies or unsupported claims.",
    #         "dependencies": ["Content Coverage"],
    #         "assigned_agent": {
    #         "role_name": "Factual Judge",
    #         "role_description": "Check which summary remains more faithful to the factual content of the source context.",
    #         "evaluation_task": "Compare factual correctness between the two summaries and determine which is more accurate.",
    #         "evaluation_steps": [
    #             "Step 1: Review key points and coverage results from the Content Judge.",
    #             "Step 2: Cross-check statements in both summaries with the source context.",
    #             "Step 3: Identify any factual errors, distortions, or hallucinations.",
    #             "Step 4: Decide which summary is more factually consistent, or mark as Tie.",
    #             "Step 5: Write a concise explanation citing evidence."
    #         ],
    #         "expected_output": {"result":"[Score 1-5]", "confidence":"0-1", "rationale":"[Reasoning]", "evidence":[]}
    #         }
    #     }
    #     ],
    #     "evaluation_graph": {
    #     "nodes": [
    #         "Fluency and Coherence",
    #         "Content Coverage",
    #         "Factual Consistency"
    #     ],
    #     "edges": [
    #         {"from": "Factual Consistency", "to": "Content Coverage"},
    #         {"from": "Content Coverage", "to": "Factual Consistency"}
    #     ]
    #     }
    # }

    # # Mismatched dependencies and graph edges
    # plan_json_invalid_deps = {
    #     "task_type": "writing",
    #     "evaluation_dimensions": [
    #     {
    #         "name": "A",
    #         "definition": "...", "rationale": "...", "weight": 0.5, "scoring_scale": "binary preference",
    #         "high_score_indicator": "...", "low_score_indicator": "...",
    #         # Declares dependency on C, but no edge from C to A exists in the graph.
    #         "dependencies": ["B", "C"], 
    #         "assigned_agent": {
    #         "role_name": "Judge A", "role_description": "...", "evaluation_task": "...", 
    #         "evaluation_steps": ["..."], "expected_output": {"result":"[Score 1-5]", "confidence":"0-1", "rationale":"[Reasoning]", "evidence":[]}
    #         }
    #     },
    #     {
    #         "name": "B",
    #         "definition": "...", "rationale": "...", "weight": 0.5, "scoring_scale": "binary preference",               
    #         "high_score_indicator": "...", "low_score_indicator": "...",
    #         # Should depend on A (due to incoming edge), but dependency is missing.
    #         "dependencies": [], 
    #         "assigned_agent": {
    #         "role_name": "Judge B", "role_description": "...", "evaluation_task": "...", 
    #         "evaluation_steps": ["..."], "expected_output": {"result":"[Score 1-5]", "confidence":"0-1", "rationale":"[Reasoning]", "evidence":[]}
    #         }
    #     }
    #     ],
    #     "evaluation_graph": {
    #     "nodes": ["A", "B"],
    #     "edges": [
    #         # Edge B -> A (Means A should depend on B)
    #         {"from": "B", "to": "A"}, 
    #         # Edge A -> B (Means B should depend on A, but B's dependencies list is empty)
    #         {"from": "A", "to": "B"} 
    #     ]
    #     }
    # }

    # # Check plan jsons
    # is_valid_1, message_1 = validate_plan_json(plan_json_valid)
    # print("\n--- Validation Test 1: User's Example (Expected: Valid) ---")
    # print(f"Valid: {is_valid_1}, Message: {message_1}")

    # is_valid_2, message_2 = validate_plan_json(plan_json_invalid_deps)
    # print("\n--- Validation Test 2: Invalid Dependencies/Edges (Expected: Invalid) ---")
    # print(f"Valid: {is_valid_2}, Message: {message_2}")


    plan_json = load_json('../../result/testcase/plan-writing-zeroshot-llama3.json')
    is_valid, message = validate_plan_json(plan_json)
    print(f"Valid: {is_valid}, Message: {message}")