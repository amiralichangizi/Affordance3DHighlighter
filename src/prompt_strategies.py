def generate_affordance_prompt(shape_class, affordance_type, strategy="basic"):
    """
    Generate different types of prompts based on the strategy.
    
    Strategies:
    - basic: Simple highlighting prompt
    - functional: Focuses on human interaction
    - descriptive: More detailed description
    - action: Emphasizes the action possibility
    - interactive: Focuses on interaction regions
    """
    prompts = {
        "basic": f"A 3D render of a gray {shape_class} with highlighted {affordance_type} region",
        "functional": f"A 3D render of a gray {shape_class} showing where a human would {affordance_type} it",
        "descriptive": f"A 3D render of a gray {shape_class} with the {affordance_type}able parts highlighted in yellow",
        "action": f"A 3D render showing which parts of the gray {shape_class} can be used to {affordance_type}",
        "interactive": f"A 3D visualization of a gray {shape_class} highlighting regions for {affordance_type} interaction"
    }
    
    return prompts.get(strategy, prompts["basic"])
