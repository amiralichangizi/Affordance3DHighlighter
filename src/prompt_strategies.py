def generate_affordance_prompt(shape_class, affordance_type, strategy="basic"):
    """
    Generate different types of prompts based on the strategy.
    
    Strategies:
    - basic: Simple highlighting prompt
    - functional: Focuses on human interaction
    - descriptive: More detailed description
    - action: Emphasizes the action possibility
    - interactive: Focuses on interaction regions
    - affordance_specific: Adds context specific to the affordance
    - utility: Describes the object's practical use with the affordance
    """
    prompts = {
        "basic": f"A 3D render of a gray {shape_class} with highlighted {affordance_type} regions",
        "functional": f"A 3D render of a gray {shape_class} showing where a person might {affordance_type} it",
        "descriptive": f"A detailed 3D render of a gray {shape_class} with its {affordance_type}-enabled parts highlighted",
        "action": f"A 3D render indicating the parts of the gray {shape_class} that can be used to {affordance_type}",
        "interactive": f"A 3D visualization of a gray {shape_class} highlighting regions where {affordance_type} interaction is possible",
        "affordance_specific": {
            "push": f"A 3D render of a gray {shape_class} showing areas suitable for pushing",
            "pull": f"A 3D render of a gray {shape_class} highlighting regions for pulling or opening",
            "support": f"A 3D visualization of a gray {shape_class} showing parts that provide structural support",
            "openable": f"A 3D render of a gray {shape_class} emphasizing parts that can be opened",
            "pourable": f"A 3D render of a gray {shape_class} showing regions designed for pouring",
            "sittable": f"A 3D visualization of a gray {shape_class} indicating areas suitable for sitting",
            "cut": f"A 3D render of a gray {shape_class} highlighting regions designed for cutting",
            "stab": f"A 3D render of a gray {shape_class} emphasizing parts designed for stabbing",
            "press": f"A 3D render of a gray {shape_class} showing areas designed for pressing",
            "wear": f"A 3D visualization of a gray {shape_class} showing parts designed to be worn",
            "listen": f"A 3D render of a gray {shape_class} emphasizing areas designed for listening interaction",
            "grasp": f"A 3D render of a gray {shape_class} highlighting all graspable elements including handles, straps, grip points, and reinforced edges designed for secure hand-holding and carrying",
            "contain": f"A 3D render of a gray {shape_class} emphasizing the main storage compartment, internal volume, and any additional pockets or compartments designed to hold and organize items",
            "lift": f"A 3D render of a gray {shape_class} highlighting load-bearing straps, handles, reinforced bottom, and key structural points designed for lifting when the bag is filled",
            "openable": f"A 3D render of a gray {shape_class} emphasizing access points including zippers, clasps, flaps, drawstrings, and other closure mechanisms that allow the bag to be opened and closed",

        }.get(affordance_type, f"A 3D render of a gray {shape_class} with highlighted {affordance_type} regions"),
        "utility": f"A 3D render of a gray {shape_class}, emphasizing the practical use of its {affordance_type} features",
    }

    return prompts.get(strategy, prompts["affordance_specific"])
