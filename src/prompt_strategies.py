def generate_affordance_prompt(shape_class, affordance_type, strategy="basic"):
    """
    Generate different types of prompts based on the strategy.
    
    Strategies:
    - basic: Simple highlighting prompt
    - action: Emphasizes the action possibility
    - affordance_specific: Adds context specific to the affordance
    """
    prompts = {
        "basic": f"A 3D render of a gray {shape_class} with highlighted {affordance_type} regions",
        "action": f"A 3D render indicating the parts of the gray {shape_class} that can be used to {affordance_type}",
        "affordance_specific": {
            "openable": f"A 3D render of a gray {shape_class} with highlighted hinge regions and handle areas that enable opening movement",
            "pushable": f"A 3D render of a gray {shape_class} with highlighted flat surface regions designed for pushing",
            "pull": f"A 3D render of a gray {shape_class} with highlighted regions showing where you can grab and pull: the handles, grip spots, and edges you can hold",
            "push": f"A 3D render of a gray {shape_class} showing areas suitable for pushing",
            "support": f"A 3D visualization of a gray {shape_class} showing parts that provide structural support",
            "pourable": f"A 3D render of a gray {shape_class} showing regions designed for pouring",
            "sittable": f"A 3D visualization of a gray {shape_class} indicating areas suitable for sitting",
            "cut": f"A 3D render of a gray {shape_class} with highlighted regions showing the sharp blade edge and cutting tip, emphasizing the main cutting surface and pointed end",
            "stab": f"A 3D render of a gray {shape_class} emphasizing parts designed for stabbing",
            "press": f"A 3D render of a gray {shape_class} showing areas designed for pressing",
            "wear": f"A 3D visualization of a gray {shape_class} showing parts designed to be worn",
            "listen": f"A 3D render of a gray {shape_class} emphasizing areas designed for listening interaction",
            "grasp": f"A 3D render of a gray {shape_class} highlighting all graspable elements including handles, straps, grip points, and reinforced edges designed for secure hand-holding and carrying",
            "contain": f"A 3D render of a gray {shape_class} emphasizing the main storage compartment, internal volume, and any additional pockets or compartments designed to hold and organize items",
            "lift": f"A 3D render of a gray {shape_class} highlighting load-bearing straps, handles, reinforced bottom, and key structural points designed for lifting when the bag is filled",
        }.get(affordance_type, f"A 3D render of a gray {shape_class} with highlighted {affordance_type} regions"),
        "utility": f"A 3D render of a gray {shape_class}, emphasizing the practical use of its {affordance_type} features",
    }

    return prompts.get(strategy, prompts["affordance_specific"])
