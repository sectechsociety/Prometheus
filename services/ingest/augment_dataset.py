"""
Model-specific dataset augmentation for Prometheus fine-tuning.

This script generates training examples that teach the model to understand
different AI model prompting conventions (ChatGPT, Claude, Gemini).
"""

import json
import random
import uuid
from datetime import datetime

# Model-specific style templates based on official documentation
MODEL_STYLES = {
    "ChatGPT": {
        "tone": ["conversational", "friendly", "example-driven"],
        "structure": [
            "with practical examples",
            "using step-by-step explanations",
            "with analogies from everyday life",
            "in a tutorial format",
            "with real-world applications",
        ],
        "constraints": [
            "suitable for beginners",
            "with code examples where relevant",
            "avoiding jargon",
            "with hands-on practice suggestions",
            "including common pitfalls to avoid",
        ],
    },
    "Claude": {
        "tone": ["structured", "precise", "analytical"],
        "structure": [
            "with clear sections using headings",
            "in XML format with appropriate tags",
            "with explicit thinking steps",
            "following a logical progression",
            "organized hierarchically",
        ],
        "constraints": [
            "with <thinking> tags for reasoning",
            "formatted with XML structure",
            "including step-by-step methodology",
            "with explicit success criteria",
            "showing your reasoning process",
        ],
    },
    "Gemini": {
        "tone": ["concise", "action-oriented", "practical"],
        "structure": [
            "in bullet points",
            "as actionable steps",
            "in a table format when appropriate",
            "with clear takeaways",
            "organized by priority",
        ],
        "constraints": [
            "brief and to-the-point",
            "with Google Workspace integration tips",
            "focusing on productivity",
            "with measurable outcomes",
            "highlighting key actions",
        ],
    },
}

# Audience variations (universal across models)
AUDIENCES = [
    "for a complete beginner",
    "for an intermediate practitioner",
    "for an expert in the field",
    "for a business executive",
    "for a technical professional",
    "for a 12-year-old",
    "for a college student",
    "for someone with no technical background",
]

# Format constraints (universal)
FORMATS = [
    "in 50 words or less",
    "in 100-150 words",
    "in detail (500+ words)",
    "in 3-5 key points",
    "as a comparison table",
    "with pros and cons",
    "with before/after examples",
]

# Context additions (domain-specific)
CONTEXTS = [
    "in the context of e-commerce",
    "for healthcare applications",
    "for financial services",
    "for education and training",
    "for marketing campaigns",
    "for software development",
    "for data analysis",
    "for creative writing",
]


def generate_model_specific_enhancement(seed: dict, target_model: str) -> str:
    """
    Generate a model-specific enhanced prompt following that model's conventions.

    This is the CORE FUNCTION that teaches your fine-tuned model to understand
    different prompting styles.
    """
    input_prompt = seed["input_prompt"]
    style_templates = MODEL_STYLES[target_model]

    # Randomly select style elements
    tone = random.choice(style_templates["tone"])
    structure = random.choice(style_templates["structure"])
    constraint = random.choice(style_templates["constraints"])

    # Build enhanced prompt following model conventions
    if target_model == "ChatGPT":
        # ChatGPT style: conversational, example-rich
        enhanced = (
            f"{input_prompt}, using a {tone} tone. "
            f"Structure your response {structure}. "
            f"Make it {constraint}."
        )

    elif target_model == "Claude":
        # Claude style: structured with XML tags
        enhanced = (
            f"<task>{input_prompt}</task>\n"
            f"<style>{tone}</style>\n"
            f"<format>{structure}</format>\n"
            f"<requirements>{constraint}</requirements>"
        )

    elif target_model == "Gemini":
        # Gemini style: concise and actionable
        enhanced = f"{input_prompt}. Be {tone}, present {structure}, and ensure it's {constraint}."

    return enhanced


def augment_seed(seed: dict, num_variations: int = 20) -> list[dict]:
    """
    Generate multiple variations from one seed prompt.

    Variations include:
    1. Model-specific style (40% of variations) - PRIMARY FOCUS
    2. Audience targeting (30%)
    3. Format constraints (20%)
    4. Context additions (10%)
    """
    variations = []
    target_model = seed["target_model"]
    base_chunk_id = seed.get("chunk_id", str(uuid.uuid4())[:8])

    # Distribution of augmentation strategies
    style_count = int(num_variations * 0.4)
    audience_count = int(num_variations * 0.3)
    format_count = int(num_variations * 0.2)
    context_count = num_variations - (style_count + audience_count + format_count)

    # 1. Model-specific style variations (HIGHEST PRIORITY)
    for i in range(style_count):
        enhanced = generate_model_specific_enhancement(seed, target_model)
        variations.append(
            {
                "input_prompt": seed["input_prompt"],
                "enhanced_prompt": enhanced,
                "target_model": target_model,
                "source": seed.get("source", "seed_augmentation"),
                "chunk_id": f"{base_chunk_id}_style{i}",
                "created_at": datetime.now().isoformat(),
                "augmentation_type": "model_style",
                "tags": seed.get("tags", []) + ["augmented", "model_specific"],
            }
        )

    # 2. Audience variations
    for i in range(audience_count):
        audience = random.choice(AUDIENCES)
        enhanced = f"{seed['input_prompt']} {audience}"
        variations.append(
            {
                "input_prompt": seed["input_prompt"],
                "enhanced_prompt": enhanced,
                "target_model": target_model,
                "source": seed.get("source", "seed_augmentation"),
                "chunk_id": f"{base_chunk_id}_aud{i}",
                "created_at": datetime.now().isoformat(),
                "augmentation_type": "audience",
                "tags": seed.get("tags", []) + ["augmented", "audience_targeted"],
            }
        )

    # 3. Format variations
    for i in range(format_count):
        fmt = random.choice(FORMATS)
        enhanced = f"{seed['input_prompt']} {fmt}"
        variations.append(
            {
                "input_prompt": seed["input_prompt"],
                "enhanced_prompt": enhanced,
                "target_model": target_model,
                "source": seed.get("source", "seed_augmentation"),
                "chunk_id": f"{base_chunk_id}_fmt{i}",
                "created_at": datetime.now().isoformat(),
                "augmentation_type": "format",
                "tags": seed.get("tags", []) + ["augmented", "format_constrained"],
            }
        )

    # 4. Context variations
    for i in range(context_count):
        context = random.choice(CONTEXTS)
        enhanced = f"{seed['input_prompt']} {context}"
        variations.append(
            {
                "input_prompt": seed["input_prompt"],
                "enhanced_prompt": enhanced,
                "target_model": target_model,
                "source": seed.get("source", "seed_augmentation"),
                "chunk_id": f"{base_chunk_id}_ctx{i}",
                "created_at": datetime.now().isoformat(),
                "augmentation_type": "context",
                "tags": seed.get("tags", []) + ["augmented", "context_specific"],
            }
        )

    return variations


def augment_dataset(input_path: str, output_path: str, variations_per_seed: int = 20):
    """
    Main augmentation pipeline with model-specific approach.
    """
    print("🚀 Starting model-specific augmentation pipeline\n")

    with open(input_path) as f:
        seeds = [json.loads(line) for line in f]

    print(f"📂 Loaded {len(seeds)} seed prompts")

    # Count seeds per model
    model_counts = {}
    for seed in seeds:
        model = seed.get("target_model", "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1

    print(f"📊 Seeds per model: {model_counts}")
    print(f"🔢 Generating {variations_per_seed} variations per seed...\n")

    augmented = []
    for i, seed in enumerate(seeds, 1):
        if i % 10 == 0:
            print(f"   Processing seed {i}/{len(seeds)}...")
        variations = augment_seed(seed, variations_per_seed)
        augmented.extend(variations)

    with open(output_path, "w") as f:
        for item in augmented:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Final statistics
    final_counts = {}
    aug_type_counts = {}
    for item in augmented:
        model = item.get("target_model", "unknown")
        aug_type = item.get("augmentation_type", "unknown")
        final_counts[model] = final_counts.get(model, 0) + 1
        aug_type_counts[aug_type] = aug_type_counts.get(aug_type, 0) + 1

    print(f"\n✅ Generated {len(augmented)} augmented examples")
    print(f"📊 Final distribution by model: {final_counts}")
    print(f"📊 Distribution by augmentation type: {aug_type_counts}")
    print(f"💾 Saved to {output_path}")

    # Calculate percentages
    total = len(augmented)
    print("\n📈 Augmentation Strategy Breakdown:")
    for aug_type, count in sorted(aug_type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100
        print(f"   {aug_type}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate model-specific training examples for Prometheus"
    )
    parser.add_argument("--input", required=True, help="Path to seed prompts JSONL")
    parser.add_argument("--output", required=True, help="Path to output training dataset")
    parser.add_argument(
        "--variations-per-seed",
        type=int,
        default=20,
        help="Number of variations per seed (default: 20)",
    )
    args = parser.parse_args()

    augment_dataset(args.input, args.output, args.variations_per_seed)
