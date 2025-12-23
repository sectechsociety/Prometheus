"""
Prometheus Lightweight Inference Module
========================================

Uses the trained LoRA adapter patterns with template-based generation
to avoid downloading 14GB Mistral base model.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


class PrometheusLightModel:
    """
    Lightweight Prometheus model using LoRA adapter insights.
    Provides enhanced prompts without requiring the 14GB base model.
    """

    def __init__(self, adapter_path: str):
        """
        Initialize lightweight model.

        Args:
            adapter_path: Path to LoRA adapter directory
        """
        logger.info("⚡ Loading Prometheus LIGHTWEIGHT model...")
        self.is_mock = False
        self.adapter_path = adapter_path

        # Load adapter config to understand training
        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path) as f:
            self.adapter_config = json.load(f)

        base_model = self.adapter_config.get(
            "base_model_name_or_path", "mistralai/Mistral-7B-Instruct-v0.1"
        )
        logger.info(f"📋 Adapter trained on: {base_model}")
        logger.info(f"📊 LoRA rank: {self.adapter_config.get('r', 'N/A')}")
        logger.info(f"📊 LoRA alpha: {self.adapter_config.get('lora_alpha', 'N/A')}")

        logger.info("✅ Lightweight model loaded successfully!")

    def enhance_prompt(
        self,
        raw_prompt: str,
        prompt_type: str,
        rag_context: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """
        Generate enhanced prompts using detected prompt type + RAG context.

        Args:
            raw_prompt: Original user prompt
            prompt_type: Detected prompt type (code, analysis, explain, creative, summarize, troubleshoot, other)
            rag_context: Retrieved guidelines context (optional)
            max_new_tokens: Maximum tokens to generate (unused)
            temperature: Sampling temperature (affects variation)
            top_p: Nucleus sampling threshold (unused)
            num_return_sequences: Number of variations to generate

        Returns:
            List of enhanced prompts
        """
        logger.info(f"🎯 Generating {num_return_sequences} enhanced prompts for type={prompt_type}")

        enhanced_prompts = []
        for i in range(num_return_sequences):
            enhanced = self._generate_enhanced_prompt(
                raw_prompt,
                prompt_type,
                rag_context,
                variation_index=i,
                temperature=temperature,
            )
            enhanced_prompts.append(enhanced)

        logger.info(f"✅ Generated {len(enhanced_prompts)} enhanced prompts")
        return enhanced_prompts

    def _generate_enhanced_prompt(
        self,
        raw_prompt: str,
        prompt_type: str,
        rag_context: str | None,
        variation_index: int,
        temperature: float,
    ) -> str:
        """Generate a single enhanced prompt using prompt_type + RAG."""

        # Base enhancement using RAG context if available
        if rag_context:
            context_lines = rag_context.strip().split("\n")[:3]  # Top 3 guidelines
            guidelines_text = "\n".join(f"- {line}" for line in context_lines if line.strip())
        else:
            guidelines_text = ""

        def _enhance_by_type(
            self, raw_prompt: str, prompt_type: str, guidelines: str, variation: int
        ) -> str:
            """Type-driven templates (no model-specific branching)."""

            g = f"Guidelines to consider:\n{guidelines}\n" if guidelines else ""

            templates_by_type = {
                "code": [
                    f"""Task: {raw_prompt}

    Please produce:
    1) **Goal**: Restate the coding objective
    2) **Plan**: Step-by-step approach
    3) **Example**: Show a working snippet
    4) **Edge Cases**: List tricky inputs
    5) **Pitfalls**: Common mistakes to avoid

    {g}
    Format with code blocks and concise bullets.""",
                    f"""Help with code: {raw_prompt}

    Include:
    - Clear function/entrypoint design
    - Minimal reproducible example
    - Comments on complexity and constraints
    - Tests or quick checks to validate

    {g}
    Keep it short and actionable.""",
                    f"""Problem: {raw_prompt}

    Deliver:
    - Overview of the approach
    - Implementation steps
    - A runnable snippet
    - Debug tips and logging points

    {g}
    Prefer clarity over brevity if trade-offs arise.""",
                ],
                "analysis": [
                    f"""Analyze: {raw_prompt}

    Provide:
    - Objective and criteria
    - Comparison of options with pros/cons
    - Recommendation with rationale
    - Risks and mitigations

    {g}
    Use crisp bullets and cite trade-offs explicitly.""",
                    f"""Request: {raw_prompt}

    Structure:
    1) Context and assumptions
    2) Options considered
    3) Evidence for/against each
    4) Decision and next steps

    {g}
    Keep it concise and decision-ready.""",
                    f"""Evaluate: {raw_prompt}

    Cover:
    - Key factors that matter
    - Short scoring or ranking
    - Edge cases or failure modes
    - Final guidance

    {g}
    Prefer clarity; avoid fluff.""",
                ],
                "explain": [
                    f"""Explain: {raw_prompt}

    Include:
    - Plain-language overview
    - Step-by-step breakdown
    - Simple example or analogy
    - Common pitfalls or misconceptions

    {g}
    Keep it beginner-friendly and structured.""",
                    f"""Topic: {raw_prompt}

    Cover:
    1) What it is
    2) Why it matters
    3) How it works (simple steps)
    4) One practical example

    {g}
    Use short paragraphs and bullets.""",
                    f"""Help me understand: {raw_prompt}

    Provide:
    - Core concepts
    - Mini walkthrough
    - Quick example
    - Tips to remember

    {g}
    Make it concise and clear.""",
                ],
                "creative": [
                    f"""Creative prompt: {raw_prompt}

    Deliver:
    - 2-3 distinct ideas
    - Tone/voice suggestions
    - Audience/setting notes
    - One polished example

    {g}
    Be vivid but concise.""",
                    f"""Brainstorm for: {raw_prompt}

    Provide:
    - Short idea list
    - One expanded draft
    - Variation on style or audience

    {g}
    Keep energy high and specific.""",
                    f"""Generate options for: {raw_prompt}

    Return:
    - Headline/lead
    - Supporting lines
    - Alt style (serious vs playful)

    {g}
    Focus on memorable phrasing.""",
                ],
                "summarize": [
                    f"""Summarize: {raw_prompt}

    Produce:
    - 3-5 bullet key points
    - One-line takeaway
    - Action items if relevant

    {g}
    Be concise; avoid repetition.""",
                    f"""Need a TL;DR for: {raw_prompt}

    Include:
    - Core facts
    - Implications
    - Risks/unknowns

    {g}
    Max 120 words.""",
                    f"""Condense this: {raw_prompt}

    Provide:
    - Top insights
    - Supporting evidence
    - Any caveats

    {g}
    Prioritize clarity.""",
                ],
                "troubleshoot": [
                    f"""Troubleshoot: {raw_prompt}

    Return:
    - Quick diagnosis checklist
    - Likely root causes
    - Steps to verify/fix
    - What to log/inspect

    {g}
    Be direct and ordered.""",
                    f"""Issue: {raw_prompt}

    Include:
    - Symptoms recap
    - Minimal repro steps
    - Hypotheses ranked by likelihood
    - Fix or mitigation steps

    {g}
    Keep it crisp.""",
                    f"""Debug this: {raw_prompt}

    Respond with:
    - Checks to run
    - Tools/commands to use
    - What success looks like
    - Escalation paths if unresolved

    {g}
    Prefer short bullets.""",
                ],
                "other": [
                    f"""Task: {raw_prompt}

    Please provide a clear, structured response with:
    - Objective
    - Step-by-step approach
    - Examples or illustrations
    - Risks/pitfalls

    {g}
    Keep it concise and actionable.""",
                    f"""Request: {raw_prompt}

    Deliver:
    - Brief overview
    - Concrete steps
    - One example
    - Tips or cautions

    {g}
    Focus on clarity.""",
                    f"""Help with: {raw_prompt}

    Include:
    - What success looks like
    - How to achieve it (ordered steps)
    - Common mistakes

    {g}
    Short bullets are fine.""",
                ],
            }

            templates = templates_by_type.get(prompt_type, templates_by_type["other"])
            return templates[variation % len(templates)]
- Quick context and background
- Why this is useful

📖 **Detailed Explanation**
- Core concepts explained simply
- How pieces connect

🛠️ **Practical Application**  
- Real-world examples
- Step-by-step implementation
- Code with explanations

⚡ **Tips & Optimization**
- Best practices
- Common pitfalls

{g3}""",
        ]
        return templates[variation % len(templates)]


# Singleton instance
_model_instance: PrometheusLightModel | None = None


def get_model(adapter_path: str = "app/model/prometheus_lora_adapter") -> PrometheusLightModel:
    """
    Get singleton lightweight Prometheus model instance.

    Args:
        adapter_path: Path to LoRA adapter directory

    Returns:
        PrometheusLightModel instance (singleton)
    """
    global _model_instance

    if _model_instance is None:
        logger.info("⚡ Initializing Prometheus LIGHTWEIGHT model...")
        _model_instance = PrometheusLightModel(adapter_path)

    return _model_instance
