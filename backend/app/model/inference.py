"""
Prometheus Inference Module with LoRA Fine-Tuning
==================================================

Loads Mistral-7B with trained LoRA adapters for intelligent prompt enhancement.
Falls back to template-based generation if model loading fails.
"""

import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🖥️ Using device: {DEVICE}")


class PrometheusModel:
    """
    Prometheus model with LoRA fine-tuning support.
    
    Attempts to load Mistral-7B with trained LoRA adapters.
    Falls back to template-based generation if loading fails.
    """

    def __init__(self, adapter_path: str):
        """
        Initialize Prometheus model.

        Args:
            adapter_path: Path to LoRA adapter directory
        """
        self.adapter_path = Path(adapter_path)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.use_templates = False  # Fallback mode
        
        # Verify adapter files exist - REQUIRED for this version
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [f for f in required_files if not (self.adapter_path / f).exists()]
        
        if missing:
            error_msg = f"❌ FATAL: Missing fine-tuned adapter files: {missing}\n"
            error_msg += "━" * 70 + "\n"
            error_msg += "Prometheus requires a fine-tuned LoRA model to run.\n"
            error_msg += "You must train the model on Google Colab first:\n\n"
            error_msg += "  1. Open: Fine_Tune_Prometheus.ipynb in Google Colab\n"
            error_msg += "  2. Upload training_dataset.jsonl to Google Drive\n"
            error_msg += "  3. Run the notebook with a T4 GPU (~2 hours)\n"
            error_msg += "  4. Download the trained adapter_model.safetensors\n"
            error_msg += "  5. Replace the file in: backend/app/model/prometheus_lora_adapter/\n"
            error_msg += "  6. Restart the backend\n"
            error_msg += "━" * 70
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Try to load full model with LoRA
        self._load_model_with_lora()

    def _load_adapter_config(self):
        """Load adapter config for metadata even in template mode."""
        config_path = self.adapter_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.adapter_config = json.load(f)
            logger.info(f"📋 Adapter config loaded (template mode)")
        else:
            self.adapter_config = {}

    def _load_model_with_lora(self):
        """Load Prometheus model with LoRA adapters - FAST quantized version."""
        try:
            logger.info("=" * 60)
            logger.info("🚀 Loading Prometheus Model with LoRA Adapters (FAST)...")
            logger.info("=" * 60)
            
            # Import here to avoid slow startup if not needed
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            # Load adapter config
            config_path = self.adapter_path / "adapter_config.json"
            with open(config_path) as f:
                self.adapter_config = json.load(f)
            
            # Use llama-cpp-python for ultra-fast GGUF loading (3-5GB download)
            # Falls back to quantized transformers if llama-cpp not available
            try:
                logger.info("📥 Attempting ultra-fast GGUF model loading...")
                from llama_cpp import Llama
                
                # Download GGUF version (3-5GB, much faster)
                base_model_name = "TheBloke/Mistral-7B-Instruct-v0.1-Q5_K_M.gguf"
                logger.info(f"📦 Base model: {base_model_name} (Q5_K_M GGUF, ~3.5GB)")
                logger.info(f"📊 LoRA rank: {self.adapter_config.get('r', 'N/A')}")
                logger.info(f"📊 LoRA alpha: {self.adapter_config.get('lora_alpha', 'N/A')}")
                
                logger.info("⏳ Downloading GGUF model (~3-5 min on good internet)...")
                
                # This uses huggingface-hub to download GGUF
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                    filename="mistral-7b-instruct-v0.1.Q5_K_M.gguf",
                    cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
                )
                
                logger.info(f"✅ Downloaded to: {model_path}")
                logger.info("⏳ Loading GGUF model into memory...")
                
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1 if DEVICE == "cuda" else 0,
                    n_ctx=2048,
                    verbose=False,
                )
                
                # Load tokenizer for reference
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.adapter_path),
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.is_loaded = True
                logger.info("=" * 60)
                logger.info("✅ Prometheus Model loaded successfully (GGUF)!")
                logger.info(f"   Device: {DEVICE}")
                logger.info(f"   Format: GGUF Q5_K_M (quantized)")
                logger.info(f"   Size: ~3.5GB")
                logger.info("=" * 60)
                
            except ImportError:
                logger.warning("⚠️  llama-cpp-python not available, using standard quantized loading...")
                # Fallback to standard quantized loading
                self._load_model_quantized()
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error("   Please install llama-cpp-python: pip install llama-cpp-python")
            raise

    def _load_model_quantized(self):
        """Fallback: Load with standard quantized transformers (slower download but works)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import os
        
        logger.info("📦 Using quantized transformers loader...")
        
        # Load adapter config
        config_path = self.adapter_path / "adapter_config.json"
        with open(config_path) as f:
            self.adapter_config = json.load(f)
        
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        
        logger.info(f"📦 Base model: {base_model_name}")
        logger.info(f"📊 LoRA rank: {self.adapter_config.get('r', 'N/A')}")
        
        # Load tokenizer
        logger.info("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.adapter_path),
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Your MX550 has only 2GB VRAM - not enough for inference buffers
        # We'll run entirely on CPU with float16 for reasonable performance
        logger.info("🔧 Configuring CPU-only mode (GPU has insufficient VRAM)...")
        logger.info("   This will be slower but stable. Consider using llama.cpp for faster inference.")
        
        # Load base model on CPU only
        logger.info("⏳ Loading base model on CPU...")
        logger.info("   This may take 2-3 minutes and use ~14GB RAM...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": "cpu"},  # Force everything to CPU
            trust_remote_code=True,
            torch_dtype=torch.float32,  # CPU works better with float32
            low_cpu_mem_usage=True,
        )
        
        # Load LoRA adapter
        logger.info("🔗 Loading LoRA adapter weights...")
        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.adapter_path),
            is_trainable=False,
        )
        self.model.eval()
        
        self.is_loaded = True
        self.device = "cpu"  # Track that we're on CPU
        logger.info("=" * 60)
        logger.info("✅ Prometheus Model loaded successfully!")
        logger.info("   Mode: CPU-only (float32)")
        logger.info("   Note: Inference takes 30-60 seconds per request")
        logger.info("   Tip: Install llama-cpp-python for 5-10x faster inference")
        logger.info("=" * 60)

    def enhance_prompt(
        self,
        raw_prompt: str,
        prompt_type: str,
        rag_context: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 3,
    ) -> list[str]:
        """
        Generate enhanced prompts using the fine-tuned model.

        Args:
            raw_prompt: Original user prompt
            prompt_type: Detected prompt type (code, analysis, explain, etc.)
            rag_context: Retrieved guidelines context (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of variations to generate

        Returns:
            List of enhanced prompts
        """
        if self.use_templates or not self.is_loaded:
            logger.info("📋 Using template-based generation (model not loaded)")
            return self._generate_with_templates(
                raw_prompt, prompt_type, rag_context, num_return_sequences
            )
        
        logger.info(f"🤖 Generating {num_return_sequences} prompts with fine-tuned model")
        return self._generate_with_model(
            raw_prompt, prompt_type, rag_context,
            max_new_tokens, temperature, top_p, num_return_sequences
        )

    def _generate_with_model(
        self,
        raw_prompt: str,
        prompt_type: str,
        rag_context: str | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_return_sequences: int,
    ) -> list[str]:
        """Generate using the fine-tuned Mistral model."""
        
        # Build instruction prompt (same format as training)
        guidelines = ""
        if rag_context:
            context_lines = rag_context.strip().split("\n")[:5]
            guidelines = "\n".join(f"- {line}" for line in context_lines if line.strip())
        
        system_instruction = f"""You are Prometheus, an AI assistant specialized in enhancing prompts.

Your task is to transform the user's basic prompt into a more effective, detailed prompt that will get better results from AI assistants.

Prompt type detected: {prompt_type}

{"Relevant guidelines:\n" + guidelines if guidelines else ""}

Instructions:
1. Keep the original intent intact
2. Add specific details and context
3. Include format/structure guidance
4. Make it clear and actionable"""

        # Mistral instruction format
        conversation = f"<s>[INST] {system_instruction}\n\nOriginal prompt: {raw_prompt}\n\nEnhance this prompt: [/INST]"
        
        # Tokenize - send to the correct device
        device = getattr(self, 'device', 'cpu')  # Default to CPU
        inputs = self.tokenizer(
            conversation,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        # Move to correct device (CPU or CUDA)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate multiple sequences
        enhanced_prompts = []
        
        logger.info(f"  🔄 Starting generation on {device} (this may take 30-60 seconds)...")
        
        with torch.no_grad():
            for i in range(num_return_sequences):
                # Vary temperature slightly for diversity
                temp = temperature + (i * 0.1)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode and extract enhanced prompt
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the part after [/INST]
                if "[/INST]" in generated:
                    enhanced = generated.split("[/INST]")[-1].strip()
                else:
                    enhanced = generated[len(conversation):].strip()
                
                # Clean up
                enhanced = enhanced.replace("</s>", "").strip()
                
                if enhanced:
                    enhanced_prompts.append(enhanced)
                    logger.info(f"  ✅ Generated variation {i+1}/{num_return_sequences}")
        
        # Fallback to templates if generation failed
        if not enhanced_prompts:
            logger.warning("⚠️ Model generation produced no results, using templates")
            return self._generate_with_templates(
                raw_prompt, prompt_type, rag_context, num_return_sequences
            )
        
        return enhanced_prompts

    def _generate_with_templates(
        self,
        raw_prompt: str,
        prompt_type: str,
        rag_context: str | None,
        num_sequences: int,
    ) -> list[str]:
        """Fallback template-based generation."""
        
        guidelines = ""
        if rag_context:
            context_lines = rag_context.strip().split("\n")[:3]
            guidelines = "\n".join(f"- {line}" for line in context_lines if line.strip())
        
        g = f"\nGuidelines to consider:\n{guidelines}\n" if guidelines else ""
        
        templates_by_type = {
            "code": [
                f"""Task: {raw_prompt}

Please produce:
1. Goal: Restate the coding objective
2. Plan: Step-by-step approach
3. Example: Show a working snippet
4. Edge Cases: List tricky inputs
5. Pitfalls: Common mistakes to avoid
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
1. Context and assumptions
2. Options considered
3. Evidence for/against each
4. Decision and next steps
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
1. What it is
2. Why it matters
3. How it works (simple steps)
4. One practical example
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
        return [templates[i % len(templates)] for i in range(num_sequences)]


# Singleton instance
_model_instance: PrometheusModel | None = None


def get_model(adapter_path: str = "app/model/prometheus_lora_adapter") -> PrometheusModel:
    """
    Get singleton Prometheus model instance.

    Args:
        adapter_path: Path to LoRA adapter directory

    Returns:
        PrometheusModel instance (singleton)
    """
    global _model_instance

    if _model_instance is None:
        logger.info("🔥 Initializing Prometheus Model...")
        _model_instance = PrometheusModel(adapter_path)

    return _model_instance
