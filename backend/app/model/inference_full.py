"""
Prometheus Model Inference Module
==================================

Loads the fine-tuned Mistral-7B model with LoRA adapters for prompt enhancement.
Supports efficient inference with RAG context integration.
"""

from typing import List, Optional
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)


class PrometheusModel:
    """
    Fine-tuned Mistral-7B model with LoRA adapters for prompt enhancement.
    Supports RAG context integration and efficient inference.
    """
    
    def __init__(self, adapter_path: str, device: str = "auto"):
        """
        Initialize and load the fine-tuned model.
        
        Args:
            adapter_path: Path to LoRA adapter directory
            device: Device for inference ("auto", "cuda", "cpu")
        """
        logger.info("🚀 Loading Prometheus fine-tuned model...")
        self.is_mock = False
        self.adapter_path = adapter_path
        
        # Detect available device
        has_cuda = torch.cuda.is_available()
        use_device = "cuda" if has_cuda and device != "cpu" else "cpu"
        logger.info(f"🖥️  Using device: {use_device}")
        
        # Check if bitsandbytes is available for quantization
        use_quantization = False
        if has_cuda:
            try:
                import bitsandbytes
                from transformers import BitsAndBytesConfig
                use_quantization = True
                logger.info("✅ 8-bit quantization enabled (GPU + bitsandbytes)")
            except ImportError:
                logger.warning("⚠️  bitsandbytes not available, using full precision")
        
        # Load base model
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        logger.info(f"📦 Loading base model: {base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load with or without quantization based on availability
        if use_quantization:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            # CPU or GPU without quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map={"": use_device},
                trust_remote_code=True,
                torch_dtype=torch.float16 if has_cuda else torch.float32,
                low_cpu_mem_usage=True,
            )
        
        # Load LoRA adapter
        logger.info(f"🔧 Loading LoRA adapter from: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
        logger.info("✅ Prometheus model loaded successfully!")
    
    def enhance_prompt(
        self,
        raw_prompt: str,
        target_model: str = "ChatGPT",
        rag_context: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate enhanced prompts using the fine-tuned model.
        
        Args:
            raw_prompt: Original user prompt
            target_model: Target AI model (ChatGPT/Claude/Gemini)
            rag_context: Retrieved guidelines context (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of variations to generate
            
        Returns:
            List of enhanced prompts
        """
        logger.info(f"🎯 Generating {num_return_sequences} enhanced prompts for {target_model}")
        
        # Format input according to training format
        instruction = self._format_instruction(raw_prompt, target_model, rag_context)
        
        # Tokenize input
        inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate enhanced prompts
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        enhanced_prompts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Extract only the enhanced prompt (after the instruction)
            enhanced = self._extract_enhanced_prompt(text, instruction)
            enhanced_prompts.append(enhanced)
        
        logger.info(f"✅ Generated {len(enhanced_prompts)} enhanced prompts")
        return enhanced_prompts
    
    def _format_instruction(self, raw_prompt: str, target_model: str, rag_context: Optional[str]) -> str:
        """Format input according to the training format (Mistral Instruct)."""
        context_part = f"\n\nGuidelines:\n{rag_context}" if rag_context else ""
        
        instruction = f"""[INST] You are Prometheus, an expert prompt engineer. Enhance the following prompt for {target_model}.{context_part}

Raw Prompt: {raw_prompt}

Enhanced Prompt: [/INST]"""
        
        return instruction
    
    def _extract_enhanced_prompt(self, generated_text: str, instruction: str) -> str:
        """Extract the enhanced prompt from the generated text."""
        # Remove the instruction part
        if "[/INST]" in generated_text:
            enhanced = generated_text.split("[/INST]")[-1].strip()
        else:
            enhanced = generated_text.replace(instruction, "").strip()
        
        return enhanced


# Singleton instance
_model_instance: Optional[PrometheusModel] = None


def get_model(adapter_path: str = "app/model/prometheus_lora_adapter") -> PrometheusModel:
    """
    Get singleton Prometheus model instance.
    
    This uses lazy loading - the model is only initialized on first call.
    
    Args:
        adapter_path: Path to LoRA adapter directory
        
    Returns:
        PrometheusModel instance (singleton)
    """
    global _model_instance
    
    if _model_instance is None:
        logger.info("🚀 Initializing Prometheus fine-tuned model...")
        _model_instance = PrometheusModel(adapter_path)
    
    return _model_instance
