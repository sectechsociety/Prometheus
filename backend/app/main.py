from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

# Import our modules
from .model.inference import get_model
from .rag.retriever import retrieve_context, format_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Prometheus API",
    description="AI-powered prompt enhancement service with RAG + Fine-tuned Model",
    version="1.0.0"
)

# CORS for local dev (Vite at 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AugmentRequest(BaseModel):
    """Request to enhance a prompt."""
    raw_prompt: str = Field(..., min_length=1, max_length=2000, description="Original user prompt")
    target_model: str = Field(
        default="ChatGPT",
        pattern="^(ChatGPT|Claude|Gemini)$",
        description="Target AI model (ChatGPT, Claude, or Gemini)"
    )
    num_variations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of enhanced prompt variations to generate (1-5)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for generation (0.0-1.0, higher = more creative)"
    )
    use_rag: bool = Field(
        default=True,
        description="Include RAG context from knowledge base"
    )


class AugmentResponse(BaseModel):
    """Enhanced prompt response."""
    enhanced_prompts: List[str] = Field(..., description="List of enhanced prompts")
    target_model: str = Field(..., description="Target model used for enhancement")
    original_prompt: str = Field(..., description="Original input prompt")
    rag_context_used: bool = Field(..., description="Whether RAG context was retrieved and used")
    rag_chunks_count: int = Field(default=0, description="Number of RAG chunks retrieved")
    model_type: str = Field(..., description="Model type: 'mock' or 'fine-tuned'")


# Startup event - load model
@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    try:
        logger.info("=" * 80)
        logger.info("🚀 Starting Prometheus API Server...")
        logger.info("=" * 80)
        
        # Pre-load model to avoid cold start on first request
        model = get_model()
        
        if hasattr(model, 'is_mock') and model.is_mock:
            logger.warning("⚠️  MOCK MODEL ACTIVE")
            logger.warning("   Using rule-based templates until fine-tuning completes")
            logger.warning("   To replace: Update backend/app/model/inference.py after training")
        else:
            logger.info("✅ Fine-tuned model loaded successfully")
        
        # Check RAG system
        try:
            from .rag.vector_store import get_vector_store
            store = get_vector_store()
            collection = store.get_collection()
            count = collection.count()
            logger.info(f"✅ RAG system ready ({count} guidelines indexed)")
        except Exception as e:
            logger.warning(f"⚠️  RAG system unavailable: {e}")
        
        logger.info("=" * 80)
        logger.info("✅ Server ready to accept requests")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}")
        logger.warning("⚠️  Server will start but requests may fail")


# Root endpoint
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Prometheus API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "augment": "/augment",
            "health": "/health",
            "docs": "/docs"
        }
    }


# Main augment endpoint
@app.post("/augment", response_model=AugmentResponse)
async def augment_prompt(request: AugmentRequest):
    """
    Enhance a raw prompt using RAG + Model.
    
    Pipeline:
    1. Retrieve relevant guidelines from vector DB (if use_rag=True)
    2. Format context for readability
    3. Generate enhanced prompts using model (mock or fine-tuned)
    4. Return results with metadata
    
    Args:
        request: AugmentRequest with raw_prompt, target_model, etc.
        
    Returns:
        AugmentResponse with enhanced prompts and metadata
    """
    try:
        logger.info("=" * 60)
        logger.info(f"📝 Augmentation request received")
        logger.info(f"   Target model: {request.target_model}")
        logger.info(f"   Variations: {request.num_variations}")
        logger.info(f"   RAG enabled: {request.use_rag}")
        logger.info(f"   Prompt: {request.raw_prompt[:100]}...")
        
        # Step 1: Retrieve context from RAG (optional)
        rag_context_used = False
        rag_chunks_count = 0
        context_text = None
        
        if request.use_rag:
            try:
                logger.info("🔍 Retrieving relevant guidelines from RAG...")
                
                # Retrieve relevant guidelines
                chunks = retrieve_context(
                    query=request.raw_prompt,
                    target_model=request.target_model,
                    top_k=5
                )
                
                if chunks:
                    # Format for readability
                    context_text = format_context(chunks, max_chars=1500)
                    rag_context_used = True
                    rag_chunks_count = len(chunks)
                    
                    logger.info(f"✅ Retrieved {len(chunks)} relevant guidelines")
                    logger.info(f"   Top score: {chunks[0].score:.3f}")
                else:
                    logger.warning("⚠️  No RAG context found for this query")
                    
            except Exception as e:
                logger.error(f"❌ RAG retrieval failed: {e}")
                logger.warning("   Continuing without RAG context...")
                # Continue without RAG context
        else:
            logger.info("⏭️  RAG disabled, skipping context retrieval")
        
        # Step 2: Get model instance
        logger.info("🤖 Loading model...")
        model = get_model()
        
        # Step 3: Generate enhanced prompts with RAG context
        logger.info(f"✨ Generating {request.num_variations} enhanced prompts...")
        
        enhanced_prompts = model.enhance_prompt(
            raw_prompt=request.raw_prompt,
            target_model=request.target_model,
            rag_context=context_text if context_text else None,
            num_return_sequences=request.num_variations,
            temperature=request.temperature
        )
        
        logger.info(f"✅ Generated {len(enhanced_prompts)} enhanced prompts")
        
        # Determine model type
        model_type = "mock" if (hasattr(model, 'is_mock') and model.is_mock) else "fine-tuned"
        
        logger.info("=" * 60)
        
        return AugmentResponse(
            enhanced_prompts=enhanced_prompts,
            target_model=request.target_model,
            original_prompt=request.raw_prompt,
            rag_context_used=rag_context_used,
            rag_chunks_count=rag_chunks_count,
            model_type=model_type
        )
        
    except Exception as e:
        logger.error(f"❌ Enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prompt enhancement failed: {str(e)}"
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Detailed health check with model and RAG status."""
    health_status = {
        "service": "Prometheus API",
        "status": "unknown",
        "model": {},
        "rag": {},
        "timestamp": None
    }
    
    try:
        # Check model status
        model = get_model()
        model_loaded = model is not None
        is_mock = hasattr(model, 'is_mock') and model.is_mock
        
        health_status["model"] = {
            "loaded": model_loaded,
            "type": "mock" if is_mock else "fine-tuned" if model_loaded else "unknown",
            "ready": model_loaded
        }
        
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        health_status["model"] = {
            "loaded": False,
            "type": "unknown",
            "ready": False,
            "error": str(e)
        }
    
    try:
        # Check RAG system
        from .rag.vector_store import get_vector_store
        store = get_vector_store()
        collection = store.get_collection()
        count = collection.count()
        
        health_status["rag"] = {
            "available": True,
            "collection": "prometheus_guidelines",
            "document_count": count,
            "ready": count > 0
        }
        
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        health_status["rag"] = {
            "available": False,
            "ready": False,
            "error": str(e)
        }
    
    # Overall status
    model_ok = health_status["model"].get("ready", False)
    rag_ok = health_status["rag"].get("ready", False)
    
    if model_ok and rag_ok:
        health_status["status"] = "healthy"
    elif model_ok or rag_ok:
        health_status["status"] = "degraded"
    else:
        health_status["status"] = "unhealthy"
    
    # Add timestamp
    from datetime import datetime
    health_status["timestamp"] = datetime.utcnow().isoformat()
    
    return health_status
