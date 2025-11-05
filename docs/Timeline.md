### Phase 1: Foundation & Data Collection (Weeks 1-2) 🏗️
**Status: ✅ COMPLETE**

#### **Week 1: Project Kickstart & Scaffolding**
* **Primary Focus:** Environment Setup & Initial Prototypes
* **Tasks Completed:**
    * ✅ Initialize Git repository with branching strategy (main, Week1, Week2)
    * ✅ Create project structure: `backend/`, `frontend/`, `services/ingest/`, `docs/`
    * ✅ Write comprehensive `README.md` with architecture diagram
    * ✅ Set up FastAPI backend skeleton in `backend/app/main.py`
        - ✅ POST `/augment` endpoint stub with Pydantic models
        - ✅ CORS middleware for localhost:5173
        - ✅ Backend Dockerfile with Python 3.11-slim
    * ✅ Create React + Vite frontend application
        - ✅ Components: PromptBar, Results, ResultCard
        - ✅ Dark/light theme toggle with CSS variables
        - ✅ Custom fonts: BBH Sans Hegarty, Montserrat
        - ✅ 3D textarea effects and minimal UI design
        - ✅ Model selector: ChatGPT, Gemini, Claude
    * ✅ Docker Compose configuration for local development
    * ✅ VS Code tasks for quick commands
    * ✅ Create `.github/copilot-instructions.md` for AI agent guidance

#### **Week 2: Data Refinement & Dataset Creation**
* **Primary Focus:** Building the Ingest Pipeline
* **Tasks Completed:**
    * ✅ Implement `services/ingest/ingest.py`:
        - ✅ `clean_html()` with BeautifulSoup and plain-text fallback
        - ✅ `normalize_text()` for whitespace and punctuation
        - ✅ `chunk_text()` with sentence boundaries (configurable max_chars)
        - ✅ `dedupe_and_filter()` for quality control
        - ✅ `export_jsonl()` with metadata (UUID, timestamp, model, tags)
        - ✅ CLI with argparse: --source-dir, --out, --max-chars, --target-model
    * ✅ Create `services/ingest/validate_jsonl.py`:
        - ✅ Line-by-line validation with required fields check
        - ✅ Statistics reporting (total, valid, invalid, per-model)
    * ✅ Collect initial prompting guides:
        - ✅ ChatGPT best practices (stored in `docs/Datasets/ChatGPT/`)
        - ✅ Gemini prompting documentation (stored in `docs/Datasets/Gemini/`)
    * ✅ Update `docs/Progress Log.md` with Week 2 contributions

---

### Phase 2: Core Model Development (Weeks 3-5) 🧠
**Status: 🚧 IN PROGRESS (Week 3)**

#### **Week 3: Building the RAG Knowledge Base**
* **Primary Focus:** Vector Database Setup & Retrieval System
* **Day 1-2: Dataset Generation (200-300 examples)**
    * ✅ Run ingest pipeline on collected HTML/TXT sources:
        ```bash
        python services/ingest/ingest.py \
          --source-dir docs/Datasets/ChatGPT \
          --out services/ingest/data/chatgpt_guidelines.jsonl \
          --target-model ChatGPT
        ```
    * ✅ Repeat for Gemini and Claude datasets
    * ✅ Validate all outputs with `validate_jsonl.py`
    * ✅ Merge validated JSONL files into single dataset
    * ✅ Target: 200-300 quality chunks → **Achieved: 811 chunks!**
        - ChatGPT: 513 chunks
        - Gemini: 148 chunks
        - Claude: 150 chunks
        - Merged file: `services/ingest/data/all_guidelines.jsonl`

* **Day 3-4: Vector Database Setup**
    * ✅ Install ChromaDB and embeddings stack: `pip install chromadb sentence-transformers torch`
    * ✅ Create `backend/app/rag/__init__.py` (empty module marker)
    * ✅ Create `backend/app/rag/vector_store.py`:
        - ✅ Initialize ChromaDB persistent client (persisted to `services/ingest/chroma_db`)
        - ✅ Define collection and metadata storage (source, target_model, chunk_id, created_at)
        - ✅ Write `add_documents()` function (batch-embeds + insert)
        - ✅ Write `search()` function (query → embedding → top-k results, optional target_model filter)
    * ✅ Create `backend/app/rag/embeddings.py`:
        - ✅ Load sentence-transformer model: `all-MiniLM-L6-v2`
        - ✅ `generate_embedding()` for single text
        - ✅ `batch_generate_embeddings()` for efficiency
    * ✅ Test vector store with sample inserts and queries
        - Result: Collection created and query works. Sample query "write persuasive ad copy for a SaaS product" (ChatGPT filter) returned relevant chunks. 
        - Current collection count after test population: 811

* **Day 5: Ingestion Script & Population**
    * ✅ Create `backend/app/rag/populate_db.py`:
        - ✅ Load JSONL dataset from `services/ingest/data/`
        - ✅ For each item: generate embedding, insert into ChromaDB
        - ✅ Add metadata: source, target_model, chunk_id, created_at
        - ✅ Print progress (every 50 items)
    * ✅ Run population script:
        - From repo root: `python -m backend.app.rag.populate_db`
    * ✅ Verify collection count matches expected documents → 811

* **Day 6-7: Retrieval Function & Testing**
    * ✅ Create `backend/app/rag/retriever.py`:
        - ✅ `retrieve_context(query: str, target_model: Optional[str], top_k: int = 5)`
        - ✅ Filter by target_model metadata (ChatGPT/Gemini/Claude)
        - ✅ Return list of `RetrievedChunk` objects with text, scores, distances, metadata
        - ✅ `format_context()` helper to build prompt-ready context string
        - ✅ CLI tool for testing: `python -m backend.app.rag.retriever --query "..." --top-k N`
    * ✅ Test retrieval quality with sample queries:
        - ✅ "Explain machine learning" → Retrieved 5 relevant ChatGPT tutorial chunks (scores: 0.44-0.43)
        - ✅ "Summarize a research paper" → Retrieved 5 summarization guidelines (scores: 0.50-0.47)
        - ✅ "Write a product description" → Retrieved 10 creative writing guidelines (scores: 0.54-0.45)
        - ✅ Model filtering verified: `--target-model ChatGPT` returns only ChatGPT chunks
    * ✅ Tune top_k parameter (tried 3, 5, 10):
        - **Recommendation:** `top_k=5` is optimal for most queries (balances relevance and context size)
        - Use `top_k=3` for tighter, more focused context (shorter prompts)
        - Use `top_k=10` for broad or complex queries requiring diverse examples
    * ✅ Document retrieval behavior in `backend/README.md`:
        - Added API surface documentation
        - Included tuning guidance and score interpretation
        - Provided CLI test examples
        - Added integration notes for `/augment` endpoint

#### **Week 4: Fine-Tuning Dataset Expansion**
* **Primary Focus:** Scale to 1,000 Training Examples
* **Day 1-2: Seed Prompt Collection**
    * [ ] Manually create 50 diverse seed prompts:
        - [ ] 15 "Explain [concept]" variations
        - [ ] 15 "Summarize [text type]" variations
        - [ ] 10 "Generate [content type]" variations
        - [ ] 10 "Analyze [topic]" variations
    * [ ] For each seed, write expert-level enhanced version
    * [ ] Store in `services/ingest/data/seed_prompts.jsonl`
    * [ ] Ensure diversity: different audiences, constraints, formats

* **Day 3-5: Synthetic Augmentation**
    * [ ] Create `services/ingest/augment_dataset.py`:
        - [ ] Load seed prompts
        - [ ] Apply augmentation strategies:
            - Add audience variations: "for a 10-year-old", "for an expert"
            - Add format constraints: "in bullet points", "as a table"
            - Add length constraints: "in 50 words", "in detail"
            - Add style variations: "formal", "casual", "technical"
        - [ ] Generate 3-5 variations per seed (50 seeds × 4 = 200 examples)
        - [ ] Add model-specific transformations (ChatGPT/Gemini/Claude styles)
    * [ ] Run augmentation: `python services/ingest/augment_dataset.py`
    * [ ] Target output: 800-1,000 total examples
    * [ ] Validate with `validate_jsonl.py`

* **Day 6-7: Quality Review & Colab Setup**
    * [ ] Random sample 10% of augmented examples (~80-100)
    * [ ] Manual review: check for quality, diversity, realism
    * [ ] Remove low-quality or duplicate examples
    * [ ] Finalize training dataset: `services/ingest/data/training_dataset.jsonl`
    * [ ] Upload dataset to Google Drive: `/Prometheus/training_data/`
    * [ ] Create Google Colab notebook: `Fine_Tune_Prometheus.ipynb`
    * [ ] Install libraries in Colab:
        ```python
        !pip install transformers peft bitsandbytes accelerate datasets
        ```
    * [ ] Mount Google Drive and verify dataset path

#### **Week 5: Model Fine-Tuning**
* **Primary Focus:** Train the Fine-Tuned LLM
* **Day 1-2: Training Pipeline Setup**
    * [ ] In Colab notebook, implement base model loading:
        ```python
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = "mistralai/Mistral-7B-v0.1"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, load_in_4bit=True, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        ```
    * [ ] Implement dataset loading from JSONL:
        ```python
        dataset = load_dataset("json", data_files="/content/drive/.../training_dataset.jsonl")
        ```
    * [ ] Format dataset for training:
        - [ ] Create prompt template: `"Input: {input_prompt}\nEnhanced: {enhanced_prompt}"`
        - [ ] Tokenize with truncation (max_length=512)
    * [ ] Configure LoRA parameters:
        ```python
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        ```

* **Day 3-4: Training Execution**
    * [ ] Set training arguments:
        ```python
        training_args = TrainingArguments(
            output_dir="/content/drive/.../checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            save_steps=100,
            logging_steps=10
        )
        ```
    * [ ] Start training with SFTTrainer:
        ```python
        trainer = SFTTrainer(
            model=model, train_dataset=dataset,
            peft_config=lora_config, args=training_args
        )
        trainer.train()
        ```
    * [ ] Monitor training in Colab output:
        - [ ] Check loss curve (should decrease steadily)
        - [ ] Verify no NaN or exploding gradients
    * [ ] Training time estimate: 4-6 hours on T4 GPU

* **Day 5-6: Model Validation & Export**
    * [ ] Test model on held-out examples in Colab:
        ```python
        test_prompt = "Input: Explain quantum computing\nEnhanced:"
        output = model.generate(tokenizer.encode(test_prompt, return_tensors="pt"))
        print(tokenizer.decode(output[0]))
        ```
    * [ ] Compare outputs: base model vs fine-tuned model
    * [ ] Check for common failure modes:
        - [ ] Repetitive text
        - [ ] Off-topic responses
        - [ ] Formatting issues
    * [ ] If quality is good, merge LoRA adapter:
        ```python
        model = model.merge_and_unload()
        model.save_pretrained("/content/drive/.../prometheus_model_final")
        tokenizer.save_pretrained("/content/drive/.../prometheus_model_final")
        ```

* **Day 7: Download & Local Setup**
    * [ ] Download model from Google Drive to local machine
    * [ ] Place in `backend/models/prometheus_v1/`
    * [ ] Test local loading:
        ```python
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("backend/models/prometheus_v1")
        tokenizer = AutoTokenizer.from_pretrained("backend/models/prometheus_v1")
        ```
    * [ ] Verify model can generate text locally

---

### Phase 3: Integration & MVP (Week 6) 🔌
**Status: ⏳ NOT STARTED**

#### **Week 6: End-to-End Backend Integration**
* **Primary Focus:** Working API with RAG + Fine-Tuned Model
* **Day 1-2: Model Inference Module**
    * [ ] Create `backend/app/model/__init__.py`
    * [ ] Create `backend/app/model/inference.py`:
        - [ ] `load_model()`: Load fine-tuned model at startup
        - [ ] `generate_enhanced_prompt(raw_prompt: str, context: str, max_length: int = 256)`
        - [ ] Format input with retrieved context:
            ```
            Context: {retrieved_guidelines}
            
            Input: {raw_prompt}
            Enhanced:
            ```
        - [ ] Call model.generate() with appropriate parameters
        - [ ] Return cleaned output text
    * [ ] Test inference function standalone

* **Day 3-4: Complete `/augment` Endpoint**
    * [ ] Update `backend/app/main.py`:
        - [ ] Import retriever and inference modules
        - [ ] Load model at app startup (use @app.on_event("startup"))
        - [ ] Implement `/augment` logic:
            1. Receive `{ raw_prompt, target_model }`
            2. Call `retrieve_context(raw_prompt, target_model, top_k=5)`
            3. Format context into prompt template
            4. Call `generate_enhanced_prompt(raw_prompt, context)`
            5. Generate 3 variations (different temperature/top_p)
            6. Return `{ enhanced_prompts: [prompt1, prompt2, prompt3] }`
    * [ ] Add error handling:
        - [ ] Empty prompt → 400 Bad Request
        - [ ] Model not loaded → 503 Service Unavailable
        - [ ] Generation timeout → 504 Gateway Timeout
    * [ ] Add logging for debugging

* **Day 5: API Testing & Optimization**
    * [ ] Test with curl:
        ```bash
        curl -X POST http://localhost:8000/augment \
          -H "Content-Type: application/json" \
          -d '{"raw_prompt":"Explain DNS","target_model":"ChatGPT"}'
        ```
    * [ ] Test edge cases:
        - [ ] Very long prompt (>1000 chars)
        - [ ] Empty prompt
        - [ ] Special characters in prompt
        - [ ] Invalid target_model
    * [ ] Measure latency: should be <10s for typical prompts
    * [ ] Optimize if needed:
        - [ ] Cache model in memory (don't reload per request)
        - [ ] Use smaller top_k if retrieval is slow
        - [ ] Reduce max_length if generation is slow

* **Day 6: Frontend Integration**
    * [ ] Update `frontend/src/api/augment.js`:
        - [ ] Remove mock backend logic
        - [ ] Set API base to `http://localhost:8000`
        - [ ] Handle real API responses
    * [ ] Update `frontend/vite.config.mjs`:
        - [ ] Remove mock middleware
        - [ ] Add proxy configuration for `/augment` → `http://localhost:8000`
    * [ ] Test end-to-end flow:
        - [ ] Enter prompt in UI
        - [ ] Select model (ChatGPT/Gemini/Claude)
        - [ ] Click submit
        - [ ] Verify enhanced prompts display correctly
    * [ ] Add loading spinner while waiting for API
    * [ ] Display error messages gracefully

* **Day 7: Docker Compose & Documentation**
    * [ ] Update `docker-compose.yml`:
        - [ ] Add ChromaDB service (if not using embedded mode)
        - [ ] Ensure backend mounts model directory
        - [ ] Set environment variables for API endpoints
    * [ ] Test full stack with Docker Compose:
        ```bash
        docker-compose up --build
        ```
    * [ ] Verify frontend (port 5173) can communicate with backend (port 8000)
    * [ ] Update `README.md`:
        - [ ] Add "Quick Start" section with Docker Compose command
        - [ ] Document API endpoints and request/response formats
        - [ ] Add troubleshooting section for common issues
    * [ ] Create demo video or screenshots
    * [ ] Update `docs/Progress Log.md` with Week 6 accomplishments

---

### Phase 4: Polish & Deployment (Weeks 7-8) 📤
**Status: ⏳ OPTIONAL (Core MVP Complete by Week 6)**

#### **Week 7: Testing & Refinement (Optional)**
* **Primary Focus:** Quality Improvements
* **Optional Enhancements:**
    * [ ] Add user authentication (if needed)
    * [ ] Implement prompt history/favorites
    * [ ] Add copy-to-clipboard buttons
    * [ ] Display retrieval context in accordion (show which guidelines were used)
    * [ ] Add usage analytics/logging
    * [ ] Improve error messages and help text
    * [ ] Add loading progress indicators
    * [ ] Implement rate limiting on API

#### **Week 8: Deployment (Optional)**
* **Primary Focus:** Public Launch
* **Optional Deployment Steps:**
    * [ ] Choose deployment platform:
        - Hugging Face Spaces (free, good for demos)
        - Railway (easy Docker deployment)
        - AWS/GCP (more control, requires config)
    * [ ] Set up environment variables for production
    * [ ] Configure CORS for production domain
    * [ ] Add HTTPS/SSL certificates
    * [ ] Set up monitoring and logging
    * [ ] Create deployment documentation
    * [ ] Share with community (Twitter, Reddit, HN)

---
