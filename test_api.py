#!/usr/bin/env python3
"""
Test script for Prometheus API
Tests the /augment endpoint with various inputs
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing /health endpoint...")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_augment(raw_prompt, num_variations=2):
    """Test augment endpoint"""
    print("=" * 60)
    print(f"Testing /augment endpoint")
    print(f"Prompt: {raw_prompt}")
    print("=" * 60)
    
    payload = {
        "raw_prompt": raw_prompt,
        "num_variations": num_variations
    }
    
    response = requests.post(f"{BASE_URL}/augment", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nDetected Type: {data['detected_prompt_type']}")
        print(f"Model Type: {data['model_type']}")
        print(f"RAG Context Used: {data['rag_context_used']}")
        print(f"RAG Chunks: {data['rag_chunks_count']}")
        print(f"\nEnhanced Prompts ({len(data['enhanced_prompts'])}):")
        print("-" * 60)
        
        for i, enhanced in enumerate(data['enhanced_prompts'], 1):
            print(f"\n[Variation {i}]")
            print(enhanced)
            print("-" * 60)
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    # Test health first
    test_health()
    
    # Examples
    test_augment("Explain machine learning", 2)
    test_augment("Write a function to sort a list", 1)
    test_augment("Summarize a research paper", 1)
