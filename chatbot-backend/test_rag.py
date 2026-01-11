#!/usr/bin/env python3
"""
Standalone RAG Testing Script

This script tests the RAG functionality independently by:
1. Initializing the embedding provider (Gemini)
2. Initializing the database provider (Firestore)
3. Taking user input and retrieving relevant chunks from the vector store

Usage:
    python test_rag.py
    
Then enter your query when prompted.
"""
import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings, get_logger
from app.providers.embeddings import embedding_provider
from app.providers.database import database_provider

logger = get_logger("test_rag")


async def test_rag_retrieval(query: str) -> None:
    """Test RAG retrieval for a given query."""
    print("\n" + "=" * 80)
    print(f"Testing RAG for query: '{query}'")
    print("=" * 80)
    
    # Step 1: Check embedding provider
    print("\n[1] Checking Embedding Provider...")
    print(f"    Provider: {embedding_provider.get_provider_name()}")
    print(f"    Model: {embedding_provider.get_model_name()}")
    print(f"    Dimensions: {embedding_provider.get_dimensions()}")
    print(f"    Available: {embedding_provider.is_available()}")
    
    if not embedding_provider.is_available():
        print("    ❌ Embedding provider is not available!")
        return
    
    # Step 2: Check database provider
    print("\n[2] Checking Database Provider...")
    print(f"    Provider: {database_provider.get_provider_name()}")
    print(f"    Available: {database_provider.is_available()}")
    
    if not database_provider.is_available():
        print("    ⚠️  Database provider not initialized, attempting to initialize...")
        success = await database_provider.initialize()
        if not success:
            print("    ❌ Failed to initialize database provider!")
            print("\n📋 Troubleshooting:")
            print("    - Check if FIREBASE_CREDS_BASE64 is set in .env")
            print("    - Verify Firestore database has the vector collection")
            print(f"    - Collection name: {settings.FIRESTORE_VECTOR_COLLECTION}")
            return
        print(f"    ✅ Database provider initialized successfully")
    else:
        print("    ✅ Database provider is available")
    
    # Step 3: Generate embedding
    print("\n[3] Generating Embedding...")
    print(f"    Query: {query}")
    try:
        embedding = await embedding_provider.generate_embedding(query)
        if not embedding:
            print("    ❌ Failed to generate embedding (returned None)")
            return
        print(f"    ✅ Embedding generated successfully")
        print(f"    Vector length: {len(embedding)}")
        print(f"    First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"    ❌ Error generating embedding: {e}")
        return
    
    # Step 4: Search vector store
    print("\n[4] Searching Vector Store...")
    print(f"    Collection: {settings.FIRESTORE_VECTOR_COLLECTION}")
    print(f"    Vector Field: {settings.FIRESTORE_VECTOR_FIELD}")
    print(f"    Top K: {settings.RAG_TOP_K}")
    print(f"    Similarity Threshold: {settings.RAG_SIMILARITY_THRESHOLD}")
    
    try:
        results = await database_provider.search_similar(
            embedding=embedding,
            top_k=settings.RAG_TOP_K,
            similarity_threshold=settings.RAG_SIMILARITY_THRESHOLD,
        )
        
        print(f"\n    ✅ Search completed")
        print(f"    Results found: {len(results)}")
        
        if not results:
            print("\n    ⚠️  No results found!")
            print("\n📋 Possible reasons:")
            print("    - Vector collection is empty")
            print("    - Similarity threshold too high")
            print("    - Query not matching any indexed content")
            print("\n💡 Try:")
            print(f"    - Lowering similarity threshold (current: {settings.RAG_SIMILARITY_THRESHOLD})")
            print("    - Checking if vector collection has data")
            return
        
        # Display results
        print("\n" + "=" * 80)
        print("RAG RESULTS")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"  Similarity Score: {result.score:.4f}")
            print(f"  Text ({len(result.text)} chars):")
            # Print first 300 chars with proper indentation
            text_preview = result.text[:300].replace('\n', '\n  ')
            print(f"  {text_preview}")
            if len(result.text) > 300:
                print(f"  ... ({len(result.text) - 300} more characters)")
            
            if result.metadata:
                print(f"  Metadata: {result.metadata}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"    ❌ Error searching vector store: {e}")
        import traceback
        traceback.print_exc()
        return


async def main():
    """Main function to run the RAG test."""
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "RAG TESTING SCRIPT" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Display configuration
    print("\n📋 Configuration:")
    print(f"    Embedding Provider: {settings.EMBEDDING_PROVIDER}")
    print(f"    Embedding Model: {settings.EMBEDDING_MODEL_ID}")
    print(f"    Database Provider: {settings.DATABASE_PROVIDER}")
    print(f"    Firestore Collection: {settings.FIRESTORE_VECTOR_COLLECTION}")
    print(f"    RAG Top K: {settings.RAG_TOP_K}")
    print(f"    Similarity Threshold: {settings.RAG_SIMILARITY_THRESHOLD}")
    
    # Get user input
    print("\n" + "-" * 80)
    query = input("\n💬 Enter your query (or 'quit' to exit): ").strip()
    
    if not query or query.lower() == 'quit':
        print("Exiting...")
        return
    
    # Test RAG retrieval
    await test_rag_retrieval(query)
    
    # Ask if user wants to test another query
    print("\n" + "-" * 80)
    another = input("\n🔄 Test another query? (y/n): ").strip().lower()
    if another == 'y':
        await main()
    else:
        print("\n✨ Done! Exiting...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
