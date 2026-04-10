import os
import uuid
import time
from typing import Dict, Optional

try:
    from supermemory import Supermemory
except ImportError:
    Supermemory = None

from baselines.base import BaseMemorySystem, MemoryResponse


class SupermemoryAdapter(BaseMemorySystem):
    """
    Adapter for the cloud-hosted Supermemory API.
    Uses the supermemory python SDK.
    """

    def __init__(self, name: str = "supermemory", llm_interface=None, api_key: str = None):
        super().__init__(name, llm_interface)
        if Supermemory is None:
            raise ImportError("The 'supermemory' package is not installed. Run `pip install supermemory`")
        
        # Use provided key or environment variable
        self.api_key = api_key or os.environ.get("SUPERMEMORY_API_KEY")
        if not self.api_key:
            raise ValueError("Supermemory API Key is required. Set SUPERMEMORY_API_KEY environment variable.")
            
        # We might need to initialize the client with the api key if the SDK requires it
        # Assuming the standard signature Supermemory(api_key=...) or it auto-picks from env var
        try:
            self.client = Supermemory(api_key=self.api_key)
        except TypeError:
            # If the initialized doesn't take api_key or behaves differently
            self.client = Supermemory()
            
        # We need a unique container tag for each run so they don't leak between benchmark sessions
        self.container_tag = f"bench_{uuid.uuid4().hex[:8]}"

    def ingest(self, message: str, role: str = "user", metadata: Optional[Dict] = None):
        """Send episodic memory to supermemory API."""
        if role != "user":
            # For simplicity, we typically only ingest user facts in this benchmark
            return

        formatted_message = message
        if metadata and "timestamp" in metadata:
            formatted_message = f"[{metadata['timestamp']}] {message}"

        # client.add() is synchronous or async depending on SDK version
        # If it's async in python SDK, we'd need to handle that, but let's assume sync for standard usage
        try:
            self.client.add(content=formatted_message, container_tag=self.container_tag)
            self._ingest_count += 1
        except Exception as e:
            print(f"[Supermemory Error in ingest]: {e}")

    def query(self, question: str, context: str = "") -> MemoryResponse:
        """Retrieve memory from supermemory API and answer with LLM."""
        start_time = time.time()
        
        retrieved_texts = []
        try:
            # client.profile returns static and dynamic profiles
            result = self.client.profile(container_tag=self.container_tag, q=question)
            
            # The result structure might be result.profile.static and result.profile.dynamic
            # Let's extract them into a combined text block
            static_facts = getattr(result.profile, "static", [])
            dynamic_context = getattr(result.profile, "dynamic", [])
            
            for f in static_facts:
                retrieved_texts.append(str(f))
            for f in dynamic_context:
                retrieved_texts.append(str(f))
                
        except Exception as e:
            print(f"[Supermemory Error in query profile]: {e}")
            retrieved_texts.append(f"(Error retrieving profile: {str(e)})")

        retrieval_latency = (time.time() - start_time) * 1000

        # Construct prompt for the LLM using the retrieved context
        memory_context = "\n".join([f"- {m}" for m in retrieved_texts]) if retrieved_texts else "No relevant memories found."
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question using the provided memory context.

<memory_context>
{memory_context}
</memory_context>

<current_context>
{context}
</current_context>

Question: {question}

Answer directly and concisely."""

        # Call the local/benchmark LLM to generate the final answer
        # This ensures the LLM reasoning latency is comparable across baselines
        llm_response = self.llm.generate(prompt)
        
        # Calculate overall latency
        total_latency = (time.time() - start_time) * 1000
        
        response = MemoryResponse(
            answer=llm_response.text.strip(),
            memories_used=retrieved_texts,
            latency_ms=total_latency,
            tokens_used=llm_response.tokens_used,
        )
        
        self._query_count += 1
        self._total_latency_ms += total_latency
        
        return response

    def reset(self):
        """
        Reset changes the container tag so the next conversation starts fresh.
        """
        self.container_tag = f"bench_{uuid.uuid4().hex[:8]}"
        self._ingest_count = 0
        self._query_count = 0
        self._total_latency_ms = 0.0
