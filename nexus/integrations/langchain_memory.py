"""
LangChain Memory Integration for NEXUS.
Allows NEXUS to be used natively as a BaseChatMemory or BaseMemory component 
in any LangChain agent or chain.
"""

from typing import Dict, Any, List
import warnings

try:
    from langchain_core.memory import BaseMemory
    from pydantic import Field
except ImportError:
    raise ImportError(
        "To use the NEXUS LangChain integration, you must install langchain-core:\n"
        "pip install langchain-core"
    )

from nexus.core import NEXUS

class NexusLangChainMemory(BaseMemory):
    """
    A LangChain memory class backed by the NEXUS Dual-Process Architecture.
    
    This replaces standard ConversationBufferMemory and provides capacity-bounded 
    working memory, semantic palace routing, and asynchronous background consolidation,
    saving thousands of tokens while improving factual recall accuracy.
    """
    nexus_client: NEXUS = Field(description="The active NEXUS instance.")
    memory_key: str = "history"
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    top_k: int = 5
    
    @property
    def memory_variables(self) -> List[str]:
        """Return the keys this memory class will add to the prompt."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return the relevant history for the current input.
        This queries the NEXUS semantic palace (System 1 + System 2).
        """
        # Usually 'input' or 'human_input' is passed. 
        # Pick the first string value as the query to retrieve context.
        query = ""
        for v in inputs.values():
            if isinstance(v, str):
                query = v
                break
                
        if not query:
            return {self.memory_key: ""}

        # Fetch memories using NEXUS
        memories = self.nexus_client.recall(query, top_k=self.top_k)
        
        # Format for the LLM prompt
        if not memories:
            return {self.memory_key: ""}
            
        context_str = "Relevant Long-Term Memories:\n"
        for mem in memories:
            context_str += f"- {mem}\n"
            
        return {self.memory_key: context_str}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save the context from this conversation iteration to NEXUS.
        This feeds the Episode Buffer (System 1) instantly.
        """
        # Get user input
        user_msg = ""
        for v in inputs.values():
            if isinstance(v, str):
                user_msg = v
                break
                
        # Get AI output
        ai_msg = ""
        for v in outputs.values():
            if isinstance(v, str):
                ai_msg = v
                break

        if user_msg:
            self.nexus_client.encode(f"{self.human_prefix}: {user_msg}", auto_consolidate=True)
            
        if ai_msg:
            self.nexus_client.encode(f"{self.ai_prefix}: {ai_msg}", auto_consolidate=True)

    def clear(self) -> None:
        """Clear memory contents. (Not strictly supported for persistent long-term storage)."""
        warnings.warn("clear() is ignored by NexusLangChainMemory as NEXUS is a persistent archival storage layer.")
