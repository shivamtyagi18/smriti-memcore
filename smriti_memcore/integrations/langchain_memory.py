"""
LangChain Memory Integration for SMRITI.
Allows SMRITI to be used natively as a BaseChatMemory or BaseMemory component 
in any LangChain agent or chain.
"""

from typing import List, Dict, Any

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
except ImportError:
    raise ImportError(
        "To use the SMRITI LangChain integration, you must install langchain-core:\n"
        "pip install langchain-core"
    )

from smriti_memcore.core import SMRITI

class SmritiLangChainHistory(BaseChatMessageHistory):
    """
    A LangChain message history backed by the SMRITI Dual-Process Architecture.
    
    This provides capacity-bounded working memory, semantic palace routing, 
    and asynchronous background consolidation, saving thousands of tokens 
    while improving factual recall accuracy.
    """
    smriti_client: SMRITI
    session_id: str
    top_k: int = 5
    
    def __init__(self, smriti_client: SMRITI, session_id: str = "default", top_k: int = 5):
        self.smriti_client = smriti_client
        self.session_id = session_id
        self.top_k = top_k
        self._local_history: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        """
        Return the relevant history.
        This queries the SMRITI semantic palace (System 2) AND episodic buffer (System 1) 
        using recent local history as context to achieve full Dual-Process memory.
        """
        if not self._local_history:
            return []
            
        # Get the most recent user query to fetch context
        last_query = self._local_history[-1].content if self._local_history else "Hello"
        
        # System 2: Fetch abstract knowledge graph memories
        memories = self.smriti_client.recall(str(last_query), top_k=self.top_k)
        
        # System 1: Fetch exact raw episodic event logs
        episodes = self.smriti_client.episode_buffer.search_semantic(str(last_query), top_k=self.top_k)
        
        # Format for the LLM prompt (Injecting Dual-Process Memory as System Context)
        context_blocks = []
        if memories:
            context_blocks.append("Abstract Knowledge:\n" + "\n".join(f"- {m.content}" for m in memories))
        if episodes:
            context_blocks.append("Specific Past Events:\n" + "\n".join(f"- {ep.content}" for ep in episodes))
            
        if context_blocks:
            context_str = "Relevant Long-Term Memories:\n\n" + "\n\n".join(context_blocks)
            # Prepend context to the local conversational window
            return [AIMessage(content=context_str)] + self._local_history
            
        return self._local_history

    def add_message(self, message: BaseMessage) -> None:
        """
        Save a message to this conversation iteration and ingest into SMRITI.
        This feeds the Episode Buffer (System 1) instantly.
        """
        self._local_history.append(message)
        
        # Ingest into SMRITI Long-Term Memory
        prefix = "Human: " if isinstance(message, HumanMessage) else "AI: "
        self.smriti_client.encode(f"{prefix} {message.content}")

    def clear(self) -> None:
        """Clear the local session window. (Does not clear long-term archival SMRITI storage)."""
        self._local_history = []
