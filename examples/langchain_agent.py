"""
Example: Using NEXUS with LangChain

This script demonstrates how to plug the NEXUS Dual-Process memory architecture 
natively into a LangChain ConversationChain.
"""

from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from nexus.core import NEXUS
from nexus.integrations.langchain_memory import NexusLangChainMemory

def main():
    print("Initialize NEXUS Memory Engine...")
    nexus_engine = NEXUS(storage_path="./langchain_nexus_db")
    
    print("Wrapping NEXUS as a LangChain BaseMemory component...")
    nexus_memory = NexusLangChainMemory(nexus_client=nexus_engine, top_k=3)
    
    # Standard LangChain LLM setup
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
    
    # Custom prompt that expects the {history} key from NexusLangChainMemory
    template = """The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer, it truthfully says it does not know.

{history}

Human: {input}
AI:"""
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    
    conversation = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        memory=nexus_memory,
        verbose=True
    )
    
    # Interact with the agent
    print("\n--- Conversation Start ---")
    response1 = conversation.predict(input="Hi, my name is Alex and I'm a machine learning engineer focusing on computer vision.")
    print(f"AI: {response1}\n")
    
    response2 = conversation.predict(input="I really prefer using PyTorch over TensorFlow for my personal projects.")
    print(f"AI: {response2}\n")
    
    # The agent will recall previous turns using NEXUS fast vector search 
    # and working memory contextual priming natively in the background.
    response3 = conversation.predict(input="Can you remind me what my job is and what framework I prefer?")
    print(f"AI: {response3}\n")
    
    print("--- Conversation End ---")
    
    print("Triggering Asynchronous Consolidation (System 2)...")
    # Consolidate the conversation into the Semantic Palace graph
    nexus_engine.consolidate(depth="full")
    print("Consolidation Complete. Context is now archived in long-term graph memory.")

if __name__ == "__main__":
    main()
