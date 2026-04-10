"""
Example: Using SMRITI with LangChain

This script demonstrates how to plug the SMRITI Dual-Process memory architecture 
natively into a LangChain chain using LCEL.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from smriti_memcore.core import SMRITI, SmritiConfig
from smriti_memcore.integrations.langchain_memory import SmritiLangChainHistory

def main():
    print("Initialize SMRITI Memory Engine...")
    config = SmritiConfig(storage_path="./langchain_smriti_db", llm_model="gpt-4o-mini")
    smriti_engine = SMRITI(config=config)
    
    print("Wrapping SMRITI as a LangChain BaseChatMessageHistory component...")
    smriti_history = SmritiLangChainHistory(smriti_client=smriti_engine, session_id="test_session", top_k=3)
    
    # Standard LangChain LLM setup
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
    
    # Modern LangChain 0.3+ ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful memory-augmented AI. Use the provided context if needed."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    # Bind the SMRITI history to the chain
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: smriti_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Helper to run the chain
    def chat(user_input: str):
        print(f"Human: {user_input}")
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "test_session"}}
        )
        print(f"AI: {response.content}\n")
        return response.content

    # Interact with the agent
    print("\n--- Conversation Start ---")
    chat("Hi, my name is Alex and I'm a machine learning engineer focusing on computer vision.")
    chat("I really prefer using PyTorch over TensorFlow for my personal projects.")
    
    # The agent will recall previous turns using SMRITI fast vector search 
    # and working memory contextual priming natively in the background.
    chat("Can you remind me what my job is and what framework I prefer?")
    
    print("--- Conversation End ---")
    
    print("Triggering Asynchronous Consolidation (System 2)...")
    # Consolidate the conversation into the Semantic Palace graph
    smriti_engine.consolidate(depth="full")
    print("Consolidation Complete. Context is now archived in long-term graph memory.")

if __name__ == "__main__":
    main()
