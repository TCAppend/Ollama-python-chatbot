from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore

Message_History = [
{
    'id': 1,
    'prompt': 'What is your name',
    'response': 'My name is fred the cat, known as a pet of the family of barreras, and the family of braga'

}, 
{
    'id': 2,
    'prompt': 'Where are you from',
    'response': 'Land of the philippines, bulacan, san jose del monte'

}, 
{
    'id': 3,
    'prompt': 'Who is your owner',
    'response': 'My owner is known as Jay Marc Braga Barreras'

}
]

template = """
Here is the conversation history: {context}
Question: {question}
"""


embeddings = OllamaEmbeddings(model="llama3")
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome to the AI Bot!")
    while True:
        user_input = input("you: ")
        if user_input.lower() == "exit":
            break

        result = chain.invoke({"context": context, "question": user_input})
        print("Bot: ", result)
        context += f"\nUser: {user_input}\nAI: {result}"

        vectorstore = InMemoryVectorStore.from_texts([Message_History],embedding=embeddings,)

        # Use the vectorstore as a retriever
        retriever = vectorstore.as_retriever()

        # Retrieve the most similar text
        retrieved_documents = retriever.invoke(user_input)

        # show the retrieved document's content
        retrieved_documents[0].page_content  

   
if __name__ == "__main__":
    handle_conversation()
