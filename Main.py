from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
import pickle
import json
# Vector import
from langchain_core.vectorstores import InMemoryVectorStore

#embedding 
template = """
Conversation history: {context} 
Name(User): {name}
Question: {question}
About you(The AI): {AI_introduction}

Answer: 
"""

#database
db = {}
db ['template'] = template

#binary mode
dbfile = open('examplePickle', 'ab')

# source, destination
pickle.dump(db, dbfile)                    
dbfile.close()

#LL model
model = OllamaLLM(model="llama3", stream=True)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

    
def handle_conversation():
    context = ""
    nameuser = input("What name would you like to be called?")
    AI_introduction = input("Explain your AI: ")
    print("Type in the AI on what you want: ")
    while True:
        user_input = input("you: ")
        if user_input.lower() == "exit":
            break

        result = chain.invoke(
                {
                    "AI_introduction": AI_introduction, 
                    "name": nameuser, 
                    "context": context, 
                    "question": user_input
                }
            )
        
        print("Bot: ", result)
        
        context += f"\nUser: {user_input}\nAI: {result}"


if __name__ == "__main__":

    handle_conversation()
