from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Free tier model
    temperature=0.7
)

# Create a simple message
messages = [HumanMessage(content="Hello, Gemini!")]

# Get response
response = llm.invoke(messages)
print(response.content)
