# Import the openai library, which is used to interact with the OpenAI GPT-3 API.
import openai

# Import the configuration file (most likely containing the API key and other settings).
import config

# Set the API key for the openai library from the configuration.
openai.api_key = config.DevelopmentConfig.OPENAI_KEY

# Define a function to generate a chatbot response based on a given prompt.
def generateChatResponse(prompt):
    
    # Initialize an empty list called 'messages'.
    messages = []
    
    # Append a system message to set the context for the chat. Here, the assistant's name is set to "Karabo".
    messages.append({"role": "system", "content": "Your name is Karabo. You are a helpful assistant."})

    # Create a dictionary for the user's message, using the provided prompt.
    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)

    # Use the OpenAI API to generate a response based on the 'messages' (context + user's message).
    # The model used for this purpose is "gpt-3.5-turbo".
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    # Try to extract the generated answer from the response.
    # If successful, replace newline characters with '<br>' for better HTML formatting.
    try:
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
    except:
        # If there's any error in extracting the response, return an error message to the user.
        answer = 'Oops you beat the AI, try a different question, if the problem persists, come back later.'

    # Return the generated or error message.
    return answer
