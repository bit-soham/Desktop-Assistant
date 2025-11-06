import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyCgUiHJa9I5vcNSfmVeN1Nm54bxnHZU3QI")

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.5-flash')

# Start a new chat session
chat = model.start_chat()

# Begin the conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Ending the chat.")
        break
    response = chat.send_message(user_input)
    print("Gemini: " + response.text)


