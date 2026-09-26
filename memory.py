import ollama
messages = []
while True:
    user_input = input("You:")
    if user_input.lower() == 'exit':
        print('See you later!')
        break
    else:
        messages.append(
            {"role":"user","content":user_input}
        )
        response = ollama.chat(
            model= 'llama3.2:3b',
            messages=messages
        )
        bot_response = response['message']['content']
        messages.append(
            {"role":"assistant","content":"bot_response"}

        )
        print('Bot:',bot_response)
for message in messages:
    role=message['role']
    content=message['content']
    print(role,content)
        

