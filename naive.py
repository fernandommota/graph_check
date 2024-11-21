import ollama
def query_llm(document, claim):
    #llama3.2, mixtral:8x7b
    response = ollama.chat(model='mixtral:8x7b', messages=[
    {
        'role': 'user',
        'content': f'''
            Instructions:
                1. Convert the given the context to a graph. 
                2. Given the context as a graph and not prior knowledge, check if the follow claim is true or false. 
                3. Answer only with 'True' or 'False'. No explanation is needed. Output only 'True' for yes, 'False' for no.
            
            Context information is below.
            ---------------------
            {document}
            ---------------------

            Claim: {claim}
        ''',
    },
    ])
    return response['message']['content']