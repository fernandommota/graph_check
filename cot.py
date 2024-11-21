import ollama

def query_llm(document, claim):
    #llama3.2, mixtral:8x7b
    response = ollama.chat(model='mixtral:8x7b', options={"temperature": 0} ,messages=[
    {
        'role': 'user',
        'content': f'''
            Instructions:
                1. Convert the given the context to a graph. 
                2. Given the context as a graph and not prior knowledge, check if the follow claim is true or false.
                3. Answer only with 'True' or 'False'. No explanation is needed. Output only 'True' for yes, 'False' for no.
            
            Example 01:
                Context information is below.
                ---------------------
                John is male and has 32 years, he is marriage with Mary, female and has 28 years.
                ---------------------
                Claim: Mary has 30 years and is John's sister.
                Generated graph Data from context:
                    * Nodes:
                            + John: A person. Attributes: gender: "male", age: 32
                            + Mary: A person. Attributes: gender: "female", age: 28
                    * Edges:
                            + (John, is married to, Mary)
                Use the graph data to support verify if the claim is true.
                Correct answer for this example: False
            
            Example 02:
                Context information is below.
                ---------------------
                John is male and has 32 years, he is marriage with Mary, female and has 28 years.
                ---------------------
                Claim: Mary has 28 years and is John's wife.
                Generated graph Data from context:
                    * Nodes:
                            + John: A person. Attributes: gender: "male", age: 32
                            + Mary: A person. Attributes: gender: "female", age: 28
                    * Edges:
                            + (John, is married to, Mary)
                Use the graph data to support verify if the claim is true.
                Correct answer for this example: True

            User prompt:
                Context information is below.
                ---------------------
                {document}
                ---------------------
                Claim: {claim}.
        ''',
    },
    ])
    
    return response['message']['content']