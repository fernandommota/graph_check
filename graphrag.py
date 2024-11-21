import os

def get_graph_from_llm(ollama, model, options, document, graph_path):
    if os.path.isfile(graph_path) is True:
        f = open(graph_path, "r")
        return f.read() 
    else:
        #llama3.2, mixtral:8x7b
        response = ollama.chat(model=model, options=options ,messages=[
        {
            'role': 'user',
            'content': f'''
                Instructions:
                    1. Convert the given document to a graph. 
                    3. Answer only with the graph data. No explanation is needed.
                
                Example:
                    Document is below.
                    ---------------------
                    John is male and has 32 years, he is marriage with Mary, female and has 28 years.
                    ---------------------
                    output:
                        * Nodes:
                                + John: A person. Attributes: gender: "male", age: 32
                                + Mary: A person. Attributes: gender: "female", age: 28
                        * Edges:
                                + (John, is married to, Mary)

                User prompt:
                    Document is below.
                    ---------------------
                    {document}
                    ---------------------
            ''',
        },
        ])

        output_folder = "/".join(graph_path.split("/")[0:-1])
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        f = open(graph_path, "a")
        f.write(response['message']['content'])
        f.close()
        
        return response['message']['content']


def query_graph_llm(ollama, model, options, graph, claim):
    response = ollama.chat(model=model, options=options ,messages=[
    {
        'role': 'user',
        'content': f'''
            Instructions:
                1. Given the context as a graph and not prior knowledge, check if the follow claim is true or false.
                3. Answer only with 'True' or 'False'. No explanation is needed. Output only 'True' for yes, 'False' for no.
            
            Example 01:
                Context information is below.
                ---------------------
                * Nodes:
                        + John: A person. Attributes: gender: "male", age: 32
                        + Mary: A person. Attributes: gender: "female", age: 28
                * Edges:
                        + (John, is married to, Mary).
                ---------------------
                Claim: Mary has 30 years and is John's sister.
                Correct answer for this example: False
            
            Example 02:
                Context information is below.
                ---------------------
                * Nodes:
                        + John: A person. Attributes: gender: "male", age: 32
                        + Mary: A person. Attributes: gender: "female", age: 28
                * Edges:
                        + (John, is married to, Mary)
                ---------------------
                Claim: Mary has 28 years and is John's wife.
                Correct answer for this example: True

            User prompt:
                Context information is below.
                ---------------------
                {graph}
                ---------------------
                Claim: {claim}.
        ''',
    },
    ])
    
    return response['message']['content']


def query_document_and_graph_llm(ollama, model, options, document, graph, claim):
    prompt_content = f'''
Instructions:
    1. Given the two types of context: document and graph and not prior knowledge, check if the follow claim is true or false.
    3. Answer only with 'True' or 'False'. No explanation is needed. Output only 'True' for yes, 'False' for no.

Example 01:
    Context information is below.
    ---------------------
    
    Document: 
John is male and has 32 years, he is marriage with Mary, female and has 28 years.
    
    Graph:
* Nodes:
    + John: A person. Attributes: gender: "male", age: 32
    + Mary: A person. Attributes: gender: "female", age: 28
* Edges:
    + (John, is married to, Mary).
    
    ---------------------

Claim: Mary has 30 years and is John's sister.

Correct answer for this example: False

Example 02:
    Context information is below.
    ---------------------
    
    Document: 
John is male and has 32 years, he is marriage with Mary, female and has 28 years.
    
    Graph:
* Nodes:
        + John: A person. Attributes: gender: "male", age: 32
        + Mary: A person. Attributes: gender: "female", age: 28
* Edges:
        + (John, is married to, Mary)
    
    ---------------------

Claim: Mary has 28 years and is John's wife.

Correct answer for this example: True

User prompt:
    Context information is below.
    ---------------------
    
    Document: 
{document}
    
    Graph:
{graph}

    ---------------------
    Claim: {claim}
        '''
    #print(prompt_content)
    response = ollama.chat(model=model, options=options ,messages=[
    {
        'role': 'user',
        'content': prompt_content,
    },
    ])
    
    return response['message']['content']