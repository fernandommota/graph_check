def get_prediction_and_confidence(id, response):
    if response[0:50].lower().find("true") >= 0:
        predicted = 1
        confidence = 1
        print(f'result for {id}', response[0:50])
    elif response[0:50].lower().find("false") >= 0:
        predicted = 0
        confidence = 1
        print(f'result for {id}', response[0:50])
    else:
        predicted = 0
        confidence = 0
        print(f'result for {id}', response.strip()[0:50])
    
    return predicted, confidence