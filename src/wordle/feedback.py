#This function takes in the guess and secret word and outputs a list of length 5 where value in each index indicates the color.
#0 - grey, 1 - yellow, 2 - green
def feedback(guess, secret):
    result = [0] * 5
    dictionary = {}

    #Dictionary for secret
    for x in range(5):
        if secret[x] in dictionary:
            dictionary[secret[x]] += 1
        else:
            dictionary[secret[x]] = 1
    
    #Check for green boxes
    for x in range(5):
        if guess[x] == secret[x]:
            result[x] = 2
            dictionary[secret[x]] -= 1

    #Cheak for yellow boxes
    for x in range(5):
        if result[x] == 2:
            continue
        if guess[x] in dictionary and dictionary[guess[x]] > 0:
            result[x] = 1
            dictionary[guess[x]] -= 1

    return tuple(result)

#This functions all the words from the vocabulary that can be the secret by matching the patterns
def candidatesMatching(constraints, vocab):
    result = []

    for w in vocab:
        consistent = True

        for (guess, pattern) in constraints:
            if feedback(guess, w) != pattern:
                consistent = False
                break
        if consistent:
            result.append(w)

    return result

#Test
#print(feedback("ADVAY", "ZDVYG"))
