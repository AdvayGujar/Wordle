def encode(gameState, probes):
    vectorColor = []

    for _, pattern in gameState:
        vectorColor.extend(pattern)

    while len(vectorColor) < probes * 5:
        vectorColor.append(-1)

    return vectorColor
