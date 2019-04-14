# Ref : https://www.linkedin.com/pulse/neural-network-ai-simple-so-stop-pretending-you-genius-brandon-wirtz/?trk=eml-email_feed_ecosystem_digest_01-recommended_articles-12-Unknown&midToken=AQG1Bq6nLmQRWw&fromEmail=fromEmail&ut=1aMpt58zH2Xo41

import numpy as np


X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,0]]).T

weights0 = 2*np.random.random((3,4)) - 1
weights1 = 2*np.random.random((4,1)) - 1

for j in range(10000):
    # Forward propogation
    layer1 = 1/(1 + np.exp(-(np.dot(X,weights0))))
    layer2 = 1 / (1 + np.exp(-(np.dot(layer1, weights1))))
    # Back propogation
    layer2_diff = ( y - layer2) * (layer2*(1 - layer2))
    layer1_diff = layer2_diff.dot(weights1.T) * (layer1 * (1 - layer1))
    weights1 += layer1.T.dot(layer2_diff)
    weights0 += X.T.dot(layer2_diff)

XX = np.array([[0,0,1],[1,1,1]])
layer1 = 1/(1 + np.exp(-(np.dot(XX,weights0))))
layer2 = 1 / (1 + np.exp(-(np.dot(layer1, weights1))))
print(layer2)