import numpy as np

def SDG(train, test, epochs=100, batch_size=1,learning_rate=1):
    m, n = train.shape
    theta = np.random.rand(n)
    costs = []

    for epoch in range(epochs):
        for i in range(train.shape[0]):
            # select all data points
            X = train[i,:].reshape(1, -1)
            Y = test[i]
            pred = np.dot(X, theta)
            errors = pred -Y
            # Compute gradients
            gradients =  X.T.dot(errors)/ batch_size
            step_size =learning_rate * gradients
            theta -= step_size

        predict = train.dot(theta)
        errors = predict - Y
        cost = np.sum(errors**2) / (2 * m)
        costs.append(cost)
    return theta, costs
