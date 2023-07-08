import numpy as np
import pennylane as qml
from scipy.optimize import minimize

device = qml.device("default.qubit", wires=1)

#objective function
def objective_function(x):
    return x[0]**2 + x[1]**2


@qml.qnode(device)
def circuit(parameters, x):
    qml.RY(parameters[0], wires=0)
    qml.RZ(parameters[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# generative model 
def generative_model(parameters, num_samples):
    samples = []
    values = []

    for _ in range(num_samples):
        x = np.random.uniform(low=-1, high=1, size=2)
        samples.append(x)      

        value = circuit(parameters, x)
        values.append(value)

    return samples, values

# optimize with geo
def optimize_with_geo(num_samples, num_iterations):
    parameters = np.random.uniform(low=-np.pi, high=np.pi, size=2)  

    device = qml.device("default.qubit", wires=1)

    for _ in range(num_iterations):
        samples, values = generative_model(parameters, num_samples)

        # minimum value in generated values
        result = minimize(objective_function, samples[np.argmin(values)], method='Nelder-Mead')

        #QCBM parameters 
        gradient = np.zeros_like(parameters)
        for x, val in zip(samples, values):
            value = circuit(parameters, x)
            gradient += (val - value) * np.gradient(parameters)
        parameters -= gradient * 0.01 

    return result



num_samples = 100
num_iterations = 10

result = optimize_with_geo(num_samples, num_iterations)

print("Minimum found:", result.x)
print("Objective value at the minimum:", result.fun)
