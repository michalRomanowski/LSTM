import numpy as np
import math

class LSTM:
    def __init__(self, cell_state_size: int, hidden_state_size: int, x_size: int):
        self.__cell_state = np.zeros(cell_state_size)
        self.__hidden_state = np.zeros(hidden_state_size)
        self.__forget_gate = ForgetGate(cell_state_size, hidden_state_size, x_size)
        self.__input_gate = InputGate(cell_state_size, hidden_state_size, x_size)
        self.__output_gate = OutputGate(cell_state_size, hidden_state_size, x_size)
        
    def process(self, x: np.ndarray) -> np.ndarray:
        self.__hidden_state = np.concatenate([x, self.__hidden_state])
        
        forget_gate_output = self.__forget_gate.process(self.__hidden_state)
        self.__cell_state *= forget_gate_output
        
        input_gate_output = self.__input_gate.process(self.__hidden_state)
        self.__cell_state += input_gate_output
        
        output_gate_output = self.__output_gate.process(self.__cell_state, self.__hidden_state)
        self.__hidden_state = output_gate_output
        
        return self.__hidden_state
    
    def process_sequence(x: np.ndarray):
        pass
        
class ForgetGate():
    def __init__(self, cell_state_size: int, hidden_state_size: int, x_size: int):
        self.__neural_net = NeuralNetFactory.get_sigmoid_neural_net(
            hidden_state_size + x_size, cell_state_size)
    
    def process(self, x: np.ndarray) -> np.ndarray:
        return self.__neural_net.think(x)
        
class InputGate():
    def __init__(self, cell_state_size: int, hidden_state_size: int, x_size: int):
        self.__sigmoid_neural_net = NeuralNetFactory.get_sigmoid_neural_net(
            hidden_state_size + x_size, cell_state_size)
        
        self.__tanh_neural_net = NeuralNetFactory.get_tanh_neural_net(
            hidden_state_size + x_size, cell_state_size)
    
    def process(self, x: np.ndarray) -> np.ndarray:
        sigmoid_output = self.__sigmoid_neural_net.think(x)
        tanh_output = self.__tanh_neural_net.think(x)
        
        return sigmoid_output * tanh_output
    
class OutputGate():
    def __init__(self, cell_state_size: int, hidden_state_size: int, x_size: int):
        self.__sigmoid_neural_net = NeuralNetFactory.get_sigmoid_neural_net(
            hidden_state_size + x_size, hidden_state_size)
        
        self.__tanh_neural_net = NeuralNetFactory.get_tanh_neural_net(
            cell_state_size, hidden_state_size)
    
    def process(self, cell_state_x: np.ndarray, hidden_state_x: np.ndarray) -> np.ndarray:
        sigmoid_output = self.__sigmoid_neural_net.think(hidden_state_x)
        tanh_output = self.__tanh_neural_net.think(cell_state_x)
        
        return sigmoid_output * tanh_output
    
class NeuralNetFactory:
    def get_tanh_neural_net(input_size: int, output_size: int):
        return NeuralNet(input_size, output_size, lambda x : math.tanh(x))
    
    def get_sigmoid_neural_net(input_size: int, output_size: int):
        return NeuralNet(input_size, output_size, lambda x : (1 / (1 + math.exp(-x))))
    
class NeuralNet:
    def __init__(self, input_size: int, output_size: int, activation_function: callable):
        self.neurons = [Neuron(input_size, activation_function) for x in range(output_size)]
    
    def think(self, x: np.ndarray) -> np.ndarray:
        return np.array([neuron.think(x) for neuron in self.neurons], dtype=float)
    
class Neuron:
    def __init__(self, input_size: int, activation_function: callable):
        self.input_weights = np.zeros(input_size)
        self.activation_function = activation_function
        self.bias = 0.0
    
    def think(self, x: np.ndarray) -> float:
        net = (x * self.input_weights).sum() + self.bias
        return self.activation_function(net + self.bias)
        
tmp = LSTM(6, 7, 4)
x = np.array([1, 2, 3, 4])
tmp.process(x)
