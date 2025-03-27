# **Caspian - Deep Learning Architectures**

| [**Information**](#information)
| [**Installation**](#installation)
| [**Getting Started**](#getting-started)
| [**Examples**](#examples)
| [**Notes**](#notes)
| [**Future Plans**](#future-plans-and-developments)
|

A flexible deep learning/machine learning research library using [NumPy].




## Information

Caspian is written entirely with base Python and [NumPy], meaning no other library or
framework is required. Contains many basic tools required to create machine learning
models like neural networks, regressions, image processors, and more. 

General structure and functionality inspired by popular frameworks such as [PyTorch] and [TensorFlow].

On top of providing necessary layers and functions, Caspian also allows for simple creation
of new layers and tools that the user may need. Each part of the Caspian architecture has its
own abstraction that the user can inherit from, including:

- `cspn.Layer`
- `cspn.Activation`
- `cspn.PoolFunc`
- `cspn.Loss`
- `cspn.Optimizer`
- `cspn.Scheduler`

Caspian also provides support for [CUDA] parallel processing, using [CuPy] as an optional secondary
import.


## Installation

Before installing, this library requires:

- Python 3.10+
- NumPy v1.23.5+
- CuPy (12x) v13.0.0+ **(Optional)**

```bash
$ pip install git+https://github.com/vexives/caspian
```




## Getting Started

Caspian architectures are split into 6 different class types:

- `Layers`, the backbone behind any model and the main processors & learners.
- `Activations`, non-linear functions which assist layers in learning and processing data.
- `PoolFuncs`, similar to activations, but to be used with pooling layers and work on strided data rather than standard arrays.
- `Losses`, functions which describe the loss, or error, of a model.
- `Optimizers`, functions which assist in layer weight updating and learning.
- `Schedulers`, functions which define the learning rate at a particular step in a model's learning process.


The structure of a network differs slightly from that of [PyTorch] or [TensorFlow], where each layer,
activation, optimizer, and scheduler is separate. With Caspian, layers can contain an activation or pooling function, as well as an optimizer. Optimizers contain a scheduler, which controls the learning rate of the optimizer and layer as a whole. Some layers, like `Dropout` and `Upsampling1D` do not contain optimizers OR activations, as they do not have any learnable parameters or perform any non-linear transformations.

Some types have default classes that allow that operation to be skipped or performed at a base level, like `Linear` for activations, `StandardGD` for optimizers, and `SchedulerLR` for schedulers.
If an optimizer is required for a layer but not provided in the initialization, a default `StandardGD` optimizer with a `SchedulerLR` scheduler will automatically be assigned. Activation and pooling functions will not be defaulted if not provided, so they must be manually provided by the user.


#### GPU Computing

Caspian and its tools can also be used with [CUDA] through [CuPy] to increase speeds by a significant amount. Before importing Caspian or any of its other tools, place this segment of Python code above the other imports:
```python
from caspian.use_cuda import enable_cuda
enable_cuda() 
```
This ensures that all modules and tools from Caspian are synced with [CUDA], and [CUDA]-supported GPU computing should be enabled as long as [CuPy] and the [CUDA] toolkit are both properly installed.


If a custom tool for Caspian is expected to use both CPU and GPU computing, then use this import instead of directly importing [NumPy] or [CuPy]:
```python
from caspian.cudalib import np
```
This will automatically import the library that Caspian is currently using. This allows for easier compatibility and prevents the user from having to manually switch between the two libraries manually within their tool.




## Examples

The setup and training of a model in Caspian is similar to other deep learning libraries of its kind, here is a quick training example of a neural network to provide more information:


### Creation of a Model:

```python
import caspian as cspn
import numpy as np

class NeuralNet(cspn.Layer):
    def __init__(self, inputs: int, hiddens: int, outputs: int, 
                 activation: cspn.Activation, opt: cspn.Optimizer):
        in_size = (inputs,)
        out_size = (outputs,)
        super().__init__(in_size, out_size)

        self.x_1 = cspn.Dense(activation, inputs, hiddens, optimizer=opt.deepcopy())
        self.x_2 = cspn.Dense(activation, hiddens, outputs, optimizer=opt.deepcopy())
        self.softmax = Softmax()

    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        self.training = training
        step_1 = x_1(data, training)
        step_2 = x_2(step_1, training)
        return self.softmax(step_2)

    def backward(self, dx: np.ndarray) -> np.ndarray:
        assert self.training is True
        d_sm = self.softmax.backward(dx)
        d_2 = x_2.backward(d_sm)
        d_1 = x_1.backward(d_1)
        return d_1

    def step(self) -> None:
        x_1.step()
        x_2.step()
```

This is a simple neural network model containing two `Dense` layers, each with the same activation function and optimizer (separate instances are highly recommended) as provided. The variables `in_size` and `out_size` are a part of every layer class, and can be set for a layer using `super().__init__()`, which expects the input size and output size as tuples. If constructed like this, it can also be used inside of `Sequence` layers (similar to [PyTorch]'s [Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)).


### Creation of an Activation Function:

```python
import caspian as cspn
import numpy as np

class ReLU(cspn.Activation):
    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.maximum(0, data)

    def backward(self, data: np.ndarray) -> np.ndarray:
        return (data >= 0) * 1
```

Creating a new activation function is quite simple as well, and only expects two functions, `forward()` and `backward()`, which take and return a [NumPy] array. Activations should return an array of the same size as the input for both functions, and can also have an `__init__()` if any internal variables are necessary. The abstract class `cspn.Activation` also provides default functionality for `__call__()`, which allows it to act like a standard Python function.


### Creation of a Pooling Function:

```python
import caspian as cspn
import numpy as np

class Average(cspn.PoolFunc):
    def forward(self, partition: np.ndarray) -> np.ndarray:
        return np.average(partition)
    
    def backward(self, partition: np.ndarray) -> np.ndarray:
        return partition * (1.0 / partition.size)
```

Similar in structure to activation functions, but pooling functions return an `ndarray` with a singular value rather than an array with the same size as the partition. Like activations as well, can be called like a standard Python function if inheriting from the `PoolFunc` abstract class.


### Creation of a Loss Function:

```python
import caspian as cspn
import numpy as np

class CrossEntropy(cspn.Loss):
    @staticmethod
    def forward(actual: np.ndarray, prediction: np.ndarray) -> float:
        clip_pred = np.clip(prediction, 1e-10, 1 - 1e-10)
        return -np.sum(actual * np.log(clip_pred))

    @staticmethod
    def backward(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        return prediction - actual
```

Loss functions quantify the rate of error of a model's predictions and provides the partial derivative with respect to the output (gradient array) that a model can use to learn. Losses are not a part of any layer or other class, and unless required by some special cases, do not store any internal variables. Because of this, losses can be created as either static classes or instantiable, depending on the user's choice.


### Creation of an Optimizer:
```python
import caspian as cspn
import numpy as np

class Momentum(cspn.Optimizer):
    def __init__(self, momentum: float = 0.9, learn_rate: float = 0.01, 
                 sched: cspn.Scheduler) -> None:
        super().__init__(learn_rate, sched)
        self.momentum = momentum
        self.previous = 0.0

    def process_grad(self, grad: np.ndarray) -> np.ndarray:
        learn_rate = self.scheduler(self.learn_rate)
        velocity_grad = self.momentum * self.previous - learn_rate * grad
        self.previous = velocity_grad
        return velocity_grad
    
    def step(self) -> None:
        self.scheduler.step()
    
    def reset_grad(self) -> None:
        self.previous = 0.0
        self.scheduler.reset()

    def deepcopy(self) -> 'Momentum':
        return Momentum(self.momentum, self.learn_rate, self.scheduler.deepcopy())
```

The general framework for an optimizer is a little bit more complex, but still easy to assemble. The abstract `Optimizer` class initialization takes in two parameters, `learn_rate` as a float, and `sched` as a scheduler class.

The function `process_grad()` is the main transformation of the optimizer. It should process the given gradient array, apply the learning rate (if applicable), and return an array with the same size as the input.

The function `step()` is meant to keep track of the epoch or training iteration of the model that the optimizer is a part of. For the example above, it only calls the internal scheduler's `step()` function and does not modify any variables. However, some more advanced optimizers like `ADAM` may require an internal variable to be kept for this purpose.

Another function expected from optimizers is `reset_grad()`, which clears all previous gradient information and resets the learning rate scheduler for that optimizer.

The function `deepcopy()` is highly recommended if being used on multiple layers of a model, as each layer contains its own version of an optimizer and scheduler. It should pass a deep copy of whatever data structures it contains or needs into the initialization of a new instance.


### Creation of a Learning Rate Scheduler:
```python
import caspian as cspn
import numpy as np

class ConstantLR(cspn.Scheduler):
    def __init__(self, steps: int, const: float = 0.1) -> None:
        self.steps = steps
        self.const = const
        self.epoch = 0

    def __call__(self, init_rate: float) -> float:
        return init_rate * self.const if self.epoch < self.steps else init_rate

    def step(self) -> None:
        self.epoch += 1

    def reset(self) -> None:
        self.epoch = 0

    def deepcopy(self) -> 'ConstantLR':
        return ConstantLR(self.steps, self.const)
```

This is a basic scheduler that multiplies the initial learning rate by a set constant for a specific number of steps. The `__call__()` function is how a scheduler is called to process a learning rate, and is initialized with custom parameters that are unique to that subclass. Similar to how an optimizer is created, schedulers also have `step()`, `reset()`, and `deepcopy()` functions which perform the same operations as described for optimizers above.


### Training and Using a Model:

Now, here's an example on how to create a neural network which can recognize digits from the [MNIST](https://keras.io/api/datasets/mnist/) data set using only Caspian tools:

```python
from keras.datasets import mnist

#Function to test accuracy of model
def get_accuracy(predictions, labels):
    num_accurate = 0
    for i in range(len(new_j)):
        if new_j[i] == ytest[i]:
            num_accurate += 1
    print(f"Accuracy: {(num_accurate/len(new_j))*100}%")

#Load the data
(train_data, x), (test_data, y) = mnist.load_data()
train_data = train_data.reshape(train_data.shape[0], 784)
test_data = test_data.reshape(test_data.shape[0], 784)

train_labels = np.zeros((x.shape[0], x.max()+1), dtype=np.float32)
for i in range(len(train_labels)):
    train_labels[i][int(x[i])] = 1

test_labels = np.zeros((y.shape[0], y.max()+1), dtype=np.float32)
for i in range(len(test_labels)):
    test_labels[i][int(y[i])] = 1

#Constructing the model
optim = cspn.ADAM(learn_rate = 0.005)

l1 = cspn.Dense(ReLU(), 784, 256)
l2 = cspn.Dropout((256,), 0.45)
l3 = cspn.Dense(ReLU(), 256, 256)
l4 = cspn.Dropout((256,), 0.45)
l5 = cspn.Dense(Sigmoid(), 256, 10)
l6 = cspn.Container(Softmax())

Seq1 = cspn.Sequence([l1, l2, l3, l4, l5, l6])
Seq1.set_optimizer(optim)

ent = cspn.CrossEntropy()

#Training on given data
losses = []
for i in range(25):
    for data, label in zip(train_data, train_labels):
        prediction = Seq.forward(data, True)

        loss = ent.forward(label, prediction)
        err = ent.backward(label, prediction)

        Seq1.backward(err)
        Seq1.step()
        
        losses.append(loss)
    print(f"Epoch {i+1} - Loss: {err}")
    losses = []

#Get accuracy of newly trained model
predictions = model.forward(test_data)
predictions = [a.argmax() for a in predictions]
get_accuracy(predictions, test_labels)
```

The example above uses all of the tools that were created to stochastically train a basic neural network that can recognize digits 0 through 9 on a 28x28 size image. Improvements and changes can be made to the model for greater accuracy using other tools in the Caspian library.




## Notes

**It's important to note that this library is still a work in progress, and due to it using very little framework resources, it prioritizes both efficiency and utility over heavy safety. Here are a few things to keep in mind while using Caspian:**

### Memory Safety
> While most functions and classes in this
library are perfectly safe to use and modify, there are some that use unsafe memory operations to greatly increase the speed of that tool. An example of this would be any convolutional or pooling layers, like `Conv1D`, `Conv1DTranspose`, or `Pooling1D`. It is highly recommended for the safety of any machine that uses Caspian, DO NOT modify the internal variables or functions of these unsafe layers. Any memory unsafe layers or functions will contain a warning in their in-line documentation. Changes to necessary variables may create harmful effects such as segmentation faults.

### General Usability
> All classes in this library fit into specific categories of tools that all inherit from a basic abstraction ([**See Above**](#information)) and follow specific functionality guidelines which allow them to work seamlessly with one another. To keep the necessary functionality working as intended, it is encouraged to not modify any variables inside of any class that has already been initialized. Some variables, like the weights of a layer, for instance, may be changed safely as long as the shape and integrity is kept the same.

### Gradient Calculation
> Because [NumPy] does not have any integrated automatic differentiation functionality, all gradient calculations performed by each class is done manually. For any new layers that the user may create, they may use an auto-grad to perform any backwards passes as long as it is compatible with [NumPy].

### Further Compatibility
> Caspian only requires Python and [NumPy], so any other libraries that the user wishes to use alongside it will not be required or affected by Caspian's installation. As mentioned previously in [**Gradient Calculation**](#gradient-calculation), any custom class which inherits from a Caspian abstract container may use any helper libraries or frameworks as long as they are [NumPy] compatible.




## Future Plans and Developments

- Transformer grade layers (Attention, Encoders, Decoders, etc.)
- More activation functions, base layers, and optimizers.
- Improved model saving and loading.
- More utilities, like train/test data splitting, etc.




[NumPy]: https://github.com/numpy/numpy
[CuPy]: https://github.com/cupy/cupy
[PyTorch]: https://github.com/pytorch/pytorch
[TensorFlow]: https://github.com/tensorflow/tensorflow
[CUDA]: https://developer.nvidia.com/cuda-toolkit