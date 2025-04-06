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
$ pip install caspian-ml
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

Caspian and its tools can also be used with [CUDA] through [CuPy] to increase speeds by a significant amount. Before importing Caspian or any of its tools, place this segment of Python code above the other imports:
```python
import os
os.environ["CSPN_CUDA"] = "cuda"
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
from caspian.layers import Layer, Dense
from caspian.activations import Activation, Softmax
from caspian.optimizers import Optimizer
import numpy as np

class NeuralNet(Layer):
    def __init__(self, inputs: int, hiddens: int, outputs: int, 
                 activation: Activation, opt: Optimizer):
        in_size = (inputs,)
        out_size = (outputs,)
        super().__init__(in_size, out_size)

        self.x_1 = Dense(activation, inputs, hiddens, optimizer=opt.deepcopy())
        self.x_2 = Dense(activation, hiddens, outputs, optimizer=opt.deepcopy())
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
from caspian.activations import Activation
import numpy as np

class ReLU(Activation):
    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.maximum(0, data)

    def backward(self, data: np.ndarray) -> np.ndarray:
        return (data >= 0) * 1
```

Creating a new activation function is quite simple as well, and only expects two functions, `forward()` and `backward()`, which take and return a [NumPy] array. Activations should return an array of the same size as the input for both functions, and can also have an `__init__()` if any internal variables are necessary. The abstract class `cspn.Activation` also provides default functionality for `__call__()`, which allows it to act like a standard Python function.


### Creation of a Pooling Function:

```python
from caspian.pooling import PoolFunc
import numpy as np

class Average(PoolFunc):
    def forward(self, partition: np.ndarray) -> np.ndarray:
        return np.average(partition)
    
    def backward(self, partition: np.ndarray) -> np.ndarray:
        return partition * (1.0 / partition.shape[self.axis])
```

Similar in structure to activation functions, but pooling functions return an `ndarray` with a smaller array rather than an array with the same size as the partition. Like activations as well, can be called like a standard Python function if inheriting from the `PoolFunc` abstract class. Each pooling function will have an internal variable `self.axis` (can be set during initialization) which can be used at any point in both the forward and backward passes.


### Creation of a Loss Function:

```python
from caspian.losses import Loss
import numpy as np

class CrossEntropy(Loss):
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
from caspian.optimizers import Optimizer
from caspian.schedulers import Scheduler
import numpy as np

class Momentum(Optimizer):
    def __init__(self, momentum: float = 0.9, learn_rate: float = 0.01, 
                 sched: Scheduler) -> None:
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
from caspian.schedulers import Scheduler
import numpy as np

class ConstantLR(Scheduler):
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
import numpy as np

from caspian.layers import Conv2D, Pooling2D, Reshape, Dense, Container, Sequence
from caspian.activations import Sigmoid, ReLU, Softmax
from caspian.pooling import Maximum
from caspian.losses import BinCrossEntropy
from caspian.optimizers import StandardGD
from keras.datasets import mnist

#Import the dataset and reshape
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
x_train = np.array(xtrain).reshape(xtrain.shape[0], 784)
x_test = np.array(xtest).reshape(xtest.shape[0], 784)
y_train = np.zeros((ytrain.shape[0], ytrain.max()+1), dtype=np.float32)

for i in range(len(y_train)):
    y_train[i][int(ytrain[i])] = 1

xt = x_train.reshape(-1, 60, 1, 28, 28) / 255.0
yt = y_train.reshape(-1, 60, 10)
print(xt.shape)
print(yt.shape)

#Create the model to be trained
optim = StandardGD(0.001)

d1 = Conv2D(Sigmoid(), 32, 3, (1, 28, 28))
d2 = Pooling2D(Maximum(), 2, (32, 26, 26), 2)
d3 = Conv2D(Sigmoid(), 12, 3, (32, 13, 13))
d4 = Pooling2D(Maximum(), 2, (12, 11, 11), 2)
d5 = Reshape((-1, 12, 5, 5), (-1, 12*5*5))
d6 = Dense(ReLU(), 12*5*5, 100)
d7 = Dense(Sigmoid(), 100, 10)
d8 = Container(Softmax())

Seq1 = Sequence([d1, d2, d3, d4, d5, d6, d7])
Seq1.set_optimizer(optim)

ent = BinCrossEntropy()

#Training
losses = 0.0
for ep in range(50):
    for x, y in zip(xt, yt):
        x_r = Seq1.forward(x, True)

        err_grad = ent.backward(y, x_r)

        loss = ent.forward(y, x_r)

        Seq1.backward(err_grad)
        Seq1.step()

        losses += loss
    print(f"Epoch {ep+1} - {losses / xt.shape[0]}")
    losses = 0.0
```

The example above uses all of the tools that were created to stochastically train a basic neural network that can recognize digits 0 through 9 on a 28x28 size image. Improvements and changes can be made to the model for greater accuracy using other tools in the Caspian library.


### Saving and Loading Layers:

> [!NOTE]
> Saving and loading models may change in the future at an unknown time. In the event that it is changed, previously formatted `.cspn` files will no longer work with new ones. If this occurs, then it will be specified in the update that does so.

Once a model has been trained (or in the process of training), each layer can be exported and loaded at a different time. Layers, activations, pooling functions, optimizers, and schedulers all have methods which allow them to be encoded into strings and/or saved to files (of type `.cspn`). 


#### Saving

Layers can be encoded into a string or saved to a file using the `save_to_file()` method, as shown here:
```python
d1 = Conv2D(ReLU(), 32, 3, (1, 28, 28))
d1.save_to_file("layer1.cspn")
```
If the file name is not specified and no parameters are given to this method, then a string is returned which contains the information of that layer. This includes the activation or pooling function, optimizer, and scheduler of that layer (if any are applicable).

For other tools like optimizers, schedulers, or functions, the `repr()` function is used in place of a set saving method. It returns a string with the name of the class and all initialized attributes of the object in the order of the initialization function with `/` as a separator (except for schedulers, which use `:`). A quick example:
```python
opt = ADAM(learn_rate = 0.001, sched = StepLR(10))
opt_info = repr(opt)
#Returns "ADAM/0.9/0.99/1e-8/0.001/StepLR:10:0.1"
```


#### Loading

Once a layer has been saved to a file or encoded in a string, it can be re-loaded and re-instantiated from where it was saved before. Each layer has a static `from_save()` method, which takes two parameters. The first is a string `context`, which is either the name of the file to be loaded from or the encoded string containing the appropriate information. The second is a boolean `file_load`, which determines whether the context is either a file name or the encoded string itself. To use the method on a file:
```python
new_layer = Conv2D.from_save("layer1.cspn", True)
```
If the file provided is incorrectly formatted/modified or the file imported is not an appropriate `.cspn` file, an exception is thrown instead.


For all other saveable tools in the Caspian library, each tool folder has a function which takes the `repr` string and returns a class instance of the encoded object. The functions that correspond to each class type include:

- `Activations` -> `activations.parse_act_info()`
- `Optimizers` -> `optimizers.parse_opt_info()`
- `Pooling` -> `pooling.parse_pool_info()`
- `Schedulers` -> `schedulers.parse_sched_info()`

These classes do not have options to save directly to a file, but the user can export them and import them manually if absolutely needed. If the user creates a custom sub-class and wishes to save or load them, they will need to create an appropriate `repr()` following the same procedure as outlined above, and add the class to the tool folder dictionary:

- `Activations` -> `activations.act_funct_dict`
- `Optimizers` -> `optimizers.opt_dict`
- `Pooling` -> `pooling.pool_funct_dict`
- `Schedulers` -> `schedulers.sched_dict`

Loading a class in these categories will look similar to below:
```python
from caspian import activations as act

class CustomFunct(act.Activation):
    ...
    def __repr__():
        ...

#Create instance
a_1 = CustomFunct(...)
saved_str = repr(a_1)

#Load from context string
act.act_funct_dict["CustomFunct"] = CustomFunct
a_2 = act.parse_act_info(saved_str)
```




## Notes

**It's important to note that this library is still a work in progress, and due to it using very little framework resources, it prioritizes both efficiency and utility over heavy safety. Here are a few things to keep in mind while using Caspian:**

### Memory Safety

> [!CAUTION]
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