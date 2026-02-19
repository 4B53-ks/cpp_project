# MNIST Neural Network (C++ Edition)

## What is this project?

This is a **neural network** that learns to read handwritten numbers (0-9). It's a beginner project to learn how machine learning works!

The project uses the **MNIST dataset** - a famous collection of 70,000 handwritten digit images.

## What files do what?

| File | What it does |
|------|-------------|
| `mnist_nn.cpp` | The main program that **trains** the neural network (learns from examples) |
| `mnist_nn_test.cpp` | Tests how good the trained network is (was made with help of LLM's) |
| `mnist_model.txt` | Saves the trained model so you don't have to train again |
| `train-images.idx3-ubyte` | 60,000 training images of handwritten numbers |
| `train-labels.idx1-ubyte` | The correct answers for the training images |
| `t10k-images.idx3-ubyte` | 10,000 test images to check if it works |
| `t10k-labels.idx1-ubyte` | The correct answers for the test images |

## How to use it

### Step 1: Compile the code
```bash
g++ mnist_nn.cpp -o mnist_nn
g++ mnist_nn_test.cpp -o mnist_nn_test
```

### Step 2: Train the network
```bash
./mnist_nn train-images.idx3-ubyte train-labels.idx1-ubyte t10k-images idx3-ubyte t10k-labels.idx1-ubyte
```
This will learn from the training images and save a file called `mnist_model.txt`.

### Step 3: Test the network
```bash
./mnist_nn_test mnist_model.txt t10k-images.idx3-ubyte t10k-labels.idx1-ubyte
```
This will check how many test images it can correctly identify.

## How does it work? (Simple version)

1. **Load images**: Read 784 pixel values per image (28×28 pixels)
2. **Train**: Show the network thousands of examples and let it learn patterns
3. **Test**: Try the network on new images it's never seen

## What to expect

- Training should take a few seconds to a few minutes
- You should get about **95-97% accuracy** on the test set
- Higher accuracy = better at recognizing handwritten numbers


**"Cannot open file" error?**
- Make sure all `.idx3-ubyte` and `.idx1-ubyte` files are in the same folder as the code


## Next steps to learn more

- Change the network structure (add more layers, neurons)
- Try different learning rates
- Visualize what the network learned
- Read about backpropagation and gradient descent
