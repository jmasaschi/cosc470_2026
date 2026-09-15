# Neural Networks and Deep Learning — Chapter 1 Explained

## A section-by-section guide to Michael Nielsen's “Using neural nets to recognize handwritten digits”

This guide explains Chapter 1 of Michael Nielsen's *Neural Networks and Deep Learning* in plain English, while keeping the mathematics and Python concepts intact.

> **Source:** Michael Nielsen, [Chapter 1: Using neural nets to recognize handwritten digits](https://neuralnetworksanddeeplearning.com/chap1.html)

---

# 1. The Big Picture

The chapter's central question is:

> **How can a computer learn to recognize handwritten numbers without us explicitly programming every rule for what a 0, 1, 2, etc. looks like?**

Nielsen uses handwritten-digit recognition because it is simple enough to understand but difficult enough to demonstrate why neural networks are useful.

A traditional programming approach might try to write rules such as:

- A 0 has a loop.
- A 1 has a vertical line.
- A 2 has a curved top and diagonal section.
- etc.

The problem is that handwriting varies enormously. Trying to manually write all those rules quickly becomes a huge collection of special cases.

A neural network takes a different approach:

**Give it many examples → let it learn the rules itself.**

That is the key idea behind the entire chapter.

---

# 2. Perceptrons

## What is a perceptron?

A perceptron is one of the earliest mathematical models of an artificial neuron.

Imagine several inputs:

```text
x₁ ──┐
x₂ ──┤
x₃ ──┤──> Perceptron ──> output
x₄ ──┘
```

Each input gets a **weight**.

The perceptron calculates:

\[
w_1x_1 + w_2x_2 + \cdots + w_nx_n
\]

and then adds a **bias**:

\[
\sum_j w_jx_j + b
\]

It then applies a threshold:

\[
output =
\begin{cases}
0 & \text{if } \sum_j w_jx_j+b \leq 0\\
1 & \text{if } \sum_j w_jx_j+b > 0
\end{cases}
\]

## What do the weights mean?

A weight controls how important an input is.

For example:

```text
input A → weight 10
input B → weight 1
```

Input A has much more influence on the neuron's decision.

A **positive** weight encourages the neuron to activate.

A **negative** weight discourages it.

## What does the bias do?

The bias shifts the threshold.

Think of it as making the neuron easier or harder to activate.

---

# 3. Why Perceptrons Aren't Quite Enough

The problem with perceptrons is that their output suddenly jumps between 0 and 1.

For example:

```text
input changes slightly
        ↓
output
  0 ──────────────┐
                  │
                  │ sudden jump
                  └────────────── 1
```

Suppose changing one weight by a tiny amount causes the output to jump from 0 to 1.

That makes learning difficult.

We want something more like:

```text
small input change
        ↓
small output change
```

That leads Nielsen to the **sigmoid neuron**.

---

# 4. Sigmoid Neurons

A sigmoid neuron works similarly to a perceptron:

\[
z = \sum_j w_jx_j+b
\]

But instead of immediately converting the result into 0 or 1, it passes the result through the sigmoid function:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The output is always between 0 and 1.

For example:

| z | sigmoid(z) |
|---:|---:|
| -5 | ≈ 0.007 |
| -2 | ≈ 0.119 |
| 0 | 0.5 |
| 2 | ≈ 0.881 |
| 5 | ≈ 0.993 |

So instead of saying:

> "No" or "Yes"

the neuron can effectively say:

> "Very unlikely" → "Maybe" → "Very likely."

## Why is this important?

The sigmoid function is **smooth**.

A small change in a weight produces a small change in the output.

That makes it possible to use calculus to figure out how changing weights affects the network's error.

This is one of the most important ideas in the chapter.

---

# 5. The Architecture of Neural Networks

A neural network is built by connecting neurons into layers.

A simple network looks like:

```text
INPUT            HIDDEN             OUTPUT

x₁ ──────────→   ○ ──────────→
x₂ ──────────→   ○ ──────────→   ○
x₃ ──────────→   ○ ──────────→
x₄ ──────────→   ○
```

There are three important types of layers.

## Input layer

Contains the information we give the network.

For an image, the inputs can be pixel values.

## Hidden layer

Processes the information.

It is called "hidden" simply because it isn't directly the input or output.

## Output layer

Produces the network's answer.

---

# 6. The MNIST Network

Nielsen uses the MNIST handwritten-digit dataset.

Each image is:

\[
28 \times 28 = 784
\]

pixels.

Therefore the input layer has:

**784 neurons.**

Each neuron represents one pixel.

For example:

```text
28 × 28 image

████████████████████████████
██                        ███
██          ████          ███
██        ██    ██        ███
██       ██      ██       ███
██        ██    ██        ███
██          ████          ███
██                        ███
████████████████████████████
```

The pixel values are converted into numbers between 0 and 1.

---

# 7. Why Are There 10 Output Neurons?

The network needs to distinguish between:

```text
0 1 2 3 4 5 6 7 8 9
```

So Nielsen uses **10 output neurons**.

The desired output for a 6 looks approximately like:

```text
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
```

The seventh neuron corresponds to 6 because the outputs are numbered 0 through 9.

If the network produces:

```text
[0.01, 0.02, 0.03, 0.01, 0.05,
 0.04, 0.92, 0.01, 0.02, 0.03]
```

the largest value is at index 6.

Therefore:

**Prediction = 6**

This is a very important connection to remember when looking at neural-network code.

---

# 8. What Might the Hidden Layer Be Learning?

Nielsen gives an intuitive example.

Imagine some hidden neurons learn to detect pieces of a digit.

One neuron might respond strongly to:

```text
  ███
 ██
██
```

Another might detect:

```text
██
 ██
  ███
```

Several of these pieces can combine to represent a larger structure.

The network might therefore learn something conceptually like:

```text
pixels
  ↓
small shapes
  ↓
larger shapes
  ↓
digit
```

This is an important intuition for deep learning.

The network isn't necessarily programmed with the concept of a "curve."

It discovers useful patterns through training.

---

# 9. Why 10 Outputs Instead of 4?

Technically, four binary neurons could represent 16 different possibilities:

\[
2^4=16
\]

So four neurons are mathematically enough to encode the ten digits.

But Nielsen argues that 10 output neurons provide a more useful representation for this particular problem.

The important idea is:

**The representation used by the network can make learning easier or harder.**

For example, asking one neuron to represent a binary bit such as "the most significant bit of the digit" doesn't correspond naturally to visual features.

But asking one neuron:

> "Is this image a 6?"

has a much more natural relationship to the visual patterns in the image.

---

# 10. Learning With Gradient Descent

Now we reach the central part of the chapter.

We have a network.

We have training examples.

But how does the network actually **learn**?

The answer is:

**Change the weights and biases to reduce the network's error.**

To do this, Nielsen introduces a **cost function**.

---

# 11. The Cost Function

The cost function tells us:

> **How bad is the network right now?**

Nielsen uses the quadratic cost:

\[
C(w,b)=\frac{1}{2n}\sum_x ||y(x)-a||^2
\]

This looks intimidating, but the idea is simple.

- \(w\) = all the network's weights
- \(b\) = all the network's biases
- \(x\) = an input training image
- \(y(x)\) = the correct answer
- \(a\) = what the network actually predicted
- \(n\) = number of training examples

The difference:

\[
y-a
\]

tells us how far the prediction was from the desired answer.

Squaring the difference makes errors positive and penalizes larger errors more heavily.

---

# 12. A Simple Cost Example

Suppose the correct answer is:

```text
[0, 0, 1]
```

but the network produces:

```text
[0.1, 0.2, 0.7]
```

The network isn't perfect, but it is fairly close.

The errors are:

```text
0 - 0.1 = -0.1
0 - 0.2 = -0.2
1 - 0.7 =  0.3
```

Square them:

```text
0.01
0.04
0.09
```

The total squared error is:

```text
0.14
```

A perfect prediction would have a cost of zero.

So:

```text
LOW COST  = GOOD
HIGH COST = BAD
```

---

# 13. The Cost Function as a Landscape

One of the best ways to understand gradient descent is to imagine the cost function as a landscape.

Imagine:

```text
          high cost
             /\
            /  \
      _____/    \____
          \      /
           \____/
         low cost
```

The network's weights and biases determine where we are on this landscape.

Our goal is:

> **Find a low point.**

This is what gradient descent does.

---

# 14. Gradient Descent

Imagine standing on a mountain and wanting to reach the bottom.

You look around and determine:

> "Which direction slopes downward?"

Then you take a step in that direction.

Repeat.

That's gradient descent.

For a parameter \(w\), the basic update looks like:

\[
w' = w-\eta\frac{\partial C}{\partial w}
\]

where:

- \(w\) = current weight
- \(C\) = cost
- \(\frac{\partial C}{\partial w}\) = how much the cost changes when \(w\) changes
- \(\eta\) = learning rate

The minus sign is important.

We move **opposite the direction of increasing cost**.

---

# 15. What Does the Derivative Mean?

Suppose:

\[
\frac{\partial C}{\partial w}=0.8
\]

That means increasing \(w\) increases the cost.

So we should decrease \(w\).

If:

\[
\frac{\partial C}{\partial w}=-0.8
\]

then increasing \(w\) decreases the cost.

So we should increase \(w\).

The derivative gives us the direction to move.

---

# 16. The Learning Rate

The learning rate controls how big each step is.

Nielsen calls it:

\[
\eta
\]

Imagine hiking down a mountain.

### Learning rate too small

```text
tiny step
tiny step
tiny step
tiny step
...
```

Learning can take a very long time.

### Learning rate too large

```text
       ↓
    \     /
     \   /
      \ /
       X
      / \
     /   \
```

You can jump back and forth across the minimum.

### Good learning rate

You take reasonably sized steps toward a low-cost area.

---

# 17. Why We Don't Use Every Training Example Every Time

MNIST contains tens of thousands of training examples.

Computing the exact gradient using every example every time would be expensive.

Instead, Nielsen introduces **stochastic gradient descent (SGD)**.

The idea:

1. Take a small group of training examples.
2. Calculate the gradient for that group.
3. Update the weights.
4. Repeat.

The small groups are called **mini-batches**.

For example:

```text
60,000 training examples

        ↓

Mini-batch 1: 10 examples
Mini-batch 2: 10 examples
Mini-batch 3: 10 examples
...
```

This is much more practical.

---

# 18. Epochs

An **epoch** means going through the training data.

If there are 60,000 training examples:

```text
Epoch 1
→ network sees all training data

Epoch 2
→ network sees all training data again

Epoch 3
→ network sees all training data again
```

Nielsen's example uses:

```text
30 epochs
10 examples per mini-batch
learning rate η = 3.0
```

So the network repeatedly improves its weights.

---

# 19. Putting Learning Together

The overall learning process is:

```text
        Training image
              ↓
        Neural network
              ↓
        Prediction
              ↓
       Compare with
       correct answer
              ↓
        Calculate cost
              ↓
      Calculate gradients
              ↓
     Change weights/biases
              ↓
        Try another batch
              ↓
          Repeat
```

That loop is the heart of neural-network training.

---

# 20. Implementing the Network in Python

Nielsen implements the network largely from scratch using NumPy.

The network is represented with something like:

```python
net = network.Network([784, 30, 10])
```

This means:

```text
784 input neurons
      ↓
30 hidden neurons
      ↓
10 output neurons
```

So:

```text
[784, 30, 10]
```

describes the architecture.

---

# 21. The `Network` Class

The network stores things such as:

```python
self.num_layers
self.sizes
self.biases
self.weights
```

The important idea is that the network's **learnable parameters** are:

### Weights

The connections between neurons.

### Biases

The individual neurons' adjustable offsets.

During training, these numbers change.

---

# 22. Feedforward

Once an input image enters the network, the network performs **feedforward**.

For a neuron:

\[
z=wx+b
\]

More generally:

\[
z=Wx+b
\]

Then:

\[
a=\sigma(z)
\]

So the process is:

```text
inputs
  ↓
weighted sum
  ↓
add bias
  ↓
sigmoid
  ↓
activation
```

For multiple layers:

```text
input
  ↓
weighted sum + bias
  ↓
sigmoid
  ↓
hidden activations
  ↓
weighted sum + bias
  ↓
sigmoid
  ↓
output
```

---

# 23. Why NumPy Is Useful

Instead of calculating every neuron one at a time, NumPy allows the network to perform matrix operations.

Conceptually:

\[
a'=\sigma(Wa+b)
\]

This single equation represents an entire layer.

That is one reason understanding basic linear algebra is very useful for machine learning.

---

# 24. MNIST Data Representation

Each image becomes a vector.

A 28 × 28 image:

\[
28\times28=784
\]

becomes:

\[
x=
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
\vdots\\
x_{784}
\end{bmatrix}
\]

The correct digit becomes a 10-dimensional vector.

For example, 6:

\[
y=
\begin{bmatrix}
0\\
0\\
0\\
0\\
0\\
0\\
1\\
0\\
0\\
0
\end{bmatrix}
\]

This is sometimes called a **one-hot encoding**.

---

# 25. What the Network Is Actually Learning

This is worth emphasizing.

The programmer does **not** manually tell the network:

```text
This combination of pixels means 6.
```

Instead, the programmer provides examples:

```text
image → 6
image → 3
image → 8
image → 1
...
```

The network changes its parameters until its outputs increasingly match the correct answers.

So the programmer provides:

- architecture
- training data
- cost function
- learning algorithm
- hyperparameters

The network discovers the actual useful weights.

---

# 26. Training the Network

Nielsen's example is essentially:

```python
net.SGD(
    training_data,
    30,
    10,
    3.0,
    test_data=test_data
)
```

These numbers mean:

```text
training_data
    ↓
30 epochs
    ↓
mini-batches of 10
    ↓
learning rate = 3.0
    ↓
evaluate using test_data
```

---

# 27. What Happens During Training?

Imagine the network starts with essentially random weights.

At first:

```text
Prediction: 8
Correct:    3
```

The error is large.

The gradient tells the network which direction to modify its parameters.

After training:

```text
Prediction: 3
Correct:    3
```

The network gradually finds parameter values that work well.

Importantly, **one update doesn't magically teach the network a concept**.

Learning is incremental.

---

# 28. Learning Rate Problems

Nielsen demonstrates what happens when the learning rate is extremely large.

If:

\[
\eta=100
\]

the network performs terribly.

This illustrates an important practical lesson:

> Neural networks are sensitive to hyperparameters.

The architecture can be reasonable, the data can be correct, and the algorithm can be correct — but bad hyperparameters can still produce terrible results.

---

# 29. Hyperparameters

A hyperparameter is something we choose rather than something the network learns.

Examples include:

- number of hidden neurons
- learning rate
- number of epochs
- mini-batch size
- network architecture

By contrast, weights and biases are learned parameters.

A useful distinction is:

```text
YOU CHOOSE
↓
learning rate
network architecture
epochs
mini-batch size

NETWORK LEARNS
↓
weights
biases
```

---

# 30. Baseline Comparisons

Nielsen asks an important question:

> How good is the neural network compared with something much simpler?

Random guessing gets roughly:

\[
10\%
\]

accuracy on ten possible digits.

The neural network does dramatically better.

He also discusses a surprisingly simple baseline based on the average darkness of each digit.

The broader lesson is:

> Always compare a machine-learning model against a reasonable baseline.

A model isn't impressive merely because its accuracy sounds high.

---

# 31. Toward Deep Learning

The final section asks:

> What is the network actually learning?

Suppose we wanted to recognize a face.

We might imagine breaking the problem into:

```text
pixels
 ↓
edges
 ↓
eyes / nose / mouth
 ↓
face
```

For handwriting:

```text
pixels
 ↓
small edges and curves
 ↓
larger shapes
 ↓
parts of digits
 ↓
digit
```

This naturally leads to multiple layers.

---

# 32. Why Multiple Layers Matter

A shallow network might try to go:

```text
pixels → digit
```

A deeper network can potentially learn:

```text
pixels
   ↓
edges
   ↓
curves
   ↓
shapes
   ↓
digit
```

Each layer can build on the previous layer.

This is the basic intuition behind **deep learning**.

The "deep" in deep learning primarily refers to having multiple layers of learned transformations.

---

# 33. The Most Important Concept in the Chapter

If you remember only one thing, remember this:

> **A neural network is a function with adjustable parameters, and training is the process of adjusting those parameters so the function produces better answers.**

More concretely:

```text
INPUT
  ↓
weights + biases
  ↓
activation functions
  ↓
prediction
  ↓
compare to correct answer
  ↓
calculate error
  ↓
calculate how each parameter affected the error
  ↓
adjust parameters
  ↓
repeat
```

That is the foundation for everything that comes later.

---

# 34. How the Major Concepts Fit Together

Here is the entire chapter in one diagram:

```text
                         TRAINING DATA
                              │
                              ▼
                     ┌─────────────────┐
                     │  Input image    │
                     │    28 × 28      │
                     │   = 784 pixels  │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Input layer     │
                     │ 784 neurons     │
                     └────────┬────────┘
                              │
                         weights
                         + biases
                              │
                              ▼
                     ┌─────────────────┐
                     │ Hidden layer    │
                     │ 30 neurons      │
                     └────────┬────────┘
                              │
                         weights
                         + biases
                              │
                              ▼
                     ┌─────────────────┐
                     │ Output layer    │
                     │ 10 neurons      │
                     └────────┬────────┘
                              │
                              ▼
                       Network guess
                              │
                              ▼
                    Compare with answer
                              │
                              ▼
                         COST FUNCTION
                              │
                              ▼
                          GRADIENT
                              │
                              ▼
                    Adjust weights/biases
                              │
                              └───────────┐
                                          │
                                          ▼
                                      Repeat
```

---

# 35. A Concrete Example

Suppose the input image is a handwritten:

```text
6
```

The image becomes 784 numbers.

Those numbers enter the input layer.

The hidden neurons calculate combinations of those pixels.

For example, conceptually:

```text
pixel patterns
     ↓
edges / curves
     ↓
parts of a 6
     ↓
evidence for "6"
```

The output layer might produce:

```text
0 → 0.01
1 → 0.02
2 → 0.03
3 → 0.04
4 → 0.01
5 → 0.12
6 → 0.91  ← highest
7 → 0.02
8 → 0.03
9 → 0.01
```

The network predicts:

**6**

If the correct answer is 6, the output is good.

If the network instead produced:

```text
6 → 0.20
5 → 0.70
```

the cost would be higher.

Gradient descent then adjusts the network's parameters in a direction intended to make future predictions better.

---

# 36. The Three Things You Should Understand Before Moving On

## 1. Forward pass

The network takes an input and produces an output:

\[
x \rightarrow a
\]

Think:

**"What does the network predict?"**

---

## 2. Cost

The network compares its prediction with the correct answer:

\[
prediction \rightarrow error
\]

Think:

**"How wrong was it?"**

---

## 3. Gradient descent

The network determines how to change its parameters:

\[
error \rightarrow parameter\ updates
\]

Think:

**"What should I change to become less wrong?"**

Together:

\[
\boxed{
\text{Input}
\rightarrow
\text{Prediction}
\rightarrow
\text{Error}
\rightarrow
\text{Adjustment}
\rightarrow
\text{Repeat}
}
\]

That loop is the core of neural-network training.

---

# 37. One Important Note About the Chapter's Age

Nielsen's chapter is an excellent conceptual introduction, but its implementation is intentionally old-fashioned.

The original code uses an older Python/NumPy style and implements the neural network largely from scratch.

Modern machine-learning code would more commonly use frameworks such as:

- PyTorch
- TensorFlow/Keras
- JAX

Those frameworks automate much of the gradient and tensor machinery.

However, **the underlying concepts Nielsen is teaching are still fundamental**:

```text
weights
biases
activation functions
forward propagation
loss/cost
gradients
gradient descent
backpropagation
training data
validation data
test data
```

The next chapter goes much deeper into **backpropagation**, which explains how the network efficiently calculates the gradients needed to update all those weights and biases.

---

# 38. Quick Glossary

| Term | Plain-English meaning |
|---|---|
| Neuron | A mathematical unit that transforms inputs into an output |
| Weight | Controls how strongly an input affects a neuron |
| Bias | Adjustable offset for a neuron |
| Activation | Output produced by a neuron |
| Sigmoid | Function that converts a value into a number between 0 and 1 |
| Layer | Group of neurons at the same stage |
| Input layer | Receives the input data |
| Hidden layer | Intermediate processing layer |
| Output layer | Produces the final prediction |
| Cost/loss | Numerical measure of how wrong the network is |
| Gradient | Tells us how the cost changes with respect to parameters |
| Gradient descent | Method for reducing the cost |
| Learning rate | Controls the size of parameter updates |
| Mini-batch | Small group of training examples |
| Epoch | One pass through the training dataset |
| Training | Adjusting parameters using training examples |
| Parameter | A value learned by the network |
| Hyperparameter | A value chosen by the programmer |
| MNIST | Dataset of handwritten digits |
| One-hot encoding | Representing a category with one active output |
| Feedforward | Moving information from input to output |

---

# 39. Chapter 1 in One Sentence

**Nielsen's Chapter 1 shows how a neural network can take 784-pixel handwritten images, transform them through layers of weighted neurons, compare its 10-digit prediction with the correct answer, and repeatedly adjust its weights and biases using gradient descent until it becomes good at recognizing digits.**

