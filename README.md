# Fourier-Gradient-Descent
An implementation of the Fourier Series using Gradient Decent

Using a simple gradient descent algorithm to calculate the Fourier series coefficients.

### Overview
The project consists of two main parts:
- #### Fourier Series Approximation:
  This part uses gradient descent to optimize the Fourier series coefficients, allowing the model to learn the best fit for the function.
- #### Manim Visualization
  This part uses Manim to visualize the model's progress over time, showing how the approximation improves as the model learns.

![exp](https://github.com/user-attachments/assets/f1ca67b1-fb0d-4daa-b2d8-60c0cec3ca4a)

### Variables to edit:
* `L` The Width of the approximation (will be approximated periodically); `Default: 5`
* `function(x)` The function to be approximated 
* `NUM_TERMS` Number of terms for the Fourier series; `1 -> Acosx + BsinX` `Default: 20`
* `LEARNING_RATE` Learning Rate! `Default: 0.01` 
* `MAX_ITERATIONS` `Default: 10_000`
* `FILE_NAME` The output file with the coefficients 
