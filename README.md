# README in English

Welcome to this repository, which provides an implementation of a Markov Chain to analyze consecutive successes in a series of attempts.  
This code allows you to estimate the probability of having \(p\) consecutive successes (e.g., \(p\) consecutive wins) in a user-defined number of attempts. In addition, it can generate concrete examples of success/failure sequences, making it easier to understand and practically validate the results.

## Main Features

1. **Configuring the Markov Chain Parameters**  
   Using the `fit(attempts: int, success_probability: float)` method, you specify how many attempts (states) will be evaluated and the probability of success at each attempt.

2. **Probability Prediction**  
   The `predict(*args: int, exactly_on_last_attempt, exactly_consecutive_success, examples)` method allows you to:
   - Calculate the probability of having \(p\) consecutive successes (`args`) across a range of scenarios.
   - Indicate whether those successes must occur exactly on the last attempt (`exactly_on_last_attempt`).
   - Define whether the number of consecutive successes must occur precisely as specified (with no larger occurrences) (`exactly_consecutive_success`).
   - Generate concrete examples of “S” (success) and “F” (failure) sequences as needed (`examples`).

3. **Example Generation**  
   The `_generate_examples` method demonstrates how success/failure sequences could look in practice, simplifying hypothesis verification and model behavior analysis.

## Usage Example

```python
from consecutive_success import MarkovChainConsecutiveSuccess

# Instantiate the class
model = MarkovChainConsecutiveSuccess()

# Set up the model for 5 attempts, each with a 0.3 probability of success
model.fit(attempts=5, success_probability=0.3)

# We want the probability of exactly 2 consecutive successes,
# not larger than 2, generating 3 examples
predictions, examples = model.predict(
    2,
    exactly_on_last_attempt=False, 
    exactly_consecutive_success=True, 
    examples=3
)

print("Predictions:", predictions)
print("Examples:", examples)
```

**Expected Output**:
```
Predictions: [0.20852999999999997]
Examples: [['FFFSS', 'FFSSF', 'FSFSS']]
```

## Points of Attention and Possible Improvements

This code is still under development, and any collaboration is greatly appreciated.  
Here are three main aspects that need enhancement:

1. **Calculating the Probability of Exactly p Consecutive Successes Only on the Last Attempt**  
   - Currently, the `predict` method does not have a specific logic to handle scenarios in which we want to know the probability of having `p` consecutive successes only at the last attempt (`exactly_on_last_attempt`). Implementing such a calculation would add value to the library.

2. **Matrix Operations Optimization**  
   - The code can be optimized to handle matrices more efficiently, especially in cases involving sparse matrices. Such optimization can greatly reduce memory usage and processing time for a large number of attempts.

3. **Enabling Parallel Processes and GPU Optimization**  
   - Some calculations, such as large matrix multiplication or example generation, could benefit from parallelism and GPU usage. This is particularly relevant when there is a large number of states or more advanced computational requirements.

## How to Contribute

1. Fork this repository.  
2. Create a branch for your feature or bug fix:  
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Commit your changes and push to GitHub:  
   ```bash
   git commit -m "feat: Add my new feature"
   git push origin feature/my-new-feature
   ```
4. Open a pull request describing your contributions and wait for feedback.

Feel free to open issues or propose improvements of any kind. All help is welcome to make this project more comprehensive and robust!

---

**License**  
This project is available under the [Apache License 2.0](LICENSE). Feel free to use and modify it as needed. We appreciate any contributions to further development!
