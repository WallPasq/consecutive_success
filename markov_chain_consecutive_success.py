import numpy as np
from itertools import product
from typing import Any, Type, List, Union

class MarkovChainConsecutiveSuccess:
    """
    A class to model and compute probabilities for consecutive successes in a Markov chain.
    """

    def __init__(self):
        pass

    def fit(self, attempts: int, success_probability: float) -> None:
        """
        Configure the Markov chain parameters.

        Parameters:
        attempts (int): Number of attempts or states in the chain. Must be greater than or equal to 2.
        success_probability (float): Probability of success at each state. Must be between 0.0 and 1.0.

        Raises:
        ValueError: If input types or values are invalid.
        """
        if not isinstance(attempts, int):
            raise ValueError(f"'attempts' must be an int, got {type(attempts).__name__}.")

        if attempts < 2:
            raise ValueError("'attempts' must be greater than or equal to 2.")

        if not isinstance(success_probability, float):
            raise ValueError(f"'success_probability' must be a float, got {type(success_probability).__name__}.")

        if not (0.0 <= success_probability <= 1.0):
            raise ValueError("'success_probability' must be between 0.0 and 1.0.")

        self.attempts = attempts
        self.success_probability = success_probability
        self.failure_probability = 1.0 - success_probability
        self.probabilities_matrix = self._calculate_probabilities_matrix()

    def predict(self, *args: int, exactly_on_last_attempt: Union[bool, List[bool]] = False, exactly_consecutive_success: Union[bool, List[bool]] = False, examples: Union[int, List[int]] = 0) -> float:
        """
        Predict probabilities based on provided conditions.

        Parameters:
        args (int): Number of consecutive successes to predict.
        exactly_on_last_attempt (Union[bool, List[bool]]): Whether the successes must occur on the last attempt.
        exactly_consecutive_success (Union[bool, List[bool]]): Whether to have exactly the number of successes in a row and no more.
        examples (Union[int, List[int]]): Number of examples to generate for each case.

        Returns:
        float: Predicted probabilities.

        Raises:
        ValueError: If input types or values are invalid.
        """
        if any(not isinstance(i, int) or i < 0 or i > self.attempts for i in args):
            raise ValueError(f"Each value in 'args' must be an int between 0 and {self.attempts}.")

        is_list_exactly_on_last_attempt = self._validate_instance_or_list(bool, exactly_on_last_attempt, "exactly_on_last_attempt")
        is_list_exactly_consecutive_success = self._validate_instance_or_list(bool, exactly_consecutive_success, "exactly_consecutive_success")
        is_list_examples = self._validate_instance_or_list(int, examples, "examples")

        if is_list_exactly_on_last_attempt and len(exactly_on_last_attempt) != len(args):
            raise ValueError("The length of 'exactly_on_last_attempt' must match the number of 'args'.")

        if is_list_exactly_consecutive_success and len(exactly_consecutive_success) != len(args):
            raise ValueError("The length of 'exactly_consecutive_success' must match the number of 'args'.")

        if is_list_examples:
            if len(examples) != len(args):
                raise ValueError("The length of 'examples' must match the number of 'args'.")

            if any(i <= 0 for i in examples):
                raise ValueError("Each value in 'examples' must be an int greater than or equal to 0.")
        
        elif examples <= 0:
            raise ValueError("'examples' must be and int greater than or equal to 0.")

        if not is_list_exactly_on_last_attempt:
            exactly_on_last_attempt = [exactly_on_last_attempt] * len(args)

        if not is_list_exactly_consecutive_success:
            exactly_consecutive_success = [exactly_consecutive_success] * len(args)
        
        if not is_list_examples:
            examples = [examples] * len(args)
            
        arguments = zip(args, exactly_on_last_attempt, exactly_consecutive_success, examples)
        predictions = []
        examples = []

        for p, ela, ecs, e in arguments:
            
            if p == 0:
                predictions.append(self.probabilities_matrix[0, 0])

            # I still haven't figured out how to calculate the probability of finishing with exactly p consecutive successes without having achieved them previously.
            elif ela:
                pass
                
            elif ecs and p < self.attempts:
                predictions.append(self.probabilities_matrix[p-1, p] - self.probabilities_matrix[p, p+1])

            else:
                predictions.append(self.probabilities_matrix[p-1, p])

            
            if e:
                examples.append(self._generate_examples(p, ela, ecs, e))
            else:
                examples.append("")

        if any(examples):
            return predictions, examples
        
        return predictions

    def _validate_instance_or_list(self, instance: Type, param: Any, param_name: str) -> bool:
        """
        Validate if a parameter is of a specific type or a list of that type.

        Parameters:
        instance (Type): The type to check against.
        param (Any): The parameter to validate.
        param_name (str): The name of the parameter for error messages.

        Returns:
        bool: True if the parameter is a list, False otherwise.

        Raises:
        ValueError: If validation fails.
        """
        if not isinstance(param, (instance, list)):
            raise ValueError(f"'{param_name}' must be a {instance.__name__} or a list of {instance.__name__}.")

        if isinstance(param, list) and not all(isinstance(b, instance) for b in param):
            raise ValueError(f"All elements in '{param_name}' must be {instance.__name__}.")

        return isinstance(param, list)

    def _create_transition_matrix(self) -> np.ndarray:
        """
        Create the transition matrix for the Markov chain.

        Returns:
        np.ndarray: The transition matrix.
        """        
        transition_matrix = np.zeros((self.attempts, self.attempts+1, self.attempts+1))

        for i in range(self.attempts):
            transition_matrix[i, (i+1), (i+1)] = 1  # Absorbent state.
            transition_matrix[i, :(i+1), 0] = self.failure_probability
            np.fill_diagonal(transition_matrix[i, :, 1:(i+2)], self.success_probability)

        return transition_matrix

    def _calculate_probabilities_matrix(self) -> np.ndarray:
        """
        Calculate the probabilities matrix for the Markov chain.

        Returns:
        np.ndarray: The probabilities matrix.
        """        
        probabilities_matrix = np.zeros((self.attempts+1, self.attempts+1))
        probabilities_matrix[:, 0] = 1

        transition_matrix = self._create_transition_matrix()

        for _ in range(self.attempts):
            probabilities_matrix = np.vstack([probabilities_matrix[i] @ transition_matrix[i] for i in range(self.attempts)])

        return probabilities_matrix

    def _generate_examples(self, consecutive_success: int, exactly_on_last_attempt: bool = False, exactly_consecutive_success: bool = False, examples: int = 0) -> str:
        """
        Generate examples based on the specified parameters.

        Parameters:
        consecutive_success (int): Number of consecutive successes to generate.
        exactly_on_last_attempt (bool): Whether the successes must occur on the last attempt.
        exactly_consecutive_success (bool): Whether to have exactly the number of successes in a row and no more.
        examples (int): Number of examples to generate.

        Returns:
        str: Generated examples.

        Raises:
        ValueError: If input types or values are invalid.
        """
        if not isinstance(consecutive_success, int) or consecutive_success < 0 or consecutive_success > self.attempts:
            raise ValueError(f"'consecutive_success' must be an int between 0 and {self.attempts}.")

        if not isinstance(exactly_on_last_attempt, bool):
            raise ValueError(f"'exactly_on_last_attempt must be a boolean.")

        if not isinstance(exactly_consecutive_success, bool):
            raise ValueError(f"'exactly_consecutive_success must be a boolean.")

        if not isinstance(exactly_consecutive_success, bool):
            raise ValueError(f"'exactly_consecutive_success must be a boolean.")

        if not isinstance(exactly_consecutive_success, bool):
            raise ValueError(f"'exactly_consecutive_success must be a boolean.")

        if not isinstance(examples, int) or examples < 0:
            raise ValueError(f"'examples' must be an int greater than 0.")

        if consecutive_success == 0:
            return "F" * self.attempts

        if consecutive_success == self.attempts:
            return "S" * self.attempts

        all_possibilities = map(lambda x: "".join(x), product("FS", repeat=self.attempts))
        consecutive_success = "S" * consecutive_success
        examples_list = []

        if exactly_on_last_attempt:
            example_generator = filter(lambda x: x.endswith(consecutive_success) and consecutive_success not in x[:-1], all_possibilities)

        elif exactly_consecutive_success:
            example_generator = filter(lambda x: consecutive_success in x and consecutive_success + "S" not in x, all_possibilities)        

        else:
            example_generator = filter(lambda x: consecutive_success in x, all_possibilities)

        for _ in range(examples):
            if len(examples_list) > 0 and examples_list[-1] is None:
                examples_list = examples_list[:-1]
                break

            examples_list.append(next(example_generator, None))

        return examples_list