import pytest
from numpy import isclose
from consecutive_success import MarkovChainConsecutiveSuccess

def test_fit_invalid_attempts_type():
    model = MarkovChainConsecutiveSuccess()
    with pytest.raises(TypeError, match="'attempts' must be an int, got str."):
        model.fit(attempts="2", success_probability=0.8)

def test_fit_attempts_less_than_two():
    model = MarkovChainConsecutiveSuccess()
    with pytest.raises(ValueError, match="'attempts' must be greater than or equal to 2."):
        model.fit(attempts=1, success_probability=0.8)

def test_fit_invalid_success_probability_type():
    model = MarkovChainConsecutiveSuccess()
    with pytest.raises(TypeError, match="'success_probability' must be a float, got str."):
        model.fit(attempts=2, success_probability="0.8")

def test_fit_success_probability_out_of_range():
    model = MarkovChainConsecutiveSuccess()
    with pytest.raises(ValueError, match="'success_probability' must be between 0.0 and 1.0."):
        model.fit(attempts=2, success_probability=1.5)

def test_non_integer_args():
    model = MarkovChainConsecutiveSuccess()
    with pytest.raises(TypeError, match="Each value in 'args' must be an int."):
        model.fit(attempts=3, success_probability=0.8)
        model.predict("1")

def test_args_values_out_of_bounds():
    model = MarkovChainConsecutiveSuccess()
    with pytest.raises(ValueError, match="Each value in 'args' must be between 0 and 3."):
        model.fit(attempts=3, success_probability=0.8)
        model.predict(4)

def test_mismatched_len_of_exactly_on_last_attempt():
    model = MarkovChainConsecutiveSuccess()
    with pytest.raises(ValueError, match="The length of 'exactly_on_last_attempt' must match the number of 'args'."):
        model.fit(attempts=3, success_probability=0.8)
        model.predict(1, 2, exactly_on_last_attempt=[True], exactly_consecutive_success=[False, True], examples=[2, 3])

def test_mismatched_len_of_exactly_consecutive_success():
    model = MarkovChainConsecutiveSuccess()
    with pytest.raises(ValueError, match="The length of 'exactly_consecutive_success' must match the number of 'args'."):
        model.fit(attempts=3, success_probability=0.8)
        model.predict(1, 2, exactly_on_last_attempt=[True, False], exactly_consecutive_success=[False], examples=[2, 3])

def test_mismatched_len_of_examples():
    model = MarkovChainConsecutiveSuccess()
    with pytest.raises(ValueError, match="The length of 'examples' must match the number of 'args'."):
        model.fit(attempts=3, success_probability=0.8)
        model.predict(1, 2, exactly_on_last_attempt=[True, False], exactly_consecutive_success=[False, True], examples=[2])

def test_invalid_types_in_optional_parameters():
    model = MarkovChainConsecutiveSuccess()
    with pytest.raises(TypeError, match="'exactly_on_last_attempt' must be a bool or a list of bool."):
        model.fit(attempts=3, success_probability=0.8)
        model.predict(1, exactly_on_last_attempt="True")

def test_incorrect_number_of_examples():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=3, success_probability=0.8)
    with pytest.raises(ValueError, match="'examples' must be an int greater than or equal to 0."):
        model.predict(2, examples=-1)

def test_incorrect_number_of_examples_in_a_list():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=3, success_probability=0.8)
    with pytest.raises(ValueError, match="Each value in 'examples' must be an int greater than or equal to 0."):
        model.predict(1, 2, examples=[1, -1])

def test_incorrect_type_of_examples_in_a_list():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=3, success_probability=0.8)
    with pytest.raises(TypeError, match="All elements in 'examples' must be int."):
        model.predict(1, 2, examples=[1, "1"])

def test_probability_of_three_attempts_and_two_consecutive_success():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=3, success_probability=0.8)
    predict = model.predict(2)
    assert isclose(predict, 0.768), "The probability of 3 attempts and 2 consecutive success should be approximately 0.768."

def test_probability_of_three_attempts_and_two_consecutive_success_exactly_on_last_attempt():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=3, success_probability=0.8)
    predict = model.predict(2, exactly_on_last_attempt=True)
    assert isclose(predict, 0.128), "The probability of 3 attempts and 2 consecutive success exactly on last attempt should be approximately 0.128."
    
def test_probability_of_three_attempts_and_exactly_two_consecutive_success_and_none_more():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=3, success_probability=0.8)
    predict = model.predict(2, exactly_consecutive_success=True)
    assert isclose(predict, 0.256), "The probability of 3 attempts and 2 exactly consecutive success (and none more) should be approximately 0.256."
    
def test_probability_of_three_attempts_and_two_consecutive_success_and_three_consecutive_success():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=3, success_probability=0.8)
    predicts = model.predict(2, 3)
    assert isclose(predicts[0], 0.768), "The probability of 3 attempts and 2 consecutive success should be approximately 0.768."
    assert isclose(predicts[1], 0.512), "The probability of 3 attempts and 3 consecutive success should be approximately 0.512."
    
def test_probability_of_five_attempts_and_two_consecutive_success_exactly_on_last_attempt_and_three_consecutive_success():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=5, success_probability=0.8)
    predicts = model.predict(2, 3, exactly_on_last_attempt=[True, False])
    assert isclose(predicts[0], 0.04608), "The probability of 5 attempts and 2 consecutive success exactly on last attempt should be approximately 0.04608."
    assert isclose(predicts[1], 0.7168), "The probability of 5 attempts and 3 consecutive success should be approximately 0.7168."
    
def test_probability_of_five_attempts_and_two_consecutive_success_and_exactly_three_consecutive_success_and_none_more():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=5, success_probability=0.8)
    predicts = model.predict(2, 3, exactly_consecutive_success=[False, True])
    assert isclose(predicts[0], 0.94208), "The probability of 5 attempts and 2 consecutive success should be approximately 0.94208."
    assert isclose(predicts[1], 0.22528), "The probability of 5 attempts and exactly 3 consecutive success (and none more) should be approximately 0.22528."

def test_examples_of_three_attempts_and_zero_consecutive_success():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=3, success_probability=0.8)
    _, examples = model.predict(0, examples=3)
    assert examples[0] == ["FFF"], "The examples generated for 3 attempts and 0 consecutive successes should be ['FFF']."
    
def test_examples_of_three_attempts_and_three_consecutive_success():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=3, success_probability=0.8)
    _, examples = model.predict(3, examples=3)
    assert examples[0] == ["SSS"], "The examples generated for 3 attempts and 3 consecutive successes should be ['SSS']."

def test_examples_of_eight_attempts_and_different_number_of_attempts_and_optional_arguments():
    model = MarkovChainConsecutiveSuccess()
    model.fit(attempts=8, success_probability=0.8)
    _, examples = model.predict(
        *range(1, 8, 1),
        exactly_on_last_attempt=[True, False, True, False, False, True, False],
        exactly_consecutive_success=[False, True, False, True, True, False, False],
        examples=[2, 2, 3, 3, 2, 3, 1]
    )
    
    assert examples[0] == ["FFFFFFFS"], "The examples generated for 8 attempts and 1 consecutive success exactly on last attempt should be ['FFFFFFFS']."
    assert len(examples[1]) == 2, "The number of examples generated for 8 attempts and 2 consecutive successes should be 2."
    assert examples[1] == ["FFFFFFSS", "FFFFFSSF"], "The examples generated for 8 attempts and exactly 2 consecutive successes (and none more) should be ['FFFFFFSS', 'FFFFFSSF']."
    assert len(examples[2]) == 3, "The number of examples generated for 8 attempts and 3 consecutive successes should be 3."
    assert examples[2] == ["FFFFFSSS", "FFFSFSSS", "FFSFFSSS"], "The examples generated for 8 attempts and 3 consecutive successes exactly on last attempt should be ['FFFFFSSS', 'FFFSFSSS', 'FFSFFSSS']."
    assert len(examples[3]) == 3, "The number of examples generated for 8 attempts and 4 consecutive successes should be 3."
    assert examples[3] == ["FFFFSSSS", "FFFSSSSF", "FFSFSSSS"], "The examples generated for 8 attempts and exactly 4 consecutive successes (and none more) should be ['FFFFSSSS', 'FFFSSSSF', 'FFSFSSSS']."
    assert examples[4] == ["FFFSSSSS", "FFSSSSSF"], "The examples generated for 8 attempts and exactly 5 consecutive successes (and none more) should be ['FFFSSSSS', 'FFSSSSSF']."
    assert examples[5] == ["FFSSSSSS", "SFSSSSSS"], "The examples generated for 8 attempts and 6 consecutive successes exactly on last attempt should be ['FFSSSSSS', 'SFSSSSSS']."
    assert examples[6] == ["FSSSSSSS"], "The examples generated for 8 attempts and 7 consecutive successes should be ['FSSSSSSS']."