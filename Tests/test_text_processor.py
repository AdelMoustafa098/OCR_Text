import pytest
from text_processor import TextProcessor

@pytest.fixture
def setup_text_processor():
    """
    Fixture to initialize the TextProcessor instance with a mock character set.
    """
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"
    return TextProcessor(characters)


def test_text_to_sequence(setup_text_processor):
    """
    Test the text_to_sequence function.
    """
    text_processor = setup_text_processor
    input_text = "hello world"
    expected_sequence = [7, 4, 11, 11, 14, 52, 22, 14, 17, 11, 3]  # Based on the character set index
    result = text_processor.text_to_sequence(input_text)

    assert result == expected_sequence, f"Expected {expected_sequence}, but got {result}"


def test_text_to_sequence_with_unknown_chars(setup_text_processor):
    """
    Test text_to_sequence with unknown characters.
    """
    text_processor = setup_text_processor
    input_text = "hello@world"  # '@' is not in the character set
    result = text_processor.text_to_sequence(input_text)

    # Expect '@' to be excluded or ignored
    expected_sequence = [7, 4, 11, 11, 14, 22, 14, 17, 11, 3]
    assert result == expected_sequence, f"Expected {expected_sequence}, but got {result}"


def test_sequence_to_text(setup_text_processor):
    """
    Test the sequence_to_text function.
    """
    text_processor = setup_text_processor
    input_sequence = [7, 4, 11, 11, 14, 52, 22, 14, 17, 11, 3]
    expected_text = "hello world"
    result = text_processor.sequence_to_text(input_sequence)

    assert result == expected_text, f"Expected {expected_text}, but got {result}"


def test_sequence_to_text_with_invalid_indices(setup_text_processor):
    """
    Test sequence_to_text with invalid indices.
    """
    text_processor = setup_text_processor
    input_sequence = [7, 4, 11, 200, 14, 52]  # 200 is not a valid index
    result = text_processor.sequence_to_text(input_sequence)

    # Expect invalid indices to be skipped
    expected_text = "helo "
    assert result == expected_text, f"Expected {expected_text}, but got {result}"
