class TextProcessor:
    """
    Handles text-to-sequence and sequence-to-text conversions.
    """

    def __init__(self, characters):
        """
        Initialize the TextProcessor with a character set.

        Args:
            characters (str): All possible characters in the dataset.
        """
        self.characters = characters
        self.char_to_num = {char: i for i, char in enumerate(characters)}
        self.num_to_char = {i: char for i, char in enumerate(characters)}

    def text_to_sequence(self, text):
        """
        Convert a text string to a sequence of integers.

        Args:
            text (str): Input text string.

        Returns:
            list: List of integers representing the text.
        """

        sequence = []
        for char in text:
            if char in self.char_to_num:
                index = self.char_to_num[char]
                sequence.append(index)
        return sequence

    def sequence_to_text(self, sequence):
        """
        Convert a sequence of integers back to text.

        Args:
            sequence (list): Sequence of integers.

        Returns:
            str: Decoded text string.
        """
        text = []
        for num in sequence:
            if num in self.num_to_char:
                text.append(self.num_to_char[num])

        return "".join(text)
