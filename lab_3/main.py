import csv
import re

from configurations import REGEX_PATTERNS, VARIANT
from checksum import calculate_checksum, serialize_result


def check_invalid_row(row: list) -> bool:
    """
        function of checking each row for validity
    """
    for patterns, item in zip(REGEX_PATTERNS.keys(), row):
        if not re.search(REGEX_PATTERNS[patterns], item):
            return False
    return True


def get_no_invalid_data_index(data: list) -> list:
    """
        function finds invalid strings and writes their indexes
    """
    data_index = []
    index = 0
    for row in data:
        if not check_invalid_row(row):
            data_index.append(index)
        index += 1
    return data_index


if __name__ == "__main__":
    data = []
    with open("86.csv", "r", newline="", encoding="utf-16") as file:
        read_data = csv.reader(file, delimiter=";")
        for row in read_data:
            data.append(row)
    data.pop(0)
    serialize_result(VARIANT, calculate_checksum(
        get_no_invalid_data_index(data)))
