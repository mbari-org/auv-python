import sys

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        error_message = f"Unrecognized python interpreter: {REQUIRED_PYTHON}"
        raise ValueError(error_message)

    if system_major != required_major:
        error_message = (
            f"This project requires Python {required_major}. Found: Python {sys.version}"
        )
        raise TypeError(error_message)
    print(">>> Development environment passes all tests!")  # noqa: T201


if __name__ == "__main__":
    main()
