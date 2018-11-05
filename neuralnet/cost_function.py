class SquaredError:

    @staticmethod
    def apply(outputs, expected_outputs):
        errors = [pow(output-expected, 2) for output, expected in zip(outputs, expected_outputs)]
        return 0.5 * sum(errors)




