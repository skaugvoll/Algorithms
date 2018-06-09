def pretty_print_matrix(m):
    for day, value in enumerate(m):
        print("Day {}, value {:<30}".format(day, str(value.A1)))


def enumerate_backwards(sequence):
    n = len(sequence) - 1 
    for i in range(0, len(sequence)):
        yield n, sequence[n]
        n -= 1