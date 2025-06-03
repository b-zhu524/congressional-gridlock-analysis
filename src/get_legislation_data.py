import csv


def get_bill_frequency():
    max_congress = 118
    bill_frequency = []

    with open("../dataset-input/00-Legislation.csv", "r") as legislation:
        csv_reader = csv.reader(legislation)
        next(csv_reader)

        for row in csv_reader:
            congress_number = int(row[0][:-2])
            total_passed = int(row[4])
            total_proposed = int(row[16])
            percent_passed = total_passed / total_proposed
            bill_frequency.append((congress_number, percent_passed))

    sorted_bill_frequency = sorted(bill_frequency, key=lambda pair: pair[0])
    return sorted_bill_frequency


def get_scores_only():
    legislation_data = get_bill_frequency()
    scores_only = list(map(lambda pair: pair[1], legislation_data))
    return scores_only


def get_idx_only():
    legislation_data = get_bill_frequency()
    idx_only = list(map(lambda pair: pair[0], legislation_data))
    return idx_only


if __name__ == "__main__":
    print(get_bill_frequency())

