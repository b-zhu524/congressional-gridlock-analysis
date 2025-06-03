import csv

MAX_CONGRESS = 118


def get_all_polarization_scores(weighted=False):
    polarization_averages = [-1] * MAX_CONGRESS
    with open("../dataset-input/00-Polarization.csv", "r") as scores:
        csv_reader = csv.reader(scores)
        next(csv_reader)

        gop_score = 0
        gop_num = 0
        dem_score = 0
        dem_num = 0
        current_congress = 1
        for row in csv_reader:
            congress_number = int(row[0])

            if congress_number != current_congress or congress_number == MAX_CONGRESS + 1:
                if weighted:
                    total = gop_num + dem_num
                    polarization_averages[current_congress - 1] = abs((gop_num / total) * (gop_score / gop_num) -
                                                                      (dem_num / total) * (dem_score / dem_num))
                else:
                    polarization_averages[current_congress - 1] = abs((gop_score / gop_num) - (dem_score / dem_num))
                current_congress = congress_number
                gop_score = 0
                gop_num = 0
                dem_score = 0
                dem_num = 0
                if congress_number == MAX_CONGRESS + 1:
                    break

            dim_1 = row[13]  # Polarization Score

            if not dim_1:
                continue

            member_score = float(dim_1)
            if member_score > 0:  # conservative
                gop_score += member_score
                gop_num += 1
            else:  # liberal
                dem_score += member_score
                dem_num += 1

    return polarization_averages


def one_congress_data(congress):
    data = []
    gop_data = []
    dem_data = []
    with open("../dataset-input/00-Polarization.csv", "r") as scores:
        csv_reader = csv.reader(scores)

        gop_score = 0
        gop_num = 0
        dem_score = 0
        dem_num = 0

        next(csv_reader)

        for row in csv_reader:
            congress_number = row[0]
            if int(congress_number) > congress:
                break

            chamber = row[1]
            name = row[9]
            dim_1 = row[13]

            if int(congress_number) != congress:
                continue

            if not dim_1:
                continue

            data.append((chamber, name, float(dim_1)))

            member_score = float(dim_1)
            if member_score > 0:  # conservative
                gop_score += member_score
                gop_num += 1
            else:  # liberal
                dem_score += member_score
                dem_num += 1

        gop_avg = gop_score / gop_num
        dem_avg = dem_score / dem_num
        gop_data.append(gop_avg)
        dem_data.append(dem_avg)
        polarization_avg = abs(gop_avg - dem_avg)

    return polarization_avg, data


def get_congress_division_data():
    party_control = [-1] * MAX_CONGRESS
    party_margin = [-1] * MAX_CONGRESS
    with open("../dataset-input/00-Polarization.csv", "r") as scores:
        csv_reader = csv.reader(scores)
        next(csv_reader)

        house_dem = 0
        house_gop = 0
        senate_dem = 0
        senate_gop = 0
        current_congress = 1
        for row in csv_reader:
            congress_number = int(row[0])

            if congress_number != current_congress:
                party_margin[current_congress - 1] = abs(house_dem - house_gop)
                # add to party control
                if (house_dem >= house_gop and senate_dem >= senate_gop) or (house_gop >= house_dem and senate_gop >= senate_dem):
                    party_control[current_congress-1] = 0   # unified
                else:
                    party_control[current_congress-1] = 1   # divided
                # change values
                house_dem = 0
                house_gop = 0
                senate_dem = 0
                senate_gop = 0
                # change current congress
                current_congress = congress_number

                if current_congress == MAX_CONGRESS + 1:
                    break

            chamber = row[1]
            dim_1 = row[13]
            if not dim_1:
                continue
            dim_1 = float(dim_1)

            if chamber == "Senate":
                if dim_1 > 0:  # conservative
                    senate_gop += 1
                else:
                    senate_dem += 1
            elif chamber == "House":
                if dim_1 > 0:
                    house_gop += 1
                else:
                    house_dem += 1
    return party_control, party_margin


def get_congress_division():
    party_control, _ = get_congress_division_data()
    return party_control


def get_house_party_margin():
    _, house_party_margin = get_congress_division_data()
    return house_party_margin


if __name__ == "__main__":
    print(get_house_party_margin())
