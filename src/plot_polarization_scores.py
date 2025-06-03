import get_polarization_scores
import get_legislation_data
import seaborn
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
from scipy.stats import t


def plot_all_congress_histogram(congress_range=(1, 118)):
    polarization_averages = get_polarization_scores.get_all_polarization_scores()
    df = pd.DataFrame(polarization_averages[congress_range[0]-1:congress_range[1]])
    seaborn.histplot(data=df, binwidth=0.05, binrange=(0, 1))
    plt.title(f"Polarization Scores for Congresses {congress_range[0]}-{congress_range[1]}")
    plt.xlabel("Polarization Score")
    plt.ylabel("Frequency")
    plt.show()


def plot_legislation_histogram():
    scores_only = get_legislation_data.get_scores_only()
    df = pd.DataFrame(scores_only)
    seaborn.histplot(data=df, binwidth=0.01, binrange=(0, 0.1))
    plt.title("Percent of Legislation Passed")
    plt.xlabel("Percent Legislation Passed")
    plt.ylabel("Frequency")
    plt.show()


def plot_legislation_over_time():
    scores_only = get_legislation_data.get_scores_only()
    idx_only = get_legislation_data.get_idx_only()
    data = {
        "Congress": idx_only,
        "Bills_Passed": scores_only
    }
    df = pd.DataFrame(data)
    seaborn.lineplot(data=df, x="Congress", y="Bills_Passed", marker="o")

    axes = plt.gca()
    axes.set_xticks(df["Congress"].unique())

    plt.xlabel("Congress")
    plt.ylabel("Proportion of Bills Passed")
    plt.title("Proportion of Bills Passed Per Congress")

    plt.show()


def plot_house_margin_over_time(congress_range=(93, 118)):
    start_congress = congress_range[0]
    end_congress = congress_range[1]
    house_margin = get_polarization_scores.get_house_party_margin()[start_congress-1: end_congress]
    congress_terms = [i for i in range(start_congress, end_congress+1)]

    data = {
        "Congress": congress_terms,
        "House_Margin": house_margin
    }

    df = pd.DataFrame(data)
    seaborn.lineplot(data=df, x="Congress", y="House_Margin", marker="o")
    axes = plt.gca()
    axes.set_xticks(df["Congress"].unique())

    plt.xlabel("Congress")
    plt.ylabel("House Margin")
    plt.title("House Margin vs Congress Number")

    plt.show()


def plot_chamber_control_over_time(congress_range=(93, 118)):
    start_congress = congress_range[0]
    end_congress = congress_range[1]

    chamber_control = get_polarization_scores.get_congress_division()[start_congress-1: end_congress]
    congress_terms = [i for i in range(start_congress, end_congress + 1)]

    data = {
        "Congress": congress_terms,
        "Chamber_Control": chamber_control
    }

    df = pd.DataFrame(data)
    seaborn.lineplot(data=df, x="Congress", y="Chamber_Control", marker="o")
    axes = plt.gca()
    axes.set_xticks(df["Congress"].unique())

    plt.xlabel("Congress")
    plt.ylabel("Chamber Control (0: Unified; 1: Divided)")
    plt.title("Chamber Control vs Congress Number")

    plt.show()


def plot_all_congress_timeseries(congress_range=(1, 118)):
    polarization_averages = get_polarization_scores.get_all_polarization_scores()
    data = {
        "Congress Number": list(range(congress_range[0], congress_range[1]+1)),
        "Polarization Score": polarization_averages[congress_range[0]-1: congress_range[1] + 1]
    }

    data = pd.DataFrame(data)
    seaborn.scatterplot(x="Congress Number", y="Polarization Score", data=data)

    seaborn.set_style("whitegrid")
    seaborn.set_context("talk")
    seaborn.regplot(
        x="Congress Number", y="Polarization Score", data=data,
        scatter_kws={'s': 60, "color": "#1f77b4", "alpha": 0.7},
        line_kws={"color": "#ff7f0e"},
        ci=0
    )
    slope, intercept = np.polyfit(data["Congress Number"], data["Polarization Score"], 1)
    best_fit_line = slope * data["Congress Number"] + intercept

    plt.title("Polarization Score vs Congress Number", fontsize=20, weight="bold")
    plt.xlabel("Congress Number", fontsize=16)
    plt.ylabel("Polarization Score", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.plot(data["Congress Number"], best_fit_line, color='#ff7f0e', linewidth=2)

    equation_text = f"y = {slope:.3f}x + {intercept:.3f}"
    plt.text(
        0.05, 0.95, equation_text,
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7)
    )

    plt.show()


def plot_one_congress(congress):
    congress_polarization_avg, congress_data = get_polarization_scores.one_congress_data(congress)
    house_scores = []
    senate_scores = []
    all_scores = []

    for chamber, name, score in congress_data:
        if chamber == "House":
            house_scores.append(score)
        elif chamber == "Senate":
            senate_scores.append(score)
        all_scores.append(score)

    df = pd.DataFrame(all_scores)
    seaborn.histplot(data=df, binwidth=0.05, binrange=(-1, 1))
    plt.title(f"Polarization Scores for the {congress}th Congress")
    plt.xlabel("Polarization Score")
    plt.ylabel("Frequency (Number of Congress Members")
    plt.show()


def make_xy_dataframe():
    polarization_scores = get_polarization_scores.get_all_polarization_scores(weighted=False)
    party_control_data = get_polarization_scores.get_congress_division()
    bills_passed = get_legislation_data.get_scores_only()
    bills_passed_idx = get_legislation_data.get_idx_only()
    house_margin_data = get_polarization_scores.get_house_party_margin()

    data = {
        "Polarization Score": [],
        "Party Control": [],
        "House Margin": [],
        "Percent of Bills Passed": bills_passed
    }

    for i in range(len(bills_passed_idx)):
        idx = bills_passed_idx[i]

        score = polarization_scores[idx - 1]
        data["Polarization Score"].append(score)

        party_control = party_control_data[idx - 1]
        data["Party Control"].append(party_control)

        house_margin = house_margin_data[idx - 1]
        data["House Margin"].append(house_margin)

    df = pd.DataFrame(data)
    return df


def plot_scatterplot():
    df = make_xy_dataframe()
    seaborn.regplot(data=df, x="Polarization Score", y="Percent of Bills Passed", ci=0)
    plt.title("Proportion of Bills Passed vs Polarization Score")
    plt.xlabel("Polarization Score")
    plt.ylabel("Proportion of Bills Passed")

    slope, intercept = np.polyfit(df["Polarization Score"], df["Percent of Bills Passed"], 1)
    best_fit_line = slope * df["Polarization Score"] + intercept

    equation_text = f"y = {slope:.3f}x + {intercept:.3f}"
    plt.text(
        0.05, 0.95, equation_text,
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7)
    )

    plt.show()


def plot_party_control():
    df = make_xy_dataframe()
    seaborn.regplot(data=df, x="Party Control", y="Percent of Bills Passed")
    plt.title("Proportion of Bills Passed vs Party Control")
    plt.xlabel("Party Control")
    plt.ylabel("Percent of Bills Passed")
    plt.show()


def plot_residual(independent="Polarization Score"):
    df = make_xy_dataframe()
    seaborn.residplot(data=df, x=independent, y="Percent of Bills Passed")
    plt.title(f"Residual Plot (Proportion of Bills Passed vs {independent})")
    plt.xlabel(independent)
    plt.ylabel("Residual")
    plt.show()


def plot_residual_histogram(independent="Polarization Score"):
    df = make_xy_dataframe()

    x = sm.add_constant(df[independent])   # adds intercept
    y = df["Percent of Bills Passed"]

    model = sm.OLS(y, x).fit()
    df["Residuals"] = model.resid

    seaborn.histplot(df["Residuals"], kde=True, binwidth=0.005, binrange=(-0.03, 0.03), color="skyblue")
    plt.axvline(0, color="red", linestyle="--")
    plt.title(f"Histogram of Residuals (Percent of Bills Passed vs {independent})")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()


def get_multiple_regression():
    df = make_xy_dataframe()
    x1 = np.array(df["Polarization Score"])
    x2 = np.array(df["Party Control"])
    x3 = np.array(df["House Margin"])
    y = np.array(df["Percent of Bills Passed"])

    # combine x1 and x2 into one array
    X = np.column_stack((x1, x2, x3))

    # add intercept term
    X = sm.add_constant(X)

    # fit least squares model
    model = sm.OLS(y, X).fit()

    return model.summary()


def get_correlation_matrix():
    df = make_xy_dataframe()
    correlation_matrix = df.corr()
    print(correlation_matrix)


def get_vif():
    df = make_xy_dataframe()
    X = df[["Polarization Score", "Party Control", "House Margin"]]
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data


def plot_t_curve(df, independent, mean, se, x_val):
    seaborn.set(style="whitegrid")
    start_x = -5
    end_x = 5
    x = np.linspace(start_x, end_x, 400)
    y = t.pdf(x, df)

    plt.figure(figsize=(8, 5))
    seaborn.lineplot(x=x, y=y, label=f"t-distribution (df={df})")

    x_fill_right = np.linspace(x_val, end_x, 100)
    plt.fill_between(x_fill_right, 0, t.pdf(x_fill_right, df), color="orange", alpha=0.5)

    x_fill_left = np.linspace(start_x, -x_val, 100)
    plt.fill_between(x_fill_left, 0, t.pdf(x_fill_left, df), color="orange", alpha=0.5)

    plt.title(f"t-distribution for {independent}")
    plt.xlabel("t")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_scatterplot()
