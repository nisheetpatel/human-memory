import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_summary_stats(
    df, df_pmt, directory="./figures/pilot_data_figures/summary_stats/"
):
    ## Plotting
    palette = sns.color_palette("colorblind", 5)

    # plotting each bonus option
    df_pmt["Bonus option id"] = (df_pmt["Condition"] % 12) + 1
    df_pmt["Bonus option type"] = df_pmt["Condition"] // 12
    sns.lineplot(
        x="Bonus option id",
        y="Choice accuracy",
        hue="Subject ID",
        style="Bonus option type",
        data=df_pmt,
        palette=palette,
    )
    plt.savefig(f"{directory}choice_accuracy_split-by-option-good-bad.svg")

    # session-wise plot
    df0 = df[(df["Session type"] == "training")]
    sns.lineplot(
        x="State",
        y="Choice accuracy",
        hue="Subject ID",
        style="Session ID",
        data=df0,
        palette=palette,
    )
    plt.savefig(f"{directory}choice_accuracy_training_split-sessions.svg")
    plt.close()

    # session-wise plot
    df0 = df[(df["Session type"] == "testing")]
    sns.lineplot(
        x="State",
        y="Choice accuracy",
        hue="Subject ID",
        style="Session ID",
        data=df0,
        palette=palette,
    )
    plt.savefig(f"{directory}choice_accuracy_testing_split-sessions.svg")
    plt.close()

    # day-wise plot
    df0 = df[(df["Session type"] == "testing")]
    sns.lineplot(
        x="State",
        y="Choice accuracy",
        hue="Subject ID",
        style="Day",
        data=df0,
        palette=palette,
    )
    plt.savefig(f"{directory}choice_accuracy_testing_split-days.svg")
    plt.close()

    # subjects delta_pmt values
    fig = sns.violinplot(
        x="Subject ID",
        y="delta",
        data=df,
    )
    fig.set_xlabel("")
    fig.set_ylabel("$\Delta_{PMT}$")
    fig.set_title("Final $\Delta_{PMT}$ values for all subjects")
    table = plt.table(
        cellText=[list(df["delta"].unique().astype(str))],
        colLabels=[""] * 5,
        rowLabels=["$\Delta_{PMT}$"],
        colColours=palette,
    )
    table.scale(1, 1.6)
    plt.savefig(f"{directory}delta_pmt_all_subjects.svg")
    plt.close()

    # PMT trials only
    sns.lineplot(
        x="State", y="Choice accuracy", hue="Subject ID", data=df_pmt, palette=palette
    )
    plt.savefig(f"{directory}choice_accuracy_PMT.svg")
    plt.close()

    # PMT good subjects only
    df0 = df_pmt[(df_pmt["Subject ID"] != 1) & (df_pmt["Subject ID"] != 5)]
    sns.lineplot(
        x="State",
        y="Choice accuracy",
        hue="Subject ID",
        data=df0,
        palette=palette[1:-1],
    )
    plt.savefig(f"{directory}choice_accuracy_PMT_good-subjects.svg")
    plt.close()

    # all sessions all subjects
    sns.lineplot(
        x="State", y="Choice accuracy", hue="Subject ID", data=df, palette=palette
    )
    plt.savefig(f"{directory}choice_accuracy_all_sessions.svg")
    plt.close()

    # test sessions all subjects
    sns.lineplot(
        x="State", y="Choice accuracy", hue="Subject ID", data=df, palette=palette
    )
    plt.savefig(f"{directory}choice_accuracy_test_sessions.svg")
    plt.close()


if __name__ == "__main__":
    print("\nThis file is not supposed to be run as a script.")
    print(
        "It serves as as helper plotting function for extract_data.py, which you should run instead.\n"
    )
