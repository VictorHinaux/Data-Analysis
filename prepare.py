import pickle
import matplotlib.pyplot as plt
import csv
import pandas as pd
import collections

# Folder containing dataset
fprefix = "PPG_FieldStudy/"

# Filenames
studies = [["S1/S1_quest.csv", "S1/S1_activity.csv", "S1/S1.pkl"],
           ["S2/S2_quest.csv", "S2/S2_activity.csv", "S2/S2.pkl"],
           ["S3/S3_quest.csv", "S3/S3_activity.csv", "S3/S3.pkl"],
           ["S4/S4_quest.csv", "S4/S4_activity.csv", "S4/S4.pkl"],
           ["S5/S5_quest.csv", "S5/S5_activity.csv", "S5/S5.pkl"],
           ["S7/S7_quest.csv", "S7/S7_activity.csv", "S7/S7.pkl"],
           ["S8/S8_quest.csv", "S8/S8_activity.csv", "S8/S8.pkl"],
           ["S9/S9_quest.csv", "S9/S9_activity.csv", "S9/S9.pkl"],
           ["S10/S10_quest.csv", "S10/S10_activity.csv", "S10/S10.pkl"],
           ["S11/S11_quest.csv", "S11/S11_activity.csv", "S11/S11.pkl"],
           ["S12/S12_quest.csv", "S12/S12_activity.csv", "S12/S12.pkl"],
           ["S13/S13_quest.csv", "S13/S13_activity.csv", "S13/S13.pkl"],
           ["S14/S14_quest.csv", "S14/S14_activity.csv", "S14/S14.pkl"],
           ["S15/S15_quest.csv", "S15/S15_activity.csv", "S15/S15.pkl"],
           ]

# Pyplot colors for activity grouping
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
          "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]


def plotData(data, title, segments=None):
    plt.figure(figsize=(16, 10))
    # Adjust data scale
    plt.plot([x*2 for x in range(len(data))], data)
    # Metadata
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Heart Rate [bpm]")
    # Scaling the y axis
    axes = plt.gca()
    axes.set_ylim([0, 200])
    # Segment grouping
    if segments:
        for i, s in enumerate(segments):
            plt.axvspan(s[1], s[2], alpha=0.3, label=s[0], color=colors[i])
    plt.legend()
    plt.show()
    # plt.savefig("graphs/" + title)
    plt.close()


def computeMeanBpm(data, start, end):
    # Adjust data scale
    segment = data[int(start/2):int(end/2)]
    # Return tensor mean
    return int(sum(segment) / len(segment))


def get_segments(activity_fname):
    segments = []

    # Load CSV
    with open(fprefix + activity_fname, 'r') as f:
        reader = csv.reader(f)

        # Skip subject ID
        next(reader)

        cur_start = 0
        cur_name = ""

        # Grouping timeframes together
        for row in reader:
            activity = row[0].split('# ')[1]
            time = row[1]
            if cur_start != 0:
                segments.append([cur_name, int(cur_start), int(time)])
                if activity == "NO_ACTIVITY":
                    cur_start = 0
                else:
                    cur_start = time
                    cur_name = activity
            else:
                if activity == 'NO_ACTIVITY':
                    continue
                cur_start = time
                cur_name = activity
    return segments


def get_pinfos(infos_fname):
    pinfos = []

    # Load CSV
    with open(fprefix + infos_fname, "r") as f:
        reader = csv.reader(f)

        # Skip subject ID
        next(reader)
        for row in reader:
            # Skip last line
            if row[0] == ' ':
                continue
            lab = row[0].split('# ')[1]

            # Value casting and sanitizing
            try:
                val = int(row[1])
            except ValueError:
                val = row[1].strip(' ')
            pinfos.append([lab, val])
    return pinfos


def prepare():
    # First analysis
    analysis = []

    # Pre-panda format
    raw_df = collections.OrderedDict()

    for s in studies:
        # Activities timeframes
        segments = get_segments(s[1])

        # Participants information
        pinfos = get_pinfos(s[0])
        # Dataset loading
        with open(fprefix + s[2], 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            signal = data[b"label"]

            # Mean BPM computation per activity
            for i, seg in enumerate(segments):
                segments[i] = [seg[0], computeMeanBpm(signal, seg[1], seg[2])]
            analysis.append(pinfos + segments)

    # Data reformat
    for a in analysis:
        for feat in a:
            raw_df.setdefault(feat[0], []).append(feat[1])

    # Pandas format and export
    df = pd.DataFrame(data=raw_df)
    df.to_csv('features_and_bpms.csv', index=False)
    return 0


def main():
    return prepare()


if __name__ == '__main__':
    exit(main())
