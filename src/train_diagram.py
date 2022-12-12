import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rc("text", usetex=True)
mpl.rc("font", family="serif")
mpl.rc("font", size=10)


def plot_train_diagram(times, paths, x):
    plt.figure(figsize=(4, 3))

    colors = {0: "black", 1: "red", 2: "green", 3: "blue", 4: "orange", 5: "brown", 6: "cyan"}

    station_marks = ["s0", "", "s1", "", "s2", "", "", "s3", "s4", "", "s5", "", "s6", ""]

    for k in times.keys():
        plt.plot(times[k], paths[k], "o-", label=f"AGV {k} ", color=colors[k], linewidth=0.85, markersize=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol = 3)

    for el in x:
        plt.axhline(y = el, color="gray", linewidth=0.5, linestyle=":")

    plt.yticks(x, station_marks)

    plt.xlabel("time")
    plt.ylabel("stations")
    plt.subplots_adjust(bottom=0.19, top = 0.75)

    #plt.ylim(bottom=0, top = 42)
    #plt.title("train diagram")
    
    plt.savefig("train_diagram.pdf")