"""plots train diagram for given solutions"""
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc("text", usetex=True)
mpl.rc("font", family="serif")
mpl.rc("font", size=10)


def zones_location(track_len, n_zones, s_ofset):
    """determines zones locations and borders in space for the plot"""
    marks = {f"s{k}":0 for k in range(n_zones)} 
    zone_borders = []
    prev_s = -100
    x = 0 
    for s in marks:
        if prev_s != -100:
            x = x + s_ofset + track_len[(f"{prev_s}", f"{s}")]
        marks[s] = x + s_ofset/2
        zone_borders.append(x)
        zone_borders.append(x+s_ofset)
        prev_s = s
    return marks, zone_borders

def AGVS_coordinates(sol, agv_routes, marks, s_ofset):
    """ determines coordinates of AGVs in space and time to get their paths on the plot """
    times = {}
    spaces = {}
    for agv in agv_routes:
        times[agv] = [sol[f"t_{e}_{agv}_{s}"]  for s in agv_routes[agv] for e in ["in", "out"]]
        route = [marks[s] for s in agv_routes[agv]]
        if sorted(route) != route:
            ofset = - s_ofset
        else:
            ofset = s_ofset
        spaces[agv] = [marks[s] + e for s in agv_routes[agv] for e in [-ofset/2, ofset/2]]
    return times, spaces


def plot_train_diagram(sol, agv_routes, track_len, n_zones):
    """plots and saves train diagram"""
    plt.figure(figsize=(4, 3))
    s_ofset = 1.75  # the size of the station
    marks, zone_borders = zones_location(track_len, n_zones, s_ofset)
    times, spaces = AGVS_coordinates(sol, agv_routes, marks, s_ofset)
    colors = {0: "black", 1: "red", 2: "green", 3: "blue", 4: "orange", 5: "brown", 6: "cyan"}
    
    for agv in agv_routes:
        plt.plot(times[agv], spaces[agv], "o-", label=f"$AGV_{agv}$ ", color=colors[agv], linewidth=0.85, markersize=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol = 3)

    for el in zone_borders:
        plt.axhline(y = el, color="gray", linewidth=0.5, linestyle=":")
        locs = [marks[k] for k in marks]
    
    our_marks = [f"$s_{i}$" for i, _ in enumerate(marks) ]

    plt.yticks(locs, our_marks)
    plt.xlabel("time")
    plt.ylabel("zones")
    plt.subplots_adjust(bottom=0.19, top = 0.75)        
    plt.savefig("train_diagram.pdf")
