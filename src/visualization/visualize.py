import matplotlib.pyplot as plt


def plot(plot_dict, test_data):
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, val in plot_dict.items():
        
        plt.plot(test_data.datadate, val ,label = name)
    plt.title("Comparision plot",size= 18)
    plt.legend()
    plt.rc('legend',fontsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

