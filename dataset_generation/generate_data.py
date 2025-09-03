from graphing_functions import plot_line,plot_bar,plot_pie,plot_enhanced_line,plot_histogram,plot_log
from graphing_functions_advanced import plot_scatter, plot_boxplot,plot_multi_line, plot_heatmap,plot_radar,plot_pairwise
    
def main():
    for i in range(500): #500
        plot_line(dataset_folder='dataset_generation/generated_plots')
        plot_bar(dataset_folder='dataset_generation/generated_plots')
        plot_pie(dataset_folder='dataset_generation/generated_plots')
        plot_histogram(dataset_folder='dataset_generation/generated_plots')
        plot_log(dataset_folder='dataset_generation/generated_plots')
        plot_enhanced_line(dataset_folder='dataset_generation/generated_plots')
        plot_scatter(dataset_folder='dataset_generation/generated_plots')
        plot_boxplot(dataset_folder='dataset_generation/generated_plots')
        plot_multi_line(dataset_folder='dataset_generation/generated_plots')
        plot_heatmap(dataset_folder='dataset_generation/generated_plots') 
        plot_radar(dataset_folder='dataset_generation/generated_plots')
        plot_pairwise(dataset_folder='dataset_generation/generated_plots')

if __name__ == "__main__":
    main()
        