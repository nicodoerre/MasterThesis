import matplotlib.pyplot as plt
import numpy as np
import random
import os
from datetime import datetime
import uuid
from constants import axis_labels, titles, single_colors, line_styles, background_colors, grid_options,legend_labels,text_size_ranges,font_families,font_weights,font_styles

def plot_line( dataset_folder="dataset", dpi=160):
    '''
    Generates and saves a line plot with random parameters.
    Parameters:
    - dataset_folder: Folder to save the generated plot.
    - dpi: Resolution of the saved plot.
    '''
    title_size = np.random.uniform(*text_size_ranges["large"])
    label_size = np.random.uniform(*text_size_ranges["medium"])
    legend_size = np.random.uniform(*text_size_ranges["small"])
    
    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}


    choice_x = np.random.choice([1,2,3])
    choice_pattern = np.random.choice(['linear','sinusoidal','random'])
    include_legend = np.random.choice([True,False])
    length = np.random.randint(50, 201)
    noise_level = np.random.choice([0.05,0.1,0.15,0.2,0.25])
    
    if choice_x == 1:
        x = np.linspace(0, length, length)
    elif choice_x == 2:
        x = np.linspace(0, length, length) + np.random.normal(0, noise_level, length)
    elif choice_x == 3:
        x = np.sort(np.random.uniform(0, length, length))

    if choice_pattern == "linear":
        y = 0.5 * x + np.random.normal(0, noise_level, length)
    elif choice_pattern == "sinusoidal":
        y = 10 * np.sin(0.1 * x) + np.random.normal(0, noise_level, length)
    elif choice_pattern == "random":
        y = np.random.normal(0, 1, length)
    else:
        raise ValueError("Unsupported data_type. Choose from 'linear', 'sinusoidal', or 'random'.")

    x_label = random.choice(axis_labels["x_labels"])
    y_label = random.choice(axis_labels["y_labels"])
    title = random.choice(titles)
    line_color = random.choice(single_colors)
    line_style = random.choice(line_styles)
    background_color = random.choice(background_colors)
    plot_area_color = random.choice(background_colors)
    grid = random.choice(grid_options)
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"line_plot_{unique_id}.png"
    #line_label = random.choice(legend_labels)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(plot_area_color)
    if include_legend:
        ax.plot(x, y, color=line_color, linestyle=line_style, label=random.choice(legend_labels))
        ax.legend(prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
    else:
        ax.plot(x, y, color=line_color, linestyle=line_style)

    ax.set_xlabel(x_label, fontsize=label_size, **label_font)
    ax.set_ylabel(y_label, fontsize=label_size, **label_font)
    ax.set_title(title, fontsize=title_size, **title_font)
    ax.grid(grid)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")


def plot_bar(dataset_folder="dataset",dpi=160):
    '''
    Generates and saves a bar chart with random parameters.
    Parameters:
    - dataset_folder: Folder to save the generated plot.
    - dpi: Resolution of the saved plot.
    '''
    title_size = np.random.uniform(*text_size_ranges["large"])
    label_size = np.random.uniform(*text_size_ranges["medium"])
    legend_size = np.random.uniform(*text_size_ranges["small"])

    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    
    num_bars = np.random.randint(3,15)
    noise_level = np.random.choice([0.05,0.1,0.15,0.2,0.25])
    x = np.arange(num_bars)
    y = np.random.randint(5, 20, num_bars) + np.random.normal(0, noise_level, num_bars)

    x_label = random.choice(axis_labels["x_labels"])
    y_label = random.choice(axis_labels["y_labels"])
    title = random.choice(titles)
    bar_colors = random.sample(single_colors,num_bars)
    background_color = random.choice(background_colors)
    plot_area_color = random.choice(background_colors)
    grid = random.choice(grid_options)
    include_legend = random.choice([True, False])

    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"bar_chart_{unique_id}.png"

    fig, ax = plt.subplots()
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(plot_area_color)
    
    if include_legend:
        #ax.bar(x, y, color=bar_colors, label=random.choice(legend_labels))
        #ax.legend(prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
        bar_labels = random.sample(axis_labels["x_labels"] + axis_labels["y_labels"], num_bars)
        bars = ax.bar(x, y, color=bar_colors)
        for bar, label in zip(bars, bar_labels):
            bar.set_label(label)
        legend_positions = [
            ("best", None),
            ("upper right", None),
            ("lower left", None),
            ("center left", (1.2, 0.5)),
            ("center right", (-0.4, 0.5)),
            ("upper left", (1.2, 1.0)),
            ("lower right", (1.2, -0.1)),
        ]
        legend_loc, bbox = random.choice(legend_positions)
        if bbox:
            ax.legend(loc=legend_loc, bbox_to_anchor=bbox, prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
        else:
            ax.legend(prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
    else:
        ax.bar(x, y, color=bar_colors)
    
    ax.set_xlabel(x_label, fontsize=label_size, **label_font)
    ax.set_ylabel(y_label, fontsize=label_size, **label_font)
    ax.set_title(title, fontsize=title_size, **title_font)
    ax.grid(grid)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")


def plot_pie(dataset_folder="dataset", dpi=160):
    '''
    Generates and saves a pie chart with random parameters.
    Parameters:
    - dataset_folder: Folder to save the generated plot.
    - dpi: Resolution of the saved plot.
    '''
    title_size = np.random.uniform(*text_size_ranges["large"])
    legend_size = np.random.uniform(*text_size_ranges["small"])
    
    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    legend_font = {"weight": random.choice(font_weights), "style": random.choice(font_styles)}

    
    num_slices = np.random.randint(3, 10)
    values = np.random.randint(1, 10, num_slices)
    
    all_labels = axis_labels["x_labels"] + axis_labels["y_labels"]
    labels = random.sample(all_labels, num_slices)
    
    colors = random.sample(single_colors, num_slices)
    title = random.choice(titles)
    background_color = random.choice(background_colors)
    plot_area_color = random.choice(background_colors)
    include_legend = random.choice([True, False])

    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"pie_chart_{unique_id}.png"

    fig, ax = plt.subplots()
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(plot_area_color)

    wedges, texts, autotexts = ax.pie(
        values, colors=colors, labels=labels if include_legend else None, autopct='%1.1f%%'
    )
    for autotext in autotexts:
        autotext.set_fontsize(legend_size)
        autotext.set_weight(legend_font["weight"])
        autotext.set_style(legend_font["style"])
    
    ax.set_title(title,fontsize=title_size, **title_font)
    if include_legend:
        #ax.legend(labels, prop={"size": legend_size, "weight": legend_font["weight"], "style": legend_font["style"]})    
        legend_positions = [
            ("best", None),  
            ("upper right", None),
            ("lower left", None), 
            ("center left", (1.2, 0.5)),  
            ("center right", (-0.4, 0.5)),
            ("lower right", (1.2, -0.1)), 
        ]  
        legend_loc, bbox = random.choice(legend_positions)  
        if bbox:
            ax.legend(labels, loc=legend_loc, bbox_to_anchor=bbox, prop={"size": legend_size, "weight": legend_font["weight"], "style": legend_font["style"]})
        else:
            ax.legend(labels, loc=legend_loc, prop={"size": legend_size, "weight": legend_font["weight"], "style": legend_font["style"]})
            
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")
    
def plot_enhanced_line( dataset_folder="dataset", dpi=160):
    '''
    Generates and saves an enhanced line plot with random parameters.
    Parameters:
    - dataset_folder: Folder to save the generated plot.
    - dpi: Resolution of the saved plot.
    '''
    title_size = np.random.uniform(*text_size_ranges["large"])
    label_size = np.random.uniform(*text_size_ranges["medium"])
    legend_size = np.random.uniform(*text_size_ranges["small"])
    
    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}

    choice_x = np.random.choice([1,2,3])
    choice_dt = np.random.choice(['exponential','quadratic','logarithmic'])
    include_legend = np.random.choice([True,False])
    length = np.random.randint(50, 201)
    noise_level = np.random.choice([0.05,0.1,0.15,0.2,0.25])
    a = np.random.uniform(0.5, 5)
    b = np.random.uniform(-1, 1)
    c = np.random.uniform(-10, 10)
    
    if choice_x == 1:
        x = np.linspace(0, length, length)
    elif choice_x == 2:
        x = np.linspace(0, length, length) + np.random.normal(0, noise_level, length)
    elif choice_x == 3:
        x = np.sort(np.random.uniform(0, length, length))

    if choice_dt == "exponential":
        y = a * np.exp(b * x) + np.random.normal(0, noise_level, length)
    elif choice_dt == "quadratic":
        y = a * x**2 + b * x + c + np.random.normal(0, noise_level, length)
    elif choice_dt == "logarithmic":
        y = a * np.log(x + 1) + np.random.normal(0, noise_level, length)
    else:
        raise ValueError("Unsupported data_type. Choose from 'linear', 'sinusoidal', or 'random'.")

    x_label = random.choice(axis_labels["x_labels"])
    y_label = random.choice(axis_labels["y_labels"])
    title = random.choice(titles)
    line_color = random.choice(single_colors)
    line_style = random.choice(line_styles)
    background_color = random.choice(background_colors)
    plot_area_color = random.choice(background_colors)
    grid = random.choice(grid_options)
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"line_plot_{unique_id}.png"
    #line_label = random.choice(legend_labels)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(plot_area_color)
    if include_legend:
        ax.plot(x, y, color=line_color, linestyle=line_style, label=random.choice(legend_labels))
        ax.legend(prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
    else:
        ax.plot(x, y, color=line_color, linestyle=line_style)

    ax.set_xlabel(x_label, fontsize=label_size, **label_font)
    ax.set_ylabel(y_label, fontsize=label_size, **label_font)
    ax.set_title(title, fontsize=title_size, **title_font)
    ax.grid(grid)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")

def plot_histogram(dataset_folder="dataset", dpi=160):
    '''
    Generates and saves a histogram with random parameters.
    Parameters:
    - dataset_folder: Folder to save the generated plot.
    - dpi: Resolution of the saved plot.
    '''

    title_size = np.random.uniform(*text_size_ranges["large"])
    label_size = np.random.uniform(*text_size_ranges["medium"])
    legend_size = np.random.uniform(*text_size_ranges["small"])
    
    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    
    x_label = random.choice(axis_labels["x_labels"])
    y_label = random.choice(axis_labels["y_labels"])
    title = random.choice(titles)
    background_color = random.choice(background_colors)
    plot_area_color = random.choice(background_colors)
    bar_color = random.choice(single_colors)
    edge_color = random.choice(single_colors)
    grid = random.choice(grid_options)
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(plot_area_color)

    
    num_data_points = np.random.randint(50, 500) 
    data_range = (np.random.uniform(-10, 0), np.random.uniform(0, 10))  
    data = np.random.uniform(data_range[0], data_range[1], num_data_points)
    
    num_bins = np.random.randint(5, 20)
    include_legend = random.choice([True, False])
    num_labels = np.random.randint(1, min(num_bins, 5)) 
    bar_labels = random.sample(axis_labels["x_labels"] + axis_labels["y_labels"], num_labels)
    #ax.hist(data, bins=num_bins, color=bar_color, edgecolor=edge_color, alpha=0.7)
    bars = ax.hist(data, bins=num_bins, color=bar_color, edgecolor=edge_color, alpha=0.7)
    density, bins, _ = ax.hist(data, bins=num_bins, density=True,)
    center = (bins[:-1] + bins[1:]) / 2
    scale_factor = max(bars[0]) / max(density)
       
    if include_legend:
        for i, label in enumerate(bar_labels):
            if i < len(bars[2]): 
                bars[2][i].set_label(label)
        ax.plot(center, density * scale_factor, linestyle='--', color=random.choice(single_colors), label="Density")
        ax.legend(prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
    
    ax.set_xlabel(x_label, fontsize=label_size, **label_font)
    ax.set_ylabel(y_label, fontsize=label_size, **label_font)
    ax.set_title(title, fontsize=title_size, **title_font)
    ax.grid(grid)
    
   #if include_legend:
   #    ax.legend(prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})    
   #    
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"histogram_{unique_id}.png"
    
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")
    
def plot_log(dataset_folder="dataset", dpi=160):
    '''
    Generates and saves a log plot with random parameters.
    Parameters:
    - dataset_folder: Folder to save the generated plot.
    - dpi: Resolution of the saved plot.
    '''

    title_size = np.random.uniform(*text_size_ranges["large"])
    label_size = np.random.uniform(*text_size_ranges["medium"])
    legend_size = np.random.uniform(*text_size_ranges["small"])

    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}

    log_type = random.choice(["loglog", "semilogx", "semilogy"])

    x = np.logspace(0.1, 2, 100)  

    y = np.random.uniform(0.5, 2) * np.power(x, np.random.uniform(0.5, 2)) + np.random.normal(0, 1, len(x))

    x_label = random.choice(axis_labels["x_labels"])
    y_label = random.choice(axis_labels["y_labels"])
    title = random.choice(titles)
    line_color = random.choice(single_colors)
    line_style = random.choice(line_styles)
    background_color = random.choice(background_colors)
    plot_area_color = random.choice(background_colors)
    grid = random.choice(grid_options)
    
    include_legend = random.choice([True, False])

    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"log_plot_{unique_id}.png"

    fig, ax = plt.subplots()
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(plot_area_color)

    if log_type == "loglog":
        ax.loglog(x, y, color=line_color, linestyle=line_style, label=random.choice(legend_labels) if include_legend else None)
    elif log_type == "semilogx":
        ax.semilogx(x, y, color=line_color, linestyle=line_style, label=random.choice(legend_labels) if include_legend else None)
    elif log_type == "semilogy":
        ax.semilogy(x, y, color=line_color, linestyle=line_style, label=random.choice(legend_labels) if include_legend else None)

    ax.set_xlabel(x_label, fontsize=label_size, **label_font)
    ax.set_ylabel(y_label, fontsize=label_size, **label_font)
    ax.set_title(title, fontsize=title_size, **title_font)
    ax.grid(grid)

    if include_legend:
        ax.legend(prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")
