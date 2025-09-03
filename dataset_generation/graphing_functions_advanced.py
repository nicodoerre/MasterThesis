import matplotlib.pyplot as plt
import numpy as np
import random
import os
from datetime import datetime
import uuid
import seaborn as sns
from constants import axis_labels, titles, single_colors, line_styles, background_colors, grid_options,legend_labels,text_size_ranges,markers,dataframes,font_families,font_weights,font_styles

def plot_scatter(dataset_folder="dataset", dpi=160):
    title_size = np.random.uniform(*text_size_ranges["large"])
    label_size = np.random.uniform(*text_size_ranges["medium"])
    legend_size = np.random.uniform(*text_size_ranges["small"])
    
    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    
    x_label = random.choice(axis_labels["x_labels"])
    y_label = random.choice(axis_labels["y_labels"])
    title = random.choice(titles)
    grid = random.choice(grid_options)
    fig, ax = plt.subplots()
    background_color = random.choice(background_colors)
    plot_area_color = random.choice(background_colors)
    marker_style = random.choice(markers)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(plot_area_color)
    x_data = []
    y_data = []
    point_colors = []
    
    num_clusters = np.random.randint(1, 5)
    points_per_cluster = np.random.randint(20, 100)
    include_legend = np.random.choice([True,False])
    colors = random.sample(single_colors, num_clusters)
    include_outliers = np.random.choice([True, False])
    
    for i in range(num_clusters):
        center_x, center_y = np.random.uniform(-10, 10), np.random.uniform(-10, 10)
        x_cluster = center_x + np.random.normal(0, 2, points_per_cluster)
        y_cluster = center_y + np.random.normal(0, 2, points_per_cluster)
        x_data.extend(x_cluster)
        y_data.extend(y_cluster)
        point_colors.extend([colors[i]] * points_per_cluster)
        if include_legend:
            ax.scatter(x_cluster, y_cluster, color=colors[i], label=np.random.choice(legend_labels),marker=marker_style)

    if include_outliers:
        num_outliers = np.random.randint(5, 15)
        x_outliers = np.random.uniform(-15, 15, num_outliers)
        y_outliers = np.random.uniform(-15, 15, num_outliers)
        x_data.extend(x_outliers)
        y_data.extend(y_outliers)
        point_colors.extend(["#000000"] * num_outliers)  
    
    point_sizes = np.random.randint(1, 200, len(x_data))

    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"scatter_plot_{unique_id}.png"

    ax.scatter(x_data, y_data, c=point_colors, s=point_sizes, alpha=0.7, edgecolor="k",marker=marker_style)
    if include_legend:
        ax.legend(prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
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


def plot_boxplot(dataset_folder="dataset", dpi=160):
    title_size = np.random.uniform(*text_size_ranges["large"])
    label_size = np.random.uniform(*text_size_ranges["medium"])
    legend_size = np.random.uniform(*text_size_ranges["small"])
    
    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    
    x_label = random.choice(axis_labels["x_labels"])
    y_label = random.choice(axis_labels["y_labels"])
    all_labels = axis_labels["x_labels"] + axis_labels["y_labels"]
    title = random.choice(titles)
    grid = random.choice(grid_options)
    fig, ax = plt.subplots()
    background_color = random.choice(background_colors)
    plot_area_color = random.choice(background_colors)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(plot_area_color)
    
    num_categories = np.random.randint(3, 8)
    labels = random.sample(all_labels, num_categories)
    data = [np.random.normal(loc=np.random.uniform(-5, 5), scale=np.random.uniform(1, 3), size=100) 
            for _ in range(num_categories)]
    
    box_colors = random.sample(single_colors, num_categories)
    #category_labels = [f"Category {i+1}" for i in range(num_categories)]
    include_legend = random.choice([True, False])

    box = ax.boxplot(data, patch_artist=True)
    for patch, color, label in zip(box['boxes'], box_colors, labels):
        patch.set_facecolor(color)
        if include_legend:
            patch.set_label(label)

    if include_legend:
        ax.legend(loc="upper right", title="Categories",prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
    ax.set_xlabel(x_label, fontsize=label_size, **label_font)
    ax.set_ylabel(y_label, fontsize=label_size, **label_font)
    ax.set_title(title, fontsize=title_size, **title_font)
    ax.grid(grid)

    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"box_plot_{unique_id}.png"

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")

def plot_multi_line(dataset_folder="dataset", dpi=160):
    title_size = np.random.uniform(*text_size_ranges["large"])
    label_size = np.random.uniform(*text_size_ranges["medium"])
    legend_size = np.random.uniform(*text_size_ranges["small"])
    
    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    
    x_label = random.choice(axis_labels["x_labels"])
    y_label = random.choice(axis_labels["y_labels"])
    all_labels = axis_labels["x_labels"] + axis_labels["y_labels"]
    title = random.choice(titles)
    grid = random.choice(grid_options)
    fig, ax = plt.subplots()
    background_color = random.choice(background_colors)
    plot_area_color = random.choice(background_colors)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(plot_area_color)
    
    x = np.linspace(0, 10, 100)
    num_lines = np.random.randint(2, 6)
    include_legend = random.choice([True, False])
    labels = random.sample(all_labels, num_lines)
    
    for i in range(num_lines):
        a = np.random.uniform(0.5, 5)
        b = np.random.uniform(0.1, 1)
        noise_level = np.random.uniform(0.1, 0.5)
        choice_pattern = np.random.choice(['linear', 'sinusoidal', 'exponential', 'negative_exponential'])
        if choice_pattern == "linear":
            y = a * x + b + np.random.normal(0, noise_level, len(x))
        elif choice_pattern == "sinusoidal":
            y = a * np.sin(b * x) + np.random.normal(0, noise_level, len(x))
        elif choice_pattern == "exponential":
            y = a * np.exp(b * x) + np.random.normal(0, noise_level, len(x))
        elif choice_pattern == "negative_exponential":
            y = a * np.exp(-b * x) + np.random.normal(0, noise_level, len(x))
        
        line_color = random.choice(single_colors)
        line_style = random.choice(line_styles)
        line_width = np.random.uniform(1, 3)
        #line_label = f"Series {i+1}" if legend_choice else None
        line_label = labels[i] if include_legend else None
        
        ax.plot(x, y, color=line_color, linestyle=line_style, linewidth=line_width, label=line_label)
    
    if include_legend:
        legend_positions = [
            ("best", None),
            ("upper right", None),
            ("lower left", None),
            ("center left", (1.2, 0.5)),
            ("center right", (-0.4, 0.5)),
            ("upper left",None),
            ("lower right", None),
        ]  
        legend_loc, bbox = random.choice(legend_positions)

        if bbox:
            ax.legend(loc=legend_loc, bbox_to_anchor=bbox, prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
        else:
            ax.legend(loc=legend_loc, prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
    
    ax.set_xlabel(x_label, fontsize=label_size, **label_font)
    ax.set_ylabel(y_label, fontsize=label_size, **label_font)
    ax.set_title(title, fontsize=title_size, **title_font)
    ax.grid(grid)
    
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"multi_line_plot_{unique_id}.png"
    
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")
    
def plot_heatmap(dataset_folder="dataset", dpi=160):
    title_size = np.random.uniform(*text_size_ranges["large"])
    label_size = np.random.uniform(*text_size_ranges["medium"])
    
    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}

    
    x_label = random.choice(axis_labels["x_labels"])
    y_label = random.choice(axis_labels["y_labels"])
    title = random.choice(titles)
    grid = random.choice(grid_options)
    fig, ax = plt.subplots()
    background_color = random.choice(background_colors)
    fig.patch.set_facecolor(background_color)
    
    rows, cols = np.random.randint(10, 30, size=2)
    pattern_choice = random.choice(['gradient', 'radial', 'clusters'])

    if pattern_choice == 'gradient':
        data = np.linspace(0, 1, rows).reshape(rows, 1) * np.linspace(0, 1, cols).reshape(1, cols)
    elif pattern_choice == 'radial':
        x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
        data = np.sqrt(x**2 + y**2)  
        data = 1 - data  
    elif pattern_choice == 'clusters':
        data = np.random.uniform(0, 1, size=(rows, cols))
        num_clusters = np.random.randint(2, 5)
        for _ in range(num_clusters):
            cluster_x = np.random.randint(0, cols)
            cluster_y = np.random.randint(0, rows)
            data[cluster_y:cluster_y+3, cluster_x:cluster_x+3] += np.random.uniform(0.5, 1.0)
        data = np.clip(data, 0, 1)
    
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'cividis']
    cmap = random.choice(colormaps)

    heatmap = ax.imshow(data, cmap=cmap, aspect='auto')

    include_colorbar = random.choice([True, False])
    if include_colorbar:
        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.set_label(np.random.choice(legend_labels))
    
    ax.set_xlabel(x_label, fontsize=label_size, **label_font)
    ax.set_ylabel(y_label, fontsize=label_size, **label_font)
    ax.set_title(title, fontsize=title_size, **title_font)
    ax.grid(grid)
    
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"heatmap_{unique_id}.png"
    
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")
    
def plot_radar(dataset_folder="dataset", dpi=160):
    title_size = np.random.uniform(*text_size_ranges["large"])
    label_size = np.random.uniform(*text_size_ranges["medium"])
    legend_size = np.random.uniform(*text_size_ranges["small"])
    
    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    
    title = random.choice(titles)
    all_labels = axis_labels["x_labels"] + axis_labels["y_labels"]
    background_color = random.choice(background_colors)
    num_axes = min(np.random.randint(3, 8), len(all_labels))
    num_series = min(np.random.randint(2, 5), num_axes)  # Prevent IndexError
    labels = random.sample(all_labels, num_axes)
    data = np.random.uniform(0.1, 1, size=(num_series, num_axes))
    data = np.hstack([data, data[:, [0]]]) 

    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots( subplot_kw={'polar': True}) #figsize=(6, 6),
    fig.patch.set_facecolor(background_color)

    legend_labels = []
    for i in range(num_series):
        line_color = random.choice(single_colors)
        line_style = random.choice(line_styles)
        fill_color = random.choice(single_colors)
        fill_alpha = np.random.uniform(0.2, 0.6)
        #legend_label = f"Series {i+1}"
        legend_label = labels[i]
        legend_labels.append(legend_label)

        ax.plot(angles, data[i], linestyle=line_style, linewidth=2, color=line_color, label=legend_label)
        ax.fill(angles, data[i], color=fill_color, alpha=fill_alpha)

    ax.set_yticks([])  
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=label_size, **label_font)
    ax.set_title(title, va='bottom', fontsize=title_size, **title_font)

    include_legend = random.choice([True, False])
    if include_legend:
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
            ax.legend(loc=legend_loc, bbox_to_anchor=bbox, title="Series", prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
        else:
            ax.legend(loc=legend_loc, title="Series", prop={"size": legend_size, "weight": random.choice(font_weights), "style": random.choice(font_styles)})
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"radar_plot_{unique_id}.png"

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")
    
def plot_pairwise(dataset_folder="dataset", dpi=160):
    _,df = random.choice(list(dataframes.items()))
    num_features = df.shape[1]
    feature_names = df.columns
    fig, axes = plt.subplots(num_features, num_features) #, figsize=(15, 15)
    
    label_size = np.random.uniform(*text_size_ranges["medium"])
    title_size = np.random.uniform(*text_size_ranges["large"])
    tick_size = np.random.uniform(*text_size_ranges["small"])
    corr_text_size = np.random.uniform(*text_size_ranges["small"])
    
    title_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}
    label_font = {"fontname": random.choice(font_families), "weight": random.choice(font_weights), "style": random.choice(font_styles)}

    grid = random.choice(grid_options)

    include_legend = random.choice([True, False]) and 'category' in df.columns
    fig.patch.set_facecolor(random.choice(background_colors))

    title = random.choice(titles)
    fig.suptitle(title, fontsize=title_size, **title_font)

    for i in range(num_features):
        for j in range(num_features):
            ax = axes[i, j]
            ax.set_facecolor(random.choice(background_colors))
            if i == j:
                scatter_color = random.choice(single_colors)
                diagonal_choice = random.choice(['hist', 'kde', 'box'])
                if diagonal_choice == 'kde':
                    sns.kdeplot(df.iloc[:, i], ax=ax, fill=True, color=scatter_color, alpha=0.6)
                elif diagonal_choice == 'box':
                    ax.boxplot(df.iloc[:, i], patch_artist=True, boxprops=dict(facecolor=scatter_color, alpha=0.7))
                else:
                    ax.hist(df.iloc[:, i], bins=15, color=scatter_color, alpha=0.7)
                ax.set_title(feature_names[i], fontsize=label_size, **label_font)
            else:
                scatter_color = random.choice(single_colors)
                ax.scatter(df.iloc[:, j], df.iloc[:, i], color=scatter_color, alpha=0.7, marker=random.choice(markers))
                
                corr = np.corrcoef(df.iloc[:, j], df.iloc[:, i])[0, 1]
                ax.text(0.5, 0.9, f"r = {corr:.2f}", ha='center', va='center', transform=ax.transAxes, fontsize=corr_text_size)
                if include_legend:
                    categories = df['category'].unique()
                    for category in categories:
                        subset = df[df['category'] == category]
                        ax.scatter(subset.iloc[:, j], subset.iloc[:, i], label=category, alpha=0.7)
                    ax.legend(fontsize=np.random.uniform(*text_size_ranges["small"]))
                    
            ax.grid(grid)

            ax.tick_params(axis='both', labelsize=tick_size)
            
            if i < num_features - 1:
                ax.set_xticklabels([],**label_font)
            if j > 0:
                ax.set_yticklabels([],**label_font)

    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    file_name = f"pairwise_plot{unique_id}.png"
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    file_path = os.path.join(dataset_folder, file_name)
    plt.savefig(file_path, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved to: {file_path}")
    
