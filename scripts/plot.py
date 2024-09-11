import json
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee', 'bright']) #global style set for matplotlib and pandas plot

# Opening JSON file
f = open('/scratch/hpc-prf-agraph/bfs-gpu/data.json')

# returns JSON object as a dictionary
data = json.load(f)

# print(data)
graph_name = "rmat-21-64"
# Iterating through the json list
for name in data:
    print(name)
    for i in data[name]:
        print(i,data[name][i]['avgMTEPSFilter'])

# Closing file
f.close()

# # Number of datasets
# num_datasets = len(data)

# # Create a figure with subplots
# fig, axes = plt.subplots(nrows=num_datasets, ncols=1, figsize=(10, 6 * num_datasets))

# # If there's only one subplot, axes is not an array, so we need to handle that case
# if num_datasets == 1:
#     axes = [axes]

# # Iterate through datasets and create subplots
# for ax, (dataset_name, values) in zip(axes, data.items()):
#     x = list(values.keys())
#     y = [values[i]['avgMTEPSFilter'] for i in x]
    
#     ax.plot(x, y, marker='o', linestyle='-', color='b')
#     ax.set_title(f'Average MTEPS Filter for {dataset_name}')
#     ax.set_xlabel('Index')
#     ax.set_ylabel('Average MTEPS Filter')
#     ax.grid(True)
#     ax.set_xticks(x)  # Ensure all x values are shown

# # Adjust layout
# plt.tight_layout()

# # Save the combined plot as an image file
# plt.savefig('combined_plot.png')
# plt.show()



# # Create a single plot
# plt.figure(figsize=(12, 8))

# # Plot each dataset
# for dataset_name, values in data.items():
#     x = list(values.keys())
#     y = [values[i]['avgMTEPSFilter'] for i in x]
    
#     plt.plot(x, y, marker='o', linestyle='-', label=dataset_name)

# # Add labels and title
# plt.title('Average MTEPS Filter for All Datasets')
# plt.xlabel('Index')
# plt.ylabel('Average MTEPS Filter')
# plt.legend()  # Show a legend to differentiate datasets
# plt.grid(True)
# plt.xticks(sorted(set(idx for vals in data.values() for idx in vals.keys())))  # Ensure all x values are shown

# # Save the combined plot as an image file
# plt.savefig('combined_plot_all_datasets_avgMTEPSFilter.png')
# plt.show()


# # Categorize datasets
# categories = {}
# for dataset_name in data.keys():
#     category = dataset_name.split('_')[0]  # Extract category from dataset name
#     if category not in categories:
#         categories[category] = []
#     categories[category].append(dataset_name)

# # Create subplots for each category
# num_categories = len(categories)
# fig, axs = plt.subplots(  1,num_categories, figsize=(20, 4))  # Create subplots

# for ax, (category, datasets) in zip(axs, categories.items()):
#     for dataset_name in datasets:
#         x = list(data[dataset_name].keys())
#         y = [data[dataset_name][i]['avgMTEPSFilter'] for i in x]
#         ax.plot(x, y, marker='o', linestyle='-', label=dataset_name)
    
#     ax.set_title(f'{category}')
#     ax.set_ylabel('Average MTEPS Filter')
#     ax.set_xlabel('Number of GPU(s)')
#     ax.legend()
#     ax.grid(True)

# # Configure common x-axis

# # plt.xticks(sorted(set(idx for vals in data.values() for idx in vals.keys())))  # Ensure all x values are shown

# # Adjust layout and save the plot
# plt.tight_layout()
# plt.savefig('subplots_by_category_avgMTEPSFilter.eps', format='eps')
# # plt.show()


# Categorize datasets
categories = {}
for dataset_name in data.keys():
    category = dataset_name.split('_')[0]  # Extract category from dataset name
    if category not in categories:
        categories[category] = []
    categories[category].append(dataset_name)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 8))  # Create a 2x2 grid of subplots

# Flatten the axs array for easier iteration
axs = axs.flatten()

# Plot each category in its respective subplot
for ax, (category, datasets) in zip(axs, categories.items()):
    for dataset_name in datasets:
        x = list(data[dataset_name].keys())
        y = [data[dataset_name][i]['avgMTEPSFilter'] for i in x]
        ax.plot(x, y, marker='o', linestyle='-', label=dataset_name)
    
    ax.set_title(f'{category}')
    ax.set_xlabel('Number of GPU(s)')  # Set x-label for each subplot
    ax.set_ylabel('Average MTEPS Filter')
    ax.legend()



# Hide any unused subplots
for i in range(len(categories), len(axs)):
    fig.delaxes(axs[i])

# Adjust layout
plt.tight_layout()  # Adjust layout to prevent overlap

# Save the plot as an EPS file
plt.savefig('subplots_by_category_avgMTEPSFilter.eps', format='eps')
plt.savefig('subplots_by_category_avgMTEPSFilter.png')