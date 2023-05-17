import matplotlib.pyplot as plt

# Load your images (replace these with your own image paths)
image_paths = ['img_005.jpg', 'img_018.jpg', 'img_025.jpg',
               'img_035.jpg', 'img_049.jpg', 'img_058.jpg',]

# Create a figure and axes with a single row and six subplots
fig, axes = plt.subplots(1, 6, figsize=(15, 3))

# Set the title for the figure
# fig.suptitle("6 Sample Frames for 'Drink'", fontsize=14, fontweight='bold')

# Loop through the axes and images, and display each image with labels
for i, (ax, image_path) in enumerate(zip(axes, image_paths), start=1):
    image = plt.imread(image_path)
    ax.imshow(image)
    ax.axis('off')
    # ax.set_title(f"Frame{i}", fontsize=10)
    ax.text(0.5, -0.2, f"Frame {i}", fontsize=10,
            transform=ax.transAxes, ha='center')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.2)

# Show the plot
plt.show()
