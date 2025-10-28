**Fashion-MNIST Data Visualization and Training Curves Using TensorFlow**

Part of **IBM AI Engineering Professional Certificate — “Neural Networks and TensorFlow”**
This project demonstrates how to visualize Fashion-MNIST images and monitor simulated CNN training progress. It covers key machine learning concepts such as **dataset preprocessing, image visualization, and model performance tracking.**

**🚀 Project Overview:**
The project is divided into two main tasks, each highlighting a specific aspect of machine learning:

Task 1: **Fashion-MNIST Image Display**
In this task, we focused on **understanding and visualizing the Fashion-MNIST dataset**, which consists of grayscale images of clothing items such as shirts, shoes, and bags. We started by loading the dataset using TensorFlow and splitting it into training and validation sets. The pixel values were normalized from 0–255 to a 0–1 range to ensure compatibility with neural network inputs. We also added an additional channel dimension to the images, preparing them for TensorFlow operations.
A **custom show_data() function** was implemented to display images in a visually appealing format. Using this function, the first three images from the validation set were displayed in grayscale, side by side, without axes. This task helped in gaining an intuitive understanding of the dataset and its structure before feeding it into a model.

**Output:** A horizontal display of three sample Fashion-MNIST images, providing grayscale visualization of the types of clothing items present in the dataset.

Task 2: **Training Curves Visualization**
The second task focused on **simulating and visualizing model training progress**. Instead of training a full CNN, we generated sample training metrics to demonstrate how loss decreases and accuracy increases as a model learns. Over 10 epochs, loss values were simulated to decrease from 0.8 to 0.12, while accuracy values were set to increase from 72% to 97%.
We created a **dual-axis plot** where the left y-axis represented loss (red line) and the right y-axis represented accuracy (blue line). This allowed both metrics to be visualized simultaneously on a single graph, providing a clear picture of how the model’s performance changes over time. The plot was saved as a PNG file in the Outputs/ folder for easy reference.

**Output:** A single graph showing simulated training progress, demonstrating the typical trends of a learning model and reinforcing the importance of monitoring both loss and accuracy during training.

**TensorFlow** – for dataset handling and preprocessing
**NumPy, Matplotlib** – for data manipulation and visualization

**💡 This Project Demonstrates:**
How to preprocess and visualize image datasets in TensorFlow.
How to simulate and visualize model training progress.
The importance of data visualization in understanding machine learning workflows.
A clean and reproducible project structure for easy experimentation.
