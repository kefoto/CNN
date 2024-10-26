Author: Ke Xu

This Convolutional Neural Network consists of the Convolutional Layer, the MaxPool Layer, and the fully connected Layer to predict the MNIST Dataset in a CSV form.

The self-implemented Convolutional Neural Network model is based on the short YouTube Series by Rae, yet I improved on it by reducing spacetime and runtime.

I had another idea for this small project: I stored the model data and let users interact by drawing a number with a computer guessing the drawing after each stroke.
Others have done it using Python and implemented ResNet to accurately guess each drawn number, yet my model is lacking and needs an algorithm to locate the drawn number. 

To achieve this, JPanel tracks the mouse movement, stores the canvas as PNG after each paint stroke, and converts the PNG into a matrix with pixel values from 0 to 255.

The output accuracy is around 85%, meaning there is much room to improve.
The Current Model is still unfinished with the next idea being the Batch Normalization layer.

To build the main class, you can recompile the class by directory in the folder:

`javac javac nn/*.java math/util/*.java Main.java`

To Run nn: 

`java Main.java [Mode] [epochs/steps]`


Reference:

This CNN is based on the short YouTube series by Rae:
https://youtu.be/3MMonOWGe0M?si=Vne1G2SrE71CE1hz

3blue1brown:
https://www.youtube.com/watch?v=KuXjwB4LzSA&ab_channel=3Blue1Brown

MIT Lecture:
https://www.youtube.com/watch?v=NmLK_WQBxB4&ab_channel=AlexanderAmini

The conversion from Image into a list of integer matrices in CSV:
https://pjreddie.com/projects/mnist-in-csv/


Written on Macintosh, Implemented in Java
