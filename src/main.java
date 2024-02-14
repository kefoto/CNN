import cnn.dta.*;
import cnn.layers.*;

import java.util.List;
import java.util.Random;

import java.util.Collections;
import network.*;
//import nn.DecayingLearningRateSchedule;
//import nn.NeuralNetwork;

//for writing the trained model output weights
import java.nio.file.*;
import java.io.IOException;

public class main {
	public static void main(String[] args) {
		long SEED = new Random().nextLong();
		int epochs = 15;
		int _image_size = 28;

		System.out.println("Start loading data...");

		List<Image> imagesTest = new dataReader().readData("src/data/mnist_test.csv");
		List<Image> imagesTrain = new dataReader().readData("src/data/mnist_train.csv");

		// NeuralNetwork nn_test_load = weightCheckPoint.loadWeights("src/weights",256*100,SEED,0.1,
		// _image_size);
		// System.out.println("Success rate after importing the weight:" +
		// nn_test_load.test(imagesTest));

		System.out.println("images Train size: " + imagesTrain.size());
		System.out.println("images Test size: " + imagesTest.size());

		NetworkBuilder builder = new NetworkBuilder(_image_size, _image_size, 256 * 100);
		
		
		
		//each neuron from the previous convolution layer should have one neuron 10 -> 10
		//may be each neuron is only connected to one neuron but not all of the neuron.
		//TODO: FIX THIS OR FIND OUT ABOUT THE NEURON INPUT (note, all the input size for each layer should be the same)
		//TODO: OR I did not have to fix it, I need to change the dimensionality of 10 parts of 10 kernels.
		
		//maxpool returns 0
		builder.addConvolutionLayer(10, 5, 2, 0.1, SEED);
		builder.addMaxPoolLayer(3, 2);
		
//		builder.addConvolutionLayer(10, 5, 1, 0.1, SEED);
//		builder.addMaxPoolLayer(2, 2);
		builder.addLayer(20, 0.1, SEED);
		builder.addLayer(10, 0.1, SEED);

		NeuralNetwork net = builder.build();

		float rate = net.test(imagesTest);
		System.out.println("Pre training success rate: " + rate);

		for (int i = 0; i < epochs; i++) {
			long startTime = System.currentTimeMillis();
			Collections.shuffle(imagesTrain);
			net.train(imagesTrain);
			rate = net.test(imagesTest);
			long endTime = System.currentTimeMillis();
			System.out.println("Success rate after round " + i + ": " + rate);
			weightCheckPoint.writeWeights("src/weights", net);
			System.out.println("RunTime after round " + i + ": " + ((endTime - startTime)/ 1000));
		}

	}

}
