package network;

import cnn.layers.*;
import cnn.dta.*;
import java.util.List;
import java.util.ArrayList;

public class NeuralNetwork {

	List<Layer> _layers;
	double scaleFactor;
	
	public NeuralNetwork(List<Layer> _layers, double scaleFactor) {
		this._layers = _layers;
		set_Link_layers();
		this.scaleFactor = scaleFactor;
	}
	
	private void set_Link_layers() {
		if(_layers.size() <= 1) {
			return;
		}
		
		
		for(int i = 0; i < _layers.size(); i++) {
			if(i == 0) {
				_layers.get(i).set_next(_layers.get(i+1));
			} else if(i == _layers.size() - 1) {
				_layers.get(i).set_previous(_layers.get(i-1));
			} else {
				_layers.get(i).set_next(_layers.get(i+1));
				_layers.get(i).set_previous(_layers.get(i-1));
			}
		}
		
	}
	
	public double[] getErrors(double[] nnoutput, int correct_answer) {
		int numClasses = nnoutput.length;
		
		double[] expected = new double[numClasses];
		
		expected[correct_answer] = 1;
		
		return MatrixUtility.add(nnoutput, MatrixUtility.multiply(expected, -1));
		
	}
	
	private int getMax(double[] a) {
		double max = 0;
		int result = 0;
		
		for(int i = 0; i < a.length; i++) {
			if(a[i] >= max) {
				max = a[i];
				result = i;
			}
		}
		
		return result;
		
		
	}
	
	public int guess(Image image) {
		List<double[][]> inputList = new ArrayList<>();
		inputList.add(MatrixUtility.multiply(image.getData(), (1.0/scaleFactor)));
		
		double[] output = _layers.get(0).getOutput(inputList);
		int guess = getMax(output);
		
		return guess;
	}
	
	public float test(List<Image> images) {
		int correct = 0;
		
		for(Image img: images) {
			int guess = guess(img);
			
			if(guess == img.getLabel()) {
				correct++;
			}
		}
		
		float result = ((float)correct/images.size());
		
		return result;
	}
	
	public void train (List<Image> images) {
		for(Image img: images) {
			List<double[][]> inputList = new ArrayList<>();
			inputList.add(MatrixUtility.multiply(img.getData(), (1.0/scaleFactor)));
			
			double[] output = _layers.get(0).getOutput(inputList);
			double[] errorLoss = getErrors(output, img.getLabel());
			
			_layers.get((_layers.size() - 1)).backPropagation(errorLoss);
		}
		
	}
	
	public List<Layer> getLayer() {
		return _layers;
	}
	
}
