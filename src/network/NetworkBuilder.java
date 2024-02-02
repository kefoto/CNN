package network;


import cnn.layers.*;
import java.util.List;
import java.util.ArrayList;

public class NetworkBuilder {
	
	
	private NeuralNetwork nn;
	private int _inputRows;
	private int _inputCols;
	private double scaleFactor;
	List<Layer> _layers;
	
	public NetworkBuilder(int _inputRows, int _inputCols, double scaleFactor){
		this._inputCols = _inputCols;
		this._inputRows = _inputRows;
		this.scaleFactor = scaleFactor;
		_layers = new ArrayList<>();
	}
	
	public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long SEED) {
		if(_layers.isEmpty()) {
			_layers.add(new ConvolutionLayer(filterSize, stepSize, 1, _inputRows, _inputCols, SEED, numFilters, learningRate));
		} else {
			
			Layer prev = _layers.get(_layers.size() - 1);
			_layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), SEED, numFilters, learningRate));
		
		}
	}
	
	public void addMaxPoolLayer(int windowsize, int stepsize) {
		
		if(_layers.isEmpty()) {
			_layers.add(new MaxPoolLayer(stepsize, windowsize, 1, _inputRows, _inputCols));
		
		} else {
			Layer prev = _layers.get(_layers.size() - 1);
			_layers.add(new MaxPoolLayer(stepsize, windowsize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputRows()));
		
		}
		
	}
	

	public void addLayer(int outLength, double learningRate, long SEED) {
		if(_layers.isEmpty()) {
			_layers.add(new Basic_connected((_inputCols*_inputRows), outLength, SEED, learningRate));
			
		} else {
			Layer prev = _layers.get(_layers.size() - 1);
			_layers.add(new Basic_connected(prev.getOutputElement(), outLength, SEED, learningRate));
			
		}
	}
	
	public NeuralNetwork build() {
		nn = new NeuralNetwork(_layers, scaleFactor);
		return nn;
	}
	
	public List<Layer> getLayers() {
		return _layers;
	}
	

}
