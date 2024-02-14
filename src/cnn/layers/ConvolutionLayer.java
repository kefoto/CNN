package cnn.layers;

import java.util.List;
import java.util.Random;
import java.util.ArrayList;
import cnn.dta.*;
//import cnn.dta;

public class ConvolutionLayer extends Layer{
	private final String _id = "c";
	
	
	private long SEED;
	
	//kernel or kernel size
	private List<double[][]> _filters;
	private int _filtersize;
	private int _stepsize;
	
	private int _inLength;
	private int _inRows;
	private int _inCols;
	private double _learningRate;
	
	//save the previous layer input
	private List<double[][]>_lastInput;
	
	
	
	public ConvolutionLayer(int _filtersize, int _stepsize, int _inLength, int _inRows, int _inCols, long SEED, int numFilters, double _learningRate) {
		this._filtersize = _filtersize;
		this._stepsize = _stepsize;
		this._inLength = _inLength;
		this._inRows = _inRows;
		this._inCols = _inCols;
		this.SEED = SEED;
		this._learningRate = _learningRate;
		
		generateRandomFilter(numFilters);
		
	}
	
	//generate the random weights in the kernel
	private void generateRandomFilter(int numFilters) {
		List<double[][]> filters = new ArrayList<>();
		Random random = new Random(SEED);
		
		for(int n = 0; n < numFilters; n++) {
			
			double[][] newFilter = new double[_filtersize][_filtersize];
			
			for(int i = 0; i < _filtersize; i++) {
				for(int j = 0; j < _filtersize; j++) {
					 
					newFilter[i][j] = random.nextGaussian();
					
				}
			}
			
			filters.add(newFilter);
		}
		
		_filters = filters;
		
	}
	
	public List<double[][]> convolutionForwardPass(List<double[][]> list){
		_lastInput = list;
		
		List<double[][]> output = new ArrayList<>();
		
		for(int m = 0; m < list.size(); m++) {
			for(double[][] filter: _filters) {
				output.add(convolve(list.get(m), filter, _stepsize));
			}
		
		}
		return output;
	}
	
	
	public double[][] convolve(double[][] input, double[][] filter, int stepsize) {
		int outRows = (input.length - filter.length)/stepsize + 1;
		int outCols = (input[0].length - filter[0].length)/stepsize + 1;
		
		int inRows = input.length;
		int inCols = input[0].length;
		
		int fRows = filter.length;
		int fCols = filter[0].length;
		
		double[][] output = new double[outRows][outCols];
		
		int oR = 0;
		int oC;
		
		for(int i = 0; i <= inRows - fRows; i+= stepsize) {
			
			oC = 0;
			
			for(int j = 0; j <= inCols - fCols; j+=stepsize) {
				
				double sum = 0.0;
				//apply filter around
				
				for(int x = 0; x < fRows; x++) {
					for(int  y = 0; y < fCols; y++) {
						int inputRowIndex = i + x;
						int inputColIndex = j + y;
						
						
						double value = filter[x][y] * input[inputRowIndex][inputColIndex];
						sum+= value;
					}
				}
				
				output[oR][oC] = sum;
				oC++;
			}
			
			oR++;
		}
		
		return output;
	}
	
	
	//provide the loop with the forward pass on each output of the layer
	@Override
	public double[] getOutput(List<double[][]> input) {
		List<double[][]> output = convolutionForwardPass(input);
		
		return _next.getOutput(output);
	}

	@Override
	public double[] getOutput(double[] input) {
		List<double[][]> temp = vectorToMatrix(input, _inLength, _inRows, _inCols);
		
		return getOutput(temp);
	}
	
	
	public double[][] spaceArray(double[][] input) {
		if(_stepsize == 1) {
			return input;
		}
		
		int outRows = (input.length - 1) * _stepsize + 1;
		int outCols = (input[0].length - 1) * _stepsize + 1;
		
		double[][] output = new double[outRows][outCols];
		
		for(int i = 0; i < input.length; i++) {
			for(int j = 0; j < input[0].length; j++) {
				output[i*_stepsize][j*_stepsize] = input[i][j];
			}
		}
		
		return output;
	}
	
	
	
	@Override
	public void backPropagation(double[] loss) {
		List<double[][]> result = vectorToMatrix(loss, _inLength, _inRows, _inCols);
		
		backPropagation(result);
		
	}

	
	@Override
	public void backPropagation(List<double[][]> loss) {
		
		List<double[][]> filtersDelta = new ArrayList<>();
		List<double[][]> previousLossX = new ArrayList<>();
		
		//initiate the deltas for each filter
		for(int f = 0; f < _filters.size(); f++) {
			filtersDelta.add(new double[_filtersize][_filtersize]);
		}
		
		//check the last input
		for(int i = 0; i < _lastInput.size(); i++) {
			
			double[][] errorForInput = new double[_inRows][_inCols];
			
			for(int f = 0; f <  _filters.size(); f++) {
				double[][] currentkernel = _filters.get(f);
				//why 
				double[][] error = loss.get(i* _filters.size() + f);
				
				double[][] spacedError = spaceArray(error);
				double[][] dLdF = convolve(_lastInput.get(i), spacedError, 1);
				
				double[][] delta = MatrixUtility.multiply(dLdF, _learningRate * -1);
				double[][] newWeight = MatrixUtility.add(filtersDelta.get(f),  delta);
				filtersDelta.set(f, newWeight);
				
				double[][] flippedError = flip180(spacedError);
				
				errorForInput = MatrixUtility.add(errorForInput, fullConvolve(currentkernel, flippedError));
				
				
			}
			
			previousLossX.add(errorForInput);
		}
		
		for( int f = 0; f < _filters.size(); f++) {
			double[][] modified = MatrixUtility.add(filtersDelta.get(f), _filters.get(f));
			_filters.set(f, modified);
		}
		
		if(_previous != null) {
			_previous.backPropagation(previousLossX);
		}
	}
	
	
	
	public static double[][] flip180(double[][] a){
        double[][] result = new double[a.length][a[0].length];

        for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < a[0].length; j++) {
				result[a.length - 1 - i][a[0].length - j - 1] = a[i][j];
			}
		}
        return result; 
    }
	
	
//	public double[][] flipRows(double[][] a){
//		double[][] result = new double[a.length][a[0].length];
//		
//		for(int i = 0; i < a.length; i++) {
//			for(int j = 0; j < a[0].length; j++) {
//				result[a.length - 1 - i][j] = a[i][j];
//			}
//		}
//		
//		return result;
//	}
//	
//	public double[][] flipCols(double[][] a){
//		double[][] result = new double[a.length][a[0].length];
//		
//		for(int i = 0; i < a.length; i++) {
//			for(int j = 0; j < a[0].length; j++) {
//				result[i][a[0].length - j - 1] = a[i][j];
//			}
//		}
//		
//		return result;
//	}
	
	
	//used in the back-propagation -> fully convolve
	public double[][] fullConvolve(double[][] input, double[][] filter) {
		int outRows = (input.length + filter.length) + 1;
		int outCols = (input[0].length + filter[0].length) + 1;
		
		int inRows = input.length;
		int inCols = input[0].length;
		
		int fRows = filter.length;
		int fCols = filter[0].length;
		
		double[][] output = new double[outRows][outCols];
		
		int oR = 0;
		int oC;
		
		for(int i = -fRows + 1; i < inRows; i++) {
			
			oC = 0;
			
			for(int j = -fCols + 1; j < inCols; j++) {
				
				double sum = 0.0;
				//apply filter around
				
				for(int x = 0; x < fRows; x++) {
					for(int  y = 0; y < fCols; y++) {
						int inputRowIndex = i + x;
						int inputColIndex = j + y;
						
						if(inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < inRows && inputColIndex < inCols) {
							double value = filter[x][y] * input[inputRowIndex][inputColIndex];
							sum += value;
						}
						
					}
				}
				
				output[oR][oC] = sum;
				oC++;
			}
			
			oR++;
		}
		
		return output;
	}
	
	@Override
	public int getOutputLength() {
		return _filters.size() * _inLength;
	}

	@Override
	public int getOutputRows() {
		return (_inRows - _filtersize)/ _stepsize + 1;
	}

	@Override
	public int getOutputCols() {
		return (_inCols - _filtersize)/ _stepsize + 1;
	}

	@Override
	public int getOutputElement() {
		return _inLength * getOutputCols() * getOutputRows();
	}

	public String get_id() {
		return _id;
	}

	public List<double[][]> get_filters() {
		return _filters;
	}

	public void set_filters(List<double[][]> _filters) {
		this._filters = _filters;
	}

	public int get_filtersize() {
		return _filtersize;
	}

	public void set_filtersize(int _filtersize) {
		this._filtersize = _filtersize;
	}

	public int get_stepsize() {
		return _stepsize;
	}

	public void set_stepsize(int _stepsize) {
		this._stepsize = _stepsize;
	}

	public int get_inLength() {
		return _inLength;
	}

	public void set_inLength(int _inLength) {
		this._inLength = _inLength;
	}

	public int get_inRows() {
		return _inRows;
	}

	public void set_inRows(int _inRows) {
		this._inRows = _inRows;
	}

	public int get_inCols() {
		return _inCols;
	}

	public void set_inCols(int _inCols) {
		this._inCols = _inCols;
	}
	
	public void set_Last_input(List<double[][]> _lastInput){
		this._lastInput = _lastInput;
	}

	public List<double[][]> get_Last_input(){
		return _lastInput;
	}
}




//国际商业，傻认真，无知的执着
