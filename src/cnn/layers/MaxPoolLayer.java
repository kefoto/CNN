package cnn.layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer {
	private final String _id = "m";
	
	private int _stepsize;
	private int _windowsize;
	
	
	private int _inLength;
	private int _inRows;
	private int _inCols;
	
	//tracking the x and y coordinates in the current layer
	List<int[][]> _maxRow;
	List<int[][]> _maxCol;
	
	public MaxPoolLayer(int _stepsize, int _windowsize, int _inLength, int _inRows, int _inCols) {
		this._stepsize = _stepsize;
		this._windowsize = _windowsize;
		this._inLength = _inLength;
		this._inRows = _inRows;
		this._inCols = _inCols;
		
	}
	
	
	public List<double[][]> maxPoolForwardPass(List<double[][]> input){
		List<double[][]> output = new ArrayList<>();
		 _maxRow = new ArrayList<>();
		 _maxCol = new ArrayList<>();
		
		for (int l = 0; l < input.size(); l++) {
			output.add(pool(input.get(l)));
		}
		
		return output;
	}
	
//	helper function for forward pass
	public double[][] pool(double[][] input){
		
		double[][] result = new double[getOutputRows()][getOutputCols()];
		
		int[][] tempRows = new int[getOutputRows()][getOutputCols()];
		int[][] tempCols = new int[getOutputRows()][getOutputCols()];
		
		
		for(int r = 0; r < getOutputRows(); r += _stepsize) {
			for(int c = 0; c < getOutputCols(); c+= _stepsize) {
				double max = 0.0;
				
				
				for(int x = 0; x < _windowsize; x++) {
					for(int y = 0; y < _windowsize; y++) {
						if(max < input[r+x][c+y]) {
							max = input[r+x][c+y];
							
							tempRows[r][c] = r+x;
							tempCols[r][c] = c+y;
						}
						
					}
				}
				
				result[r][c] = max;
			}
		}
		
		_maxRow.add(tempRows);
		_maxCol.add(tempCols);
		
		return result;
	}
	@Override
	public double[] getOutput(List<double[][]> input) {
		List<double[][]> output = maxPoolForwardPass(input);
		
		
		//should provide a catch exception
//		if(_next != null) {
			return _next.getOutput(output);
//		} else {
//			return output;
//		}
	}
	
	//TODO: THe output returns and array of 100 number maxpool output instead of 10
	@Override
	public double[] getOutput(double[] input) {
		List<double[][]> temp = vectorToMatrix(input, _inLength, _inRows, _inCols);
		
		return getOutput(temp);
	}

	@Override
	public void backPropagation(double[] loss) {
		List<double[][]> temp = vectorToMatrix(loss, getOutputLength(), getOutputRows(), getOutputCols());
		backPropagation(temp);
	}

	@Override
	public void backPropagation(List<double[][]> loss) {
		
		
		List<double[][]> dXdL = new ArrayList<>();
		int l = 0;
		for(double[][] arr: loss) {
			double[][] error = new double[_inRows][_inCols];
			
			for(int r = 0; r < getOutputRows(); r++) {
				for(int c = 0; c < getOutputCols(); c++) {
					int max_i = _maxRow.get(l)[r][c];
					int max_j = _maxCol.get(l)[r][c];
					
					if(max_i != -1) { 
						error[max_i][max_j] += arr[r][c];
					}
				}
			}
			
			dXdL.add(error);
			l++;
		}
		
		if(_previous != null)
		{
			_previous.backPropagation(dXdL);
		}
	}

	@Override
	public int getOutputLength() {
		return _inLength;
	}

	@Override
	public int getOutputRows() {
		return (_inRows - _windowsize)/_stepsize + 1;
	}

	@Override
	public int getOutputCols() {

		return (_inCols - _windowsize)/_stepsize + 1;
	}

	@Override
	public int getOutputElement() {
		return _inLength * getOutputCols() * getOutputRows();
	}
	
	public String get_id() {
		return _id;
	}


	public int get_stepsize() {
		return _stepsize;
	}


	public void set_stepsize(int _stepsize) {
		this._stepsize = _stepsize;
	}


	public int get_windowsize() {
		return _windowsize;
	}


	public void set_windowsize(int _windowsize) {
		this._windowsize = _windowsize;
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


	public List<int[][]> get_maxRow() {
		return _maxRow;
	}


	public void set_maxRow(List<int[][]> _maxRow) {
		this._maxRow = _maxRow;
	}


	public List<int[][]> get_maxCol() {
		return _maxCol;
	}


	public void set_maxCol(List<int[][]> _maxCol) {
		this._maxCol = _maxCol;
	}

}
