package cnn.layers;

import java.util.List;
import java.util.Random;

public class Basic_connected extends Layer {
	
	private final String _id = "n";
	
    private double[][] _weights;
    private int _inLength;
    private int _outLength;
    private double _learningRate;
    
    //to adjust the reLu activation function;
    private final double leak = 0.01;
    private long SEED;
    
    //tracking the Z and the input of the current layer
    private double[] z;
    private double[] x;


    public Basic_connected(int _inLength, int _outLength, long SEED, double _learningRate){
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;
        this._learningRate = _learningRate;
        
        
        _weights = new double[_inLength][_outLength];
        setRandomWeights();
    
    }
    
    public double[] basicForwardPass(double[] input) {
    	x = input;
    	double[] output = new double[_outLength];
    	double[] zCurrent = new double[_outLength];

        for(int j = 0; j < _outLength; j++){
            double sum = 0.0;
            for(int i = 0; i < _inLength; i++){
                sum += input[i]*_weights[i][j];
            }
            zCurrent[j] = sum;
            output[j] = reLu(zCurrent[j]);

        }
        
        z = zCurrent;

        return output;
    	
    }


    @Override
    public double[] getOutput(List<double[][]> input) {
    	double[] vector = matrixToVector(input);
    	
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
    	double[] forwardPass = basicForwardPass(input);
    	
    	if(_next != null) {
    		return _next.getOutput(forwardPass);
    	} else {
    		return forwardPass;
    	}
    }
    
    //loss is captured as dL/dO
    @Override
    public void backPropagation(double[] loss) {
        
    	double[] dLdx = new double[_inLength];
    	
    	double dOdz;
    	double dzdw;
    	double dLdw;
    	double dzdx;
    	
    	for(int k = 0; k < _inLength; k++) {
    		
    		double dLdx_sum = 0;
    		
    		for(int j = 0; j < _outLength; j++) {
    			
    			dOdz = reLu_d(z[j]);
    			dzdw = x[k];
    			dzdx = _weights[k][j];
    			
    			dLdw = loss[j] * dOdz * dzdw;
    			
    			_weights[k][j] -= dLdw * _learningRate;
    			
    			
    			dLdx_sum += loss[j]*dOdz*dzdx;
    			
    		}
    		
    		dLdx[k] = dLdx_sum;
    	}
    	if(_previous != null) {
    		_previous.backPropagation(dLdx);
    	}
    	
    }

    @Override
    public void backPropagation(List<double[][]> loss) {
    	double[] vector = matrixToVector(loss);
    	
    	backPropagation(vector);
    }

    @Override
    public int getOutputLength() {
       return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
       return 0;
    }

    @Override
    public int getOutputElement() {
        return _outLength;
    }
    
    
    public void setRandomWeights() {
    	Random random = new Random(SEED);
    	
    	for(int i = 0; i < _inLength; i++) {
    		for(int j = 0; j < _outLength; j++) {
    			//adding Math.sqrt is to make sure the weight falls between [-1... 1) range 
    			//based on the input and output length
    			_weights[i][j] = random.nextGaussian() * Math.sqrt(2.0/(_inLength + _outLength));    			
    		}
    	}
    	
    }
    
    public double reLu(double input) {
    	if(input <= 0) {
    		return 0;
    	} else {
    		return input;
    	}
    }
    
    public double reLu_d(double input) {
    	if(input <= 0) {
    		return leak;
    	} else {
    		return 1;
    	}
    }
    
    public String get_id() {
		return _id;
	}

	public double[][] get_weights() {
		return _weights;
	}

	public void set_weights(double[][] _weights) {
		this._weights = _weights;
	}

	public int get_inLength() {
		return _inLength;
	}

	public void set_inLength(int _inLength) {
		this._inLength = _inLength;
	}

	public int get_outLength() {
		return _outLength;
	}

	public void set_outLength(int _outLength) {
		this._outLength = _outLength;
	}

}
