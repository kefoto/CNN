package cnn.layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {

	protected Layer _next;
    protected Layer _previous;

    

    public abstract double[] getOutput(List<double[][]> input);

    public abstract double[] getOutput(double[] input);

   



    public abstract void backPropagation(double[] loss);

    public abstract void backPropagation(List<double[][]> loss);

    public double[] matrixToVector(List<double[][]> input){
        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] vector = new double[length*rows*cols];

        int i = 0;

        for(int l = 0; l < length; l++){
            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    vector[i] = input.get(l)[r][c];
                    i++;
                }
            }
        }

        return vector;
    }

    public List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols){
        List<double[][]> result = new ArrayList<>();
        
        int i = 0;
        for(int l = 0; l < length; l++){

            double[][] matrix = new double[rows][cols];
            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    matrix[r][c] = input[i];
                    i++;
                }
            }

            result.add(matrix);
        }

        return result;
    }


    public Layer get_next() {
        return _next;
    }

    public void set_next(Layer _next) {
        this._next = _next;
    }

    public Layer get_previous() {
        return _previous;
    }

    public void set_previous(Layer _previous) {
        this._previous = _previous;
    }

    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();
    public abstract int getOutputElement();
}