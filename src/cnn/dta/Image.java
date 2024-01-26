package cnn.dta;

public class Image {
    private double[][] data;

    private int label;

    public int getLabel() {
        return label;
    }

    public double[][] getData() {
        return data;
    }

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

}
