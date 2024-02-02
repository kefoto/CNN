package cnn.dta;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import cnn.layers.Basic_connected;
import cnn.layers.ConvolutionLayer;
import cnn.layers.Layer;
import cnn.layers.MaxPoolLayer;
import network.NetworkBuilder;
import network.NeuralNetwork;

public class weightCheckPoint {
    public static void writeWeights(String folderpath, NeuralNetwork network) {

		int i = 0;
		String type = "";
		// for each layer has a file
		// each file records a matrix or a list of matrix of the weights

		for (Layer l : network.getLayer()) {

			ArrayList<String> lines = new ArrayList<>();
			// String type = l.get_id();

			if (l instanceof MaxPoolLayer) {
				/*
				 * Line 1: the general dimension data for Max Pool Layer
				 * R for Row, and C for cols With _maxCoodiate matrix dimension on the first two
				 * Cells
				 */

				MaxPoolLayer mpL = (MaxPoolLayer) l;
				type = "m";

				String line = "";
				lines.add(Integer.toString(mpL.get_stepsize()) + "," + Integer.toString(mpL.get_windowsize()) + "," +
						Integer.toString(mpL.get_inLength()) + "," + Integer.toString(mpL.get_inRows()) + "," +
						Integer.toString(mpL.get_inCols()));

				for (int[][] mat : mpL.get_maxRow()) {
					line = "";
					for (int[] x : mat) {
						for (int y : x) {
							line += Integer.toString(y) + ",";
						}
					}
					if (!line.isEmpty()) {
						line = line.substring(0, line.length() - 1);
					}

					lines.add(line);
				}
				lines.add("r");

				for (int[][] mat : mpL.get_maxCol()) {
					line = "";
					// line += Integer.toString(mat.length) + "," + Integer.toString(mat[0].length)
					// + ",";
					for (int[] x : mat) {
						for (int y : x) {
							line += Integer.toString(y) + ",";
						}
					}
					if (!line.isEmpty()) {
						line = line.substring(0, line.length() - 1);
					}

					lines.add(line);
				}
				lines.add("c");

			} else if (l instanceof Basic_connected) {
				/*
				 * When writing the weight for basic nn, we only need two lines: one for
				 * dimension, and another for the weights
				 */
				Basic_connected bc = (Basic_connected) l;
				String line = "";
				type = "b";

				lines.add(Integer.toString(bc.get_inLength()) + "," + Integer.toString(bc.get_outLength()));

				for (double[] x : bc.get_weights()) {
					for (double y : x) {
						line += Double.toString(y) + ",";
					}
				}

				if (!line.isEmpty()) {
					line = line.substring(0, line.length() - 1);
				}

				lines.add(line);
				line = "x,";
				for (double y : bc.get_x()) {
					line += Double.toString(y) + ",";
				}
				if (!line.isEmpty()) {
					line = line.substring(0, line.length() - 1);
				}
				lines.add(line);
				line = "z,";

				for (double y : bc.get_z()) {
					line += Double.toString(y) + ",";
				}
				if (!line.isEmpty()) {
					line = line.substring(0, line.length() - 1);
				}
				lines.add(line);
				line = "";

			} else if (l instanceof ConvolutionLayer) {
				type = "c";
				ConvolutionLayer cvL = (ConvolutionLayer) l;
				String line = "";

				lines.add(Integer.toString(cvL.get_filtersize()) + "," + Integer.toString(cvL.get_stepsize()) +
						"," + Integer.toString(cvL.get_inLength()) +
						"," + Integer.toString(cvL.get_inRows()) +
						"," + Integer.toString(cvL.get_inCols()));

				for (double[][] mat : cvL.get_filters()) {
					line = "";
					// line += Integer.toString(mat.length) + "," + Integer.toString(mat[0].length)
					// + ",";
					for (double[] x : mat) {
						for (double y : x) {
							line += Double.toString(y) + ",";
						}
					}
					if (!line.isEmpty()) {
						line = line.substring(0, line.length() - 1);
					}

					lines.add(line);
				}

			} else {
				System.err.println("Error writing to the file: " + l + "does not exist or is null");
			}

			// for max track the list of coordinates;
			// for convolution, track the list of matrix;
			// for nn, track a matrix

			String tempPath = String.format("%s/%s_%s.csv", folderpath, type, i);
			Path filepath = Path.of(tempPath);

			try {
				try {
					Files.deleteIfExists(filepath);
				} catch (IOException e) {
					e.printStackTrace();
				}
				// Write the lines to the file
				Files.write(filepath, lines, StandardOpenOption.CREATE, StandardOpenOption.WRITE);

				System.out.println("Data has been written to the file successfully.");
			} catch (IOException e) {
				System.err.println("Error writing to the file: " + e.getMessage());
			}

			i++;
		}

	}

	public static NeuralNetwork loadWeights(String folderPath, int scalar, long SEED, double learningRate,
			int _image_size) {
		// the first file will always contain the size of the input

		NetworkBuilder builder = new NetworkBuilder(_image_size, _image_size, scalar);
		List<Path> fileList = new ArrayList<>();

		try (DirectoryStream<Path> directoryStream = Files.newDirectoryStream(Paths.get(folderPath))) {
			for (Path filePath : directoryStream) {

				fileList.add(filePath);

			}

			// sort the file sequence first
			Collections.sort(fileList, (file1, file2) -> {
				int sequence1 = Integer.parseInt(String.valueOf(file1.getFileName().toString().charAt(2)));
				int sequence2 = Integer.parseInt(String.valueOf(file2.getFileName().toString().charAt(2)));
				return Integer.compare(sequence1, sequence2);
			});

			for (Path sortedFilePath : fileList) {
				String fileName = sortedFilePath.getFileName().toString();
				String type = String.valueOf(fileName.charAt(0));
				int index = Integer.parseInt(String.valueOf(fileName.charAt(2)));

				if (type.equals("c")) {
					List<String> lines = Files.readAllLines(sortedFilePath);

					List<double[][]> filter = new ArrayList<>();

					String firstLine = lines.get(0);
					List<String> restOfLines = lines.subList(1, lines.size());

					String[] inputs = firstLine.split(",");

					int filter_size = Integer.parseInt(inputs[0]);

					// int numFilters, int filterSize, int stepSize, double learningRate, long SEED
					// since we will set the filter list, we don't need the number of filters to
					// randomly generate filters.
					builder.addConvolutionLayer(restOfLines.size(), filter_size, Integer.parseInt(inputs[1]),
							learningRate, SEED);
					ConvolutionLayer tempL = (ConvolutionLayer) builder.getLayers().get(index);

					// Process each line of the file
					for (String line : restOfLines) {
						double[][] temp = new double[filter_size][filter_size];
						List<String> weightsList = new ArrayList<>(Arrays.asList(line.split(",")));

						
							int i = 0;
							for (int r = 0; r < filter_size; r++) {
								for (int c = 0; c < filter_size; c++) {
									temp[r][c] = Double.parseDouble(weightsList.get(i));
									i++;
								}
							}
							filter.add(temp);
					}
					tempL.set_filters(filter);
					builder.getLayers().set(index, tempL);

				} else if (type.equals("m")) {
					List<String> lines = Files.readAllLines(sortedFilePath);

					List<int[][]> maxRow_or_col = new ArrayList<>();
					// stepsize, windowsize, in length, in row, in col

					String firstLine = lines.get(0);
					List<String> restOfLines = lines.subList(1, lines.size());

					String[] inputs = firstLine.split(",");

					int outputdimension = (_image_size - Integer.parseInt(inputs[1])) / Integer.parseInt(inputs[0]) - 1;

					builder.addMaxPoolLayer(Integer.parseInt(inputs[1]), Integer.parseInt(inputs[0]));
					MaxPoolLayer tempL = (MaxPoolLayer) (builder.getLayers().get(index));
					// Process each line of the file
					for (String line : restOfLines) {
						int[][] temp = new int[outputdimension][outputdimension];

						String[] weights = line.split(",");
						if (weights[0].equals("r")) {
							tempL.set_maxRow(maxRow_or_col);
							maxRow_or_col = new ArrayList<>();
						} else if (weights[0].equals("c")) {
							tempL.set_maxCol(maxRow_or_col);
							maxRow_or_col = new ArrayList<>();
						} else {
							int i = 0;

							for (int r = 0; r < outputdimension; r++) {
								for (int c = 0; c < outputdimension; c++) {
									temp[r][c] = Integer.parseInt(weights[i]);
									i++;
								}
							}
							maxRow_or_col.add(temp);

						}
					}
					// TODO: Check if getLayers would change tghe things in the layers
					builder.getLayers().set(index, tempL);

				} else if (type.equals("b")) {

					List<String> lines = Files.readAllLines(sortedFilePath);

					String firstLine = lines.get(0);
					List<String> restOfLines = lines.subList(1, lines.size());

					String[] inputs = firstLine.split(",");

					int in_length = Integer.parseInt(inputs[0]);
					int out_length = Integer.parseInt(inputs[1]);

					double[][] weight_bc = new double[in_length][out_length];
					builder.addLayer(out_length, learningRate, SEED);
					Basic_connected tempL = (Basic_connected) (builder.getLayers().get(index));
					// Process each line of the file
					int i = 0;
					for (String line : restOfLines) {

						List<String> weightsList = new ArrayList<>(Arrays.asList(line.split(",")));

						if (weightsList.get(0).equals("x")) {
							weightsList = weightsList.subList(1, weightsList.size());

							double[] result = new double[weightsList.size()];

							for (int j = 0; j < result.length; j++) {
								result[j] = Double.parseDouble(weightsList.get(j));
							}

							tempL.set_x(result);
						} else if (weightsList.get(0).equals("z")) {
							weightsList = weightsList.subList(1, weightsList.size());

							double[] result = new double[weightsList.size()];

							for (int j = 0; j < result.length; j++) {
								result[j] = Double.parseDouble(weightsList.get(j));
							}
							tempL.set_z(result);
						} else {
							for (int r = 0; r < weight_bc.length; r++) {
								for (int c = 0; c < weight_bc[0].length; c++) {
									weight_bc[r][c] = Double.parseDouble(weightsList.get(i));
									i++;
								}
							}
						}
						// Add your logic for processing each line
					}
					tempL.set_weights(weight_bc);
					builder.getLayers().set(index, tempL);

					// builder.addLayer(10, 0.1, SEED);

				} else {
					System.out.println("fail to read weight files");
				}
			}

			NeuralNetwork nn = builder.build();

			return nn;

		} catch (Exception e) {

			// Handle exceptions, such as IOException
			e.printStackTrace();
		}
		return null;
	}
}
