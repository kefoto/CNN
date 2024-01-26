import cnn.dta.*;
import cnn.layers.*;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import network.*;
//import nn.DecayingLearningRateSchedule;
//import nn.NeuralNetwork;

//for writing the trained model output weights
import java.nio.file.*;

import java.io.IOException;

public class main {
    public static void main(String[] args) {
    	long SEED = 123;
    	int epochs = 1;
    	int _image_size = 28; 
    	loadWeights("src/weights",1,1,1,1, _image_size);

    	
    	System.out.println("Start loading data...");
    	
    	List<Image> imagesTest = new dataReader().readData("src/data/mnist_test.csv");
    	List<Image> imagesTrain = new dataReader().readData("src/data/mnist_train.csv");
    	
    	System.out.println("images Train size: " + imagesTrain.size());
    	System.out.println("images Test size: " + imagesTest.size());
    	
    	NetworkBuilder builder = new NetworkBuilder(_image_size, _image_size, 256*100);
    	builder.addConvolutionLayer(8, 5, 1, 0.1, SEED);
    	builder.addMaxPoolLayer(3, 2);
    	builder.addLayer(10, 0.1, SEED);
    	
    	NeuralNetwork net = builder.build();
    	
    	float rate = net.test(imagesTest);
    	System.out.println("Pre training success rate: " + rate);
    	
    	
    	for(int i = 0; i < epochs; i++) {
    		Collections.shuffle(imagesTrain);
    		net.train(imagesTrain);
            rate = net.test(imagesTest);
            System.out.println("Success rate after round " + i + ": " + rate);
            writeWeights("src/weights",net);
    	}
    	
    	
    }
    
    
    public static void writeWeights(String folderpath, NeuralNetwork network) {
    		
    	int i = 0;
    	String type = "";
    	//for each layer has a file
    	//each file records a matrix or a list of matrix of the weights
    	
    	
    	//TODO: how to get each layer but different, not the abstract one
    	for(Layer l: network.getLayer()) {
    		
    		ArrayList<String> lines = new ArrayList<>();
//    		String type = l.get_id();
    		
    		
    		if(l instanceof MaxPoolLayer) {
    			
    		/*
    		 * Line 1: the general dimension data for Max Pool Layer
    		 * R for Row, and C for cols With _maxCoodiate matrix dimension on the first two Cells
    		 */
    			
    			MaxPoolLayer mpL = (MaxPoolLayer) l;
    			type = "m";
    			
    			String line = "";
    			lines.add(Integer.toString(mpL.get_stepsize()) + "," + Integer.toString(mpL.get_windowsize()) + "," + 
    				Integer.toString(mpL.get_inLength())+ "," +Integer.toString(mpL.get_inRows()) + "," + 
    				Integer.toString(mpL.get_inCols()));
    			
  
    			lines.add("r");
    			for(int[][] mat: mpL.get_maxRow()) {
    				line = "";
    				line += Integer.toString(mat.length) + "," + Integer.toString(mat[0].length) + ",";
    				for(int[] x: mat) {
    					for(int y: x) {
    						line += Integer.toString(y) + ",";
    					}
    				}
    				if (!line.isEmpty()) {
			            line = line.substring(0, line.length() - 1);
			        }
    				
    				lines.add(line);
    			}
    			
    			lines.add("c");
    			for(int[][] mat: mpL.get_maxCol()) {
    				line = "";
//    				line += Integer.toString(mat.length) + "," + Integer.toString(mat[0].length) + ",";
    				for(int[] x: mat) {
    					for(int y: x) {
    						line += Integer.toString(y) + ",";
    					}
    				}
    				if (!line.isEmpty()) {
			            line = line.substring(0, line.length() - 1);
			        }
    				
    				lines.add(line);
    			}
    			
    		} else if(l instanceof Basic_connected){
    		/*
    		 * When writing the weight for basic nn, we only need two lines: one for dimension, and another for the weights
    		 */
    			Basic_connected bc = (Basic_connected) l;
    			String line = "";
    			type = "b";
    			
    			lines.add(Integer.toString(bc.get_inLength())+ "," + Integer.toString(bc.get_outLength()));
    			
    			for(double[] x: bc.get_weights()) {
    				for(double y: x) {
    					line += Double.toString(y) + ",";
    				}
    			}
    			
    			if (!line.isEmpty()) {
		            line = line.substring(0, line.length() - 1);
		        }
    			
    			lines.add(line);
    			
    			
    		} else if(l instanceof ConvolutionLayer){
    		/*
    		 * 
    		 */
    			
    			type = "c";
    			ConvolutionLayer cvL = (ConvolutionLayer) l;
    			String line = "";
    			
    			lines.add(Integer.toString(cvL.get_filtersize()) + "," + Integer.toString(cvL.get_stepsize()) + 
    					"," + Integer.toString(cvL.get_inLength()) + 
    					"," + Integer.toString(cvL.get_inRows()) +
    					"," + Integer.toString(cvL.get_inCols()));
    			
    			
    			
    			for(double[][] mat: cvL.get_filters()) {
    				line = "";
//    				line += Integer.toString(mat.length) + "," + Integer.toString(mat[0].length) + ",";
    				for(double[] x: mat) {
    					for(double y: x) {
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
    		
    	
    		
    		//for max track the list of coordinates;
    		//for convolution, track the list of matrix;
    		//for nn, track a matrix
    		
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
    public static void loadWeights(String folderPath, int size, int scalar, long SEED, double learningRate, int _image_size) {
    	//the first file will always contain the size of the input
    	
    	NetworkBuilder builder = new NetworkBuilder(_image_size, _image_size, scalar);
    	List<Path> fileList = new ArrayList<>();
    	
    	try (DirectoryStream<Path> directoryStream = Files.newDirectoryStream(Paths.get(folderPath))) {
            for (Path filePath : directoryStream) {
            	
            	fileList.add(filePath);
             
            }
            
            //sort the file sequence first
            Collections.sort(fileList, (file1, file2) -> {
                int sequence1 = Integer.parseInt(String.valueOf(file1.getFileName().toString().charAt(2)));
                int sequence2 = Integer.parseInt(String.valueOf(file2.getFileName().toString().charAt(2)));
                return Integer.compare(sequence1, sequence2);
            });
            
            for (Path sortedFilePath : fileList) {
            	String fileName = sortedFilePath.getFileName().toString();
            	String type = String.valueOf(fileName.charAt(0));
            	
//            	System.out.println("File: " + sortedFilePath.getFileName());
            	
            	if(type.equals("c")) {
            		 List<String> lines = Files.readAllLines(sortedFilePath);
                     
                     // Process each line of the file
                     for (String line : lines) {
                    	 
                    	 String[] weights = line.split(",");
                         // Add your logic for processing each line
                         System.out.println(line);
                     }
                     
//            		builder.addConvolutionLayer(8, 5, 1, 0.1, SEED);
            		
            	} else if(type.equals("m")) {
            			
            			//stepsize, windowsize, in length, in row, in col
            		
            		List<String> lines = Files.readAllLines(sortedFilePath);
                    
                    // Process each line of the file
                    for (String line : lines) {
                   	 
                   	 String[] weights = line.split(",");
                        // Add your logic for processing each line
                        System.out.println(line);
                    }
                    
//            		builder.addMaxPoolLayer(3, 2);
            		
            	} else if(type.equals("b")) {
            		
            		List<String> lines = Files.readAllLines(sortedFilePath);
                    
                    // Process each line of the file
                    for (String line : lines) {
                   	 
                   	 String[] weights = line.split(",");
                        // Add your logic for processing each line
                        System.out.println(line);
                    }
                    
                    
//            		builder.addLayer(10, 0.1, SEED);
            		
            	} else {
            		//error message
            	}
            }
        } catch (Exception e) {
        	
            // Handle exceptions, such as IOException
            e.printStackTrace();
        }
    	
    }
}
