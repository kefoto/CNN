package draw;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class ImageToMatrix {

    public static double[][] imageToMatrix(BufferedImage img){
        int width = img.getWidth();
        int height = img.getHeight();
        double[][] imageArray = new double[width][height];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                // Get the RGB value at each pixel
                int rgb = img.getRGB(x, y);

                // Extract individual color components (red, green, blue)
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;
                double grayscale = convertToGrayscale(red, green, blue);
                 // Store the grayscale value in the array
                 //the save function for the mnist dataset
                imageArray[x][y] = invert(grayscale);
            }
        }
        return imageArray;
        // Print the dimensions and values of the array (for demonstration purposes)         
    }

    private static double invert(double grey){
        return 255 - grey;
    }

    private static double convertToGrayscale(int red, int green, int blue) {
        return 0.299 * red + 0.587 * green + 0.114 * blue;
    }

    //for image resizing
    public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_ARGB);

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                // Calculate the average color in the region
                int avgColor = calculateAverageColor(originalImage, x * originalImage.getWidth() / targetWidth,
                        y * originalImage.getHeight() / targetHeight,
                        (x + 1) * originalImage.getWidth() / targetWidth,
                        (y + 1) * originalImage.getHeight() / targetHeight);

                resizedImage.setRGB(x, y, avgColor);
            }
        }

        return resizedImage;
    }

    private static int calculateAverageColor(BufferedImage image, int startX, int startY, int endX, int endY) {
        int totalRed = 0, totalGreen = 0, totalBlue = 0;

        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                Color color = new Color(image.getRGB(x, y));
                totalRed += color.getRed();
                totalGreen += color.getGreen();
                totalBlue += color.getBlue();
            }
        }

        int numPixels = (endX - startX) * (endY - startY);
        int avgRed = totalRed / numPixels;
        int avgGreen = totalGreen / numPixels;
        int avgBlue = totalBlue / numPixels;

        return new Color(avgRed, avgGreen, avgBlue).getRGB();
    }

    public static void main(String[] args) throws IOException {
        int targetWidth = 28;
        int targetHeight = 28;
        try {
            // Load the original image
            BufferedImage originalImage = ImageIO.read(new File("drawing.png"));

            BufferedImage resizedImage = resizeImage(originalImage, targetWidth, targetHeight);

            // Save the resized image
            ImageIO.write(resizedImage, "png", new File("compressedrawing.png"));

        } catch (IOException e) {
            e.printStackTrace();
        }
            // Read an image
        try {
            File file = new File("src/draw/snapshot/drawing.png");
            BufferedImage img = ImageIO.read(file);
            double[][] imagetest = imageToMatrix(img);

            System.out.println("Image Dimensions: Width = " +imagetest.length + ", Height = " + imagetest[0].length);
            for (int y = 0; y < imagetest.length; y++) {
                for (int x = 0; x < imagetest[0].length; x++) {
                    System.out.print(imagetest[x][y] + " ");
                }
                System.out.println();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }      
    }
}



