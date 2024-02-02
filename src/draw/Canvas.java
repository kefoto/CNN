package draw;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import network.NeuralNetwork;
import java.util.Random;
import cnn.dta.weightCheckPoint;
import network.NetworkBuilder;



//TODO: find a way to center the image just like the data set
public class Canvas extends JFrame {
    private JPanel drawingPanel;
    private int dotSize = 30; // Adjust the size of the dots as needed
    private Color dotColor = Color.black; // Set the color of the dots
    private Color bgColor = Color.white;
    private boolean isCtrlPressed = false;
    List<List<Point>> points;
    List<Point> stroke;
    static NeuralNetwork nn;

    int targetWidth = 28;
    int targetHeight = 28;

    public Canvas() {
        setTitle("Drawing Canvas");
        setSize(400, 500);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);

        drawingPanel = new JPanel(){
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                drawDots(g);
                // Graphics2D g2d = (Graphics2D) g;
            }
        };

        drawingPanel.setBackground(new Color(dotColor.getRed(), dotColor.getGreen(), dotColor.getBlue(), 255));
        points = new ArrayList<>();

        // Create a list to store the points where the dots will be drawn
        

        // Add mouse listeners for drawing
        drawingPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                // Add the initial point to the list
                stroke = new ArrayList<>();
                stroke.add(e.getPoint());
                points.add(stroke);
                repaint();
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                // points.add(stroke);
                // System.out.println(points.size());
                isCtrlPressed = false;
                saveDrawing();
            }
        });

        drawingPanel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                stroke.add(e.getPoint());
                repaint();
            }
        });

        drawingPanel.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_CONTROL) {
                    isCtrlPressed = true;
                    // System.err.println("CRTL PRESSED");
                } else if (isCtrlPressed && e.getKeyCode() == KeyEvent.VK_Z) {
                    // System.err.println("CRTL Z PRESSED");
                    undo();
                    repaint();
                    saveDrawing();
                }
            }

            @Override
            public void keyReleased(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_CONTROL) {
                    isCtrlPressed = false;

                }
            }
        });

        JButton clearButton = new JButton("Clear");
        // clearButton.setBounds(10, 10, 100, 50);
        clearButton.setPreferredSize(new Dimension(100, 50));
        clearButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                clearDrawing();
                repaint();
                // saveDrawing();
            }
        });

        getContentPane().setLayout(new BorderLayout());
        drawingPanel.setFocusable(true);
        getContentPane().add(drawingPanel, BorderLayout.CENTER);
        getContentPane().add(clearButton, BorderLayout.SOUTH);
    }


    private void drawDots(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setColor(new Color(bgColor.getRed(), bgColor.getGreen(), bgColor.getBlue(), 255));
        g2d.fillRect(0, 0, getWidth(), getHeight()); 
        // g2d.drawImage(bufferImage, 0, 0, this);

        for(List<Point> stroke: points){
            for (Point point : stroke) {
                int x = (int) point.getX();
                int y = (int) point.getY();
                g2d.setColor(new Color(dotColor.getRed(), dotColor.getGreen(), dotColor.getBlue(), 255));
                g2d.fillOval(x - dotSize / 2, y - dotSize / 2, dotSize, dotSize);
            }
        }  
        // g2d.drawImage(bufferImage, 0, 0, this);
        g2d.dispose();
        repaint();
    }


    @Override
    public void paint(Graphics g) {
        super.paint(g);
    }

    private void undo() {
        if (!points.isEmpty()) {
            points.remove(points.size() - 1);
            // clearBufferImage();
            // redrawPoints();
        }
    }
    private void clearDrawing() {
        points.clear();
    }

    private void saveDrawing() {
        BufferedImage image = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_ARGB);
        

        Graphics g = image.getGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, image.getWidth(), image.getHeight());
        paint(g);
        image = cropImage(image, 10, 30, 380, 380);

        image = ImageToMatrix.resizeImage(image, targetWidth, targetHeight);
        double[][] imageToGuess = ImageToMatrix.imageToMatrix(image);
        System.out.println("Computers Guess:" + nn.guess(imageToGuess));

        try {
            File outputFile = new File("src/draw/snapshot/drawing.png");
            ImageIO.write(image, "png", outputFile);
            // System.out.println("File Path: " + outputFile.getAbsolutePath());

            // JOptionPane.showMessageDialog(this, "Drawing saved successfully!");
        } catch (IOException ex) {
            ex.printStackTrace();
            // JOptionPane.showMessageDialog(this, "Error saving drawing", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    // private void saveImageMatrix(String pathname) {

    // }

    private BufferedImage cropImage(BufferedImage originalImage, int x, int y, int width, int height) {
        // Create a new BufferedImage with the specified width and height
        BufferedImage croppedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

        // Get the graphics context of the new image and draw the cropped part of the original image
        Graphics2D g2d = croppedImage.createGraphics();
        g2d.drawImage(originalImage.getSubimage(x, y, width, height), 0, 0, null);
        g2d.dispose();

        return croppedImage;
    }



    public static void main(String[] args) {
        nn = weightCheckPoint.loadWeights("src/weights",256*100, new Random().nextLong(),0.1,
		28);
        SwingUtilities.invokeLater(() -> {
            Canvas canvas = new Canvas();
            canvas.setVisible(true);
        });
    }
}
