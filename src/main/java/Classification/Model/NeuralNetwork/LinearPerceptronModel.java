package Classification.Model.NeuralNetwork;

import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Parameter.LinearPerceptronParameter;
import Classification.Parameter.Parameter;
import Classification.Performance.ClassificationPerformance;
import Math.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

public class LinearPerceptronModel extends NeuralNetworkModel implements Serializable {

    protected Matrix W;

    /**
     * Training algorithm for the linear perceptron algorithm. 20 percent of the data is separated as cross-validation
     * data used for selecting the best weights. 80 percent of the data is used for training the linear perceptron with
     * gradient descent.
     *
     * @param train   Training data given to the algorithm
     * @param params Parameters of the linear perceptron.
     */
    public void train(InstanceList train, Parameter params) throws DiscreteFeaturesNotAllowed {
        Vector rMinusY;
        int epoch;
        double learningRate;
        Matrix deltaW, bestW;
        ClassificationPerformance currentClassificationPerformance, bestClassificationPerformance;
        initialize(train);
        LinearPerceptronParameter parameters = (LinearPerceptronParameter) params;
        Partition partition = new Partition(train, parameters.getCrossValidationRatio(), new Random(parameters.getSeed()), true);
        InstanceList trainSet = partition.get(1);
        InstanceList validationSet = partition.get(0);
        W = allocateLayerWeights(K, d + 1, new Random(parameters.getSeed()));
        bestW = W.clone();
        bestClassificationPerformance = new ClassificationPerformance(0.0);
        epoch = parameters.getEpoch();
        learningRate = parameters.getLearningRate();
        for (int i = 0; i < epoch; i++) {
            trainSet.shuffle(parameters.getSeed());
            for (int j = 0; j < trainSet.size(); j++) {
                createInputVector(trainSet.get(j));
                try {
                    rMinusY = calculateRMinusY(trainSet.get(j), x, W);
                    deltaW = rMinusY.multiply(x);
                    deltaW.multiplyWithConstant(learningRate);
                    W.add(deltaW);
                } catch (MatrixColumnMismatch | MatrixDimensionMismatch | VectorSizeMismatch ignored) {
                }
            }
            currentClassificationPerformance = testClassifier(validationSet);
            if (currentClassificationPerformance.getAccuracy() > bestClassificationPerformance.getAccuracy()) {
                bestClassificationPerformance = currentClassificationPerformance;
                bestW = W.clone();
            }
            learningRate *= parameters.getEtaDecrease();
        }
        W = bestW;
    }

    /**
     * Loads the linear perceptron model from an input file.
     * @param fileName File name of the linear perceptron model.
     */
    @Override
    public void loadModel(String fileName) {
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            loadClassLabels(input);
            W = loadMatrix(input);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The calculateOutput method calculates the {@link Matrix} y by multiplying Matrix W with {@link Vector} x.
     */
    protected void calculateOutput() {
        try {
            y = W.multiplyWithVectorFromRight(x);
        } catch (MatrixColumnMismatch ignored) {
        }
    }

    /**
     * Saves the linear perceptron model to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            saveClassLabels(output);
            saveMatrix(output, W);
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
