package Classification.Model.NeuralNetwork;

import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Parameter.ActivationFunction;
import Classification.Parameter.MultiLayerPerceptronParameter;
import Classification.Parameter.Parameter;
import Classification.Performance.ClassificationPerformance;
import Math.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

public class MultiLayerPerceptronModel extends LinearPerceptronModel implements Serializable {
    private Matrix V;
    private ActivationFunction activationFunction;

    /**
     * The allocateWeights method allocates layers' weights of Matrix W and V.
     *
     * @param H Integer value for weights.
     * @param random Random function to set weights.
     */
    private void allocateWeights(int H, Random random) {
        W = allocateLayerWeights(H, d + 1, random);
        V = allocateLayerWeights(K, H + 1, random);
    }

    /**
     * Training algorithm for the multilayer perceptron algorithm. 20 percent of the data is separated as cross-validation
     * data used for selecting the best weights. 80 percent of the data is used for training the multilayer perceptron with
     * gradient descent.
     *
     * @param train   Training data given to the algorithm
     * @param params Parameters of the multilayer perceptron.
     */
    public void train(InstanceList train, Parameter params) throws DiscreteFeaturesNotAllowed {
        Vector rMinusY, hidden, hiddenBiased, tmph, tmpHidden, activationDerivative;
        int epoch;
        double learningRate;
        Matrix deltaW, deltaV, bestW, bestV;
        ClassificationPerformance currentClassificationPerformance, bestClassificationPerformance;
        initialize(train);
        MultiLayerPerceptronParameter parameters = (MultiLayerPerceptronParameter) params;
        Partition partition = new Partition(train, parameters.getCrossValidationRatio(), new Random(parameters.getSeed()), true);
        InstanceList trainSet = partition.get(1);
        InstanceList validationSet = partition.get(0);
        activationFunction = parameters.getActivationFunction();
        allocateWeights(parameters.getHiddenNodes(), new Random(parameters.getSeed()));
        bestW = W.clone();
        bestV = V.clone();
        bestClassificationPerformance = new ClassificationPerformance(0.0);
        epoch = parameters.getEpoch();
        learningRate = parameters.getLearningRate();
        for (int i = 0; i < epoch; i++) {
            trainSet.shuffle(parameters.getSeed());
            for (int j = 0; j < trainSet.size(); j++) {
                createInputVector(trainSet.get(j));
                try {
                    hidden = calculateHidden(x, W, activationFunction);
                    hiddenBiased = hidden.biased();
                    rMinusY = calculateRMinusY(trainSet.get(j), hiddenBiased, V);
                    deltaV = rMinusY.multiply(hiddenBiased);
                    tmph = V.multiplyWithVectorFromLeft(rMinusY);
                    tmph.remove(0);
                    activationDerivative = calculateActivationDerivative(hidden, activationFunction);
                    tmpHidden = tmph.elementProduct(activationDerivative);
                    deltaW = tmpHidden.multiply(x);
                    deltaV.multiplyWithConstant(learningRate);
                    V.add(deltaV);
                    deltaW.multiplyWithConstant(learningRate);
                    W.add(deltaW);
                } catch (MatrixColumnMismatch | MatrixRowMismatch | MatrixDimensionMismatch | VectorSizeMismatch ignored) {
                }
            }
            currentClassificationPerformance = testClassifier(validationSet);
            if (currentClassificationPerformance.getAccuracy() > bestClassificationPerformance.getAccuracy()) {
                bestClassificationPerformance = currentClassificationPerformance;
                bestW = W.clone();
                bestV = V.clone();
            }
            learningRate *= parameters.getEtaDecrease();
        }
        W = bestW;
        V = bestV;
    }

    /**
     * Loads the multi-layer perceptron model from an input file.
     * @param fileName File name of the multi-layer perceptron model.
     */
    @Override
    public void loadModel(String fileName) {
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            loadClassLabels(input);
            W = loadMatrix(input);
            V = loadMatrix(input);
            activationFunction = loadActivationFunction(input);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The calculateOutput method calculates the forward single hidden layer by using Matrices W and V.
     */
    protected void calculateOutput() {
        try {
            calculateForwardSingleHiddenLayer(W, V, activationFunction);
        } catch (MatrixColumnMismatch ignored) {
        }
    }

    /**
     * Saves the multilayer perceptron model to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            saveClassLabels(output);
            saveMatrix(output, W);
            saveMatrix(output, V);
            output.println(activationFunction.toString());
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
