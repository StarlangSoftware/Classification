package Classification.Model.NeuralNetwork;

import Classification.InstanceList.Partition;
import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Parameter.ActivationFunction;
import Classification.Parameter.Parameter;
import Classification.Performance.ClassificationPerformance;
import Classification.InstanceList.InstanceList;
import Classification.Parameter.DeepNetworkParameter;
import Math.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Random;

public class DeepNetworkModel extends NeuralNetworkModel implements Serializable {
    private ArrayList<Matrix> weights;
    private int hiddenLayerSize;
    private ActivationFunction activationFunction;


    /**
     * The allocateWeights method takes {@link DeepNetworkParameter}s as an input. First it adds random weights to the {@link ArrayList}
     * of {@link Matrix} weights' first layer. Then it loops through the layers and adds random weights till the last layer.
     * At the end it adds random weights to the last layer and also sets the hiddenLayerSize value.
     *
     * @param parameters {@link DeepNetworkParameter} input.
     */
    private void allocateWeights(DeepNetworkParameter parameters) {
        weights = new ArrayList<>();
        weights.add(allocateLayerWeights(parameters.getHiddenNodes(0), d + 1, new Random(parameters.getSeed())));
        for (int i = 0; i < parameters.layerSize() - 1; i++) {
            weights.add(allocateLayerWeights(parameters.getHiddenNodes(i + 1), parameters.getHiddenNodes(i) + 1, new Random(parameters.getSeed())));
        }
        weights.add(allocateLayerWeights(K, parameters.getHiddenNodes(parameters.layerSize() - 1) + 1, new Random(parameters.getSeed())));
        hiddenLayerSize = parameters.layerSize();
    }

    /**
     * The setBestWeights method creates an {@link ArrayList} of Matrix as bestWeights and clones the values of weights {@link ArrayList}
     * into this newly created {@link ArrayList}.
     *
     * @return An {@link ArrayList} clones from the weights ArrayList.
     */
    private ArrayList<Matrix> setBestWeights() {
        ArrayList<Matrix> bestWeights = new ArrayList<>();
        for (Matrix m : weights) {
            bestWeights.add(m.clone());
        }
        return bestWeights;
    }

    /**
     * Training algorithm for deep network classifier.
     *
     * @param train   Training data given to the algorithm.
     * @param params Parameters of the deep network algorithm. crossValidationRatio and seed are used as parameters.
     * @throws DiscreteFeaturesNotAllowed Exception for discrete features.
     */
    public void train(InstanceList train, Parameter params) throws DiscreteFeaturesNotAllowed {
        int epoch;
        double learningRate;
        Vector rMinusY, tmpHidden = new Vector(0, 0), tmph, activationDerivative;
        ClassificationPerformance currentClassificationPerformance, bestClassificationPerformance;
        initialize(train);
        DeepNetworkParameter parameters = ((DeepNetworkParameter) params);
        Partition partition = new Partition(train, parameters.getCrossValidationRatio(), new Random(parameters.getSeed()), true);
        InstanceList trainSet = partition.get(1);
        InstanceList validationSet = partition.get(0);
        ArrayList<Matrix> bestWeights;
        ArrayList<Matrix> deltaWeights = new ArrayList<>();
        ArrayList<Vector> hidden = new ArrayList<>();
        ArrayList<Vector> hiddenBiased = new ArrayList<>();
        activationFunction = parameters.getActivationFunction();
        allocateWeights(parameters);
        bestWeights = setBestWeights();
        bestClassificationPerformance = new ClassificationPerformance(0.0);
        epoch = parameters.getEpoch();
        learningRate = parameters.getLearningRate();
        for (int i = 0; i < epoch; i++) {
            trainSet.shuffle(parameters.getSeed());
            for (int j = 0; j < trainSet.size(); j++) {
                createInputVector(trainSet.get(j));
                try {
                    hidden.clear();
                    hiddenBiased.clear();
                    deltaWeights.clear();
                    for (int k = 0; k < hiddenLayerSize; k++) {
                        if (k == 0) {
                            hidden.add(calculateHidden(x, weights.get(k), activationFunction));
                        } else {
                            hidden.add(calculateHidden(hiddenBiased.get(k - 1), weights.get(k), activationFunction));
                        }
                        hiddenBiased.add(hidden.get(k).biased());
                    }
                    rMinusY = calculateRMinusY(trainSet.get(j), hiddenBiased.get(hiddenLayerSize - 1), weights.get(weights.size() - 1));
                    deltaWeights.add(0, rMinusY.multiply(hiddenBiased.get(hiddenLayerSize - 1)));
                    for (int k = weights.size() - 2; k >= 0; k--) {
                        if (k == weights.size() - 2){
                            tmph = weights.get(k + 1).multiplyWithVectorFromLeft(rMinusY);
                        } else {
                            tmph = weights.get(k + 1).multiplyWithVectorFromLeft(tmpHidden);
                        }
                        tmph.remove(0);
                        activationDerivative = calculateActivationDerivative(hidden.get(k), activationFunction);
                        tmpHidden = tmph.elementProduct(activationDerivative);
                        if (k == 0) {
                            deltaWeights.add(0, tmpHidden.multiply(x));
                        } else {
                            deltaWeights.add(0, tmpHidden.multiply(hiddenBiased.get(k - 1)));
                        }
                    }
                    for (int k = 0; k < weights.size(); k++) {
                        deltaWeights.get(k).multiplyWithConstant(learningRate);
                        weights.get(k).add(deltaWeights.get(k));
                    }
                } catch (MatrixColumnMismatch | VectorSizeMismatch | MatrixDimensionMismatch | MatrixRowMismatch mismatch) {
                    System.out.println("Error");
                }
            }
            currentClassificationPerformance = testClassifier(validationSet);
            if (currentClassificationPerformance.getAccuracy() > bestClassificationPerformance.getAccuracy()) {
                bestClassificationPerformance = currentClassificationPerformance;
                bestWeights = setBestWeights();
            }
            learningRate *= parameters.getEtaDecrease();
        }
        weights.clear();
        weights.addAll(bestWeights);
    }

    /**
     * Loads the deep network model from an input file.
     * @param fileName File name of the deep network model.
     */
    @Override
    public void loadModel(String fileName) {
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            loadClassLabels(input);
            hiddenLayerSize = Integer.parseInt(input.readLine());
            weights = new ArrayList<>();
            for (int i = 0; i < hiddenLayerSize + 1; i++){
                weights.add(loadMatrix(input));
            }
            activationFunction = loadActivationFunction(input);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The calculateOutput method loops size of the weights times and calculate one hidden layer at a time and adds bias term.
     * At the end it updates the output y value.
     */
    protected void calculateOutput() {
        Vector hidden, hiddenBiased = null;
        try {
            for (int i = 0; i < weights.size() - 1; i++) {
                if (i == 0) {
                    hidden = calculateHidden(x, weights.get(i), activationFunction);
                } else {
                    hidden = calculateHidden(hiddenBiased, weights.get(i), activationFunction);
                }
                hiddenBiased = hidden.biased();
            }
            y = weights.get(weights.size() - 1).multiplyWithVectorFromRight(hiddenBiased);
        } catch (MatrixColumnMismatch ignored) {
        }
    }

    /**
     * Saves the deep network model to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            saveClassLabels(output);
            output.println(hiddenLayerSize);
            for (Matrix matrix : weights){
                saveMatrix(output, matrix);
            }
            output.println(activationFunction.toString());
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
