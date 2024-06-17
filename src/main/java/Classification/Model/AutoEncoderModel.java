package Classification.Model;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Parameter.ActivationFunction;
import Classification.Parameter.MultiLayerPerceptronParameter;
import Classification.Parameter.Parameter;
import Classification.Performance.Performance;
import Math.*;

import java.io.Serializable;
import java.util.Random;

public class AutoEncoderModel extends NeuralNetworkModel implements Serializable {
    private Matrix V, W;

    /**
     * Training algorithm for auto encoders. An auto encoder is a neural network which attempts to replicate its input at its output.
     *
     * @param train   Training data given to the algorithm.
     * @param params Parameters of the auto encoder.
     * @throws DiscreteFeaturesNotAllowed Exception for discrete features.
     */
    public void train(InstanceList train, Parameter params) throws DiscreteFeaturesNotAllowed {
        Matrix deltaW, deltaV, bestW, bestV;
        Vector hidden, hiddenBiased, rMinusY, oneMinusHidden, tmph, tmpHidden;
        int epoch;
        double learningRate;
        Performance currentPerformance, bestPerformance;
        if (!discreteCheck(train.get(0))){
            throw new DiscreteFeaturesNotAllowed();
        }
        MultiLayerPerceptronParameter parameters = (MultiLayerPerceptronParameter) params;
        classLabels = train.getDistinctClassLabels();
        K = classLabels.size();
        d = train.get(0).continuousAttributeSize();
        Partition partition = new Partition(train, 0.2, new Random(params.getSeed()), true);
        InstanceList trainSet = partition.get(1);
        InstanceList validationSet = partition.get(0);
        K = trainSet.get(0).continuousAttributeSize();
        allocateWeights(parameters.getHiddenNodes(), new Random(parameters.getSeed()));
        bestW = W.clone();
        bestV = V.clone();
        bestPerformance = new Performance(Double.MAX_VALUE);
        epoch = parameters.getEpoch();
        learningRate = parameters.getLearningRate();
        for (int i = 0; i < epoch; i++) {
            trainSet.shuffle(parameters.getSeed());
            for (int j = 0; j < trainSet.size(); j++) {
                createInputVector(trainSet.get(j));
                r = trainSet.get(j).toVector();
                try {
                    hidden = calculateHidden(x, W, ActivationFunction.SIGMOID);
                    hiddenBiased = hidden.biased();
                    y = V.multiplyWithVectorFromRight(hiddenBiased);
                    rMinusY = r.difference(y);
                    deltaV = rMinusY.multiply(hiddenBiased);
                    oneMinusHidden = calculateOneMinusHidden(hidden);
                    tmph = V.multiplyWithVectorFromLeft(rMinusY);
                    tmph.remove(0);
                    tmpHidden = oneMinusHidden.elementProduct(hidden.elementProduct(tmph));
                    deltaW = tmpHidden.multiply(x);
                    deltaV.multiplyWithConstant(learningRate);
                    V.add(deltaV);
                    deltaW.multiplyWithConstant(learningRate);
                    W.add(deltaW);
                } catch (MatrixColumnMismatch | MatrixRowMismatch | MatrixDimensionMismatch | VectorSizeMismatch ignored) {
                }
            }
            currentPerformance = testAutoEncoder(validationSet);
            if (currentPerformance.getErrorRate() < bestPerformance.getErrorRate()) {
                bestPerformance = currentPerformance;
                bestW = W.clone();
                bestV = V.clone();
            }
            learningRate *= 0.95;
        }
        W = bestW;
        V = bestV;
    }

    @Override
    public void loadModel(String fileName) {

    }

    /**
     * The allocateWeights method takes an integer number and sets layer weights of W and V matrices according to given number.
     *
     * @param H Integer input.
     * @param random Random function to set seed.
     */
    private void allocateWeights(int H, Random random) {
        W = allocateLayerWeights(H, d + 1, random);
        V = allocateLayerWeights(K, H + 1, random);
    }

    /**
     * The testAutoEncoder method takes an {@link InstanceList} as an input and tries to predict a value and finds the difference with the
     * actual value for each item of that InstanceList. At the end, it returns an error rate by finding the mean of total errors.
     *
     * @param data {@link InstanceList} to use as validation set.
     * @return Error rate by finding the mean of total errors.
     */
    public Performance testAutoEncoder(InstanceList data) {
        double total = data.size();
        double error = 0.0;
        for (int i = 0; i < total; i++) {
            y = predictInput(data.get(i));
            r = data.get(i).toVector();
            try {
                error += r.difference(y).dotProduct();
            } catch (VectorSizeMismatch ignored) {
            }
        }
        return new Performance(error / total);
    }

    /**
     * The predictInput method takes an {@link Instance} as an input and calculates a forward single hidden layer and returns the predicted value.
     *
     * @param instance {@link Instance} to predict.
     * @return Predicted value.
     */
    private Vector predictInput(Instance instance) {
        createInputVector(instance);
        try {
            calculateForwardSingleHiddenLayer(W, V, ActivationFunction.SIGMOID);
            return y;
        } catch (MatrixColumnMismatch matrixColumnMismatch) {
            return null;
        }
    }

    /**
     * The calculateOutput method calculates a forward single hidden layer.
     */
    @Override
    protected void calculateOutput() {
        try {
            calculateForwardSingleHiddenLayer(W, V, ActivationFunction.SIGMOID);
        } catch (MatrixColumnMismatch ignored) {
        }
    }

    @Override
    public void saveTxt(String fileName) {

    }

    /**
     * A performance test for an auto encoder with the given test set..
     *
     * @param testSet Test data (list of instances) to be tested.
     * @return Error rate.
     */
    public Performance test(InstanceList testSet) {
        return testAutoEncoder(testSet);
    }

}