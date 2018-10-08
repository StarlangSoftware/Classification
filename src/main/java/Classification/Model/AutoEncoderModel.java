package Classification.Model;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Parameter.MultiLayerPerceptronParameter;
import Classification.Performance.Performance;
import Math.*;

import java.io.Serializable;

public class AutoEncoderModel extends NeuralNetworkModel implements Serializable {
    private Matrix V, W;

    /**
     * The allocateWeights method takes an integer number and sets layer weights of W and V matrices according to given number.
     *
     * @param H Integer input.
     */
    private void allocateWeights(int H) {
        W = allocateLayerWeights(H, d + 1);
        V = allocateLayerWeights(K, H + 1);
    }

    /**
     * The {@link AutoEncoderModel} method takes two {@link InstanceList}s as inputs; train set and validation set. First it allocates
     * the weights of W and V matrices using given {@link MultiLayerPerceptronParameter} and takes the clones of these matrices as the bestW and bestV.
     * Then, it gets the epoch and starts to iterate over them. First it shuffles the train set and tries to find the new W and V matrices.
     * At the end it tests the autoencoder with given validation set and if its performance is better than the previous one,
     * it reassigns the bestW and bestV matrices. Continue to iterate with a lower learning rate till the end of an episode.
     *
     * @param trainSet      {@link InstanceList} to use as train set.
     * @param validationSet {@link InstanceList} to use as validation set.
     * @param parameters    {@link MultiLayerPerceptronParameter} is used to get the parameters.
     */
    public AutoEncoderModel(InstanceList trainSet, InstanceList validationSet, MultiLayerPerceptronParameter parameters) {
        super(trainSet);
        Matrix deltaW, deltaV, bestW, bestV;
        Vector hidden, hiddenBiased, rMinusY, oneMinusHidden, tmph, tmpHidden;
        int epoch;
        double learningRate;
        Performance currentPerformance, bestPerformance;
        K = trainSet.get(0).continuousAttributeSize();
        allocateWeights(parameters.getHiddenNodes());
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
                    hidden = calculateHidden(x, W);
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
                } catch (MatrixColumnMismatch | MatrixRowMismatch | MatrixDimensionMismatch | VectorSizeMismatch mismatch) {
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

    /**
     * Th testAutoEncoder method takes an {@link InstanceList} as an input and tries to predict a value and finds the difference with the
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
            } catch (VectorSizeMismatch vectorSizeMismatch) {
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
            calculateForwardSingleHiddenLayer(W, V);
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
            calculateForwardSingleHiddenLayer(W, V);
        } catch (MatrixColumnMismatch matrixColumnMismatch) {
        }
    }
}