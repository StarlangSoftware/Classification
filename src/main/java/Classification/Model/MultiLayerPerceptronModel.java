package Classification.Model;

import Classification.InstanceList.InstanceList;
import Classification.Parameter.ActivationFunction;
import Classification.Parameter.MultiLayerPerceptronParameter;
import Classification.Performance.ClassificationPerformance;
import Math.*;

import java.io.Serializable;
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
     * A constructor that takes {@link InstanceList}s as trainsSet and validationSet. It  sets the {@link NeuralNetworkModel}
     * nodes with given {@link InstanceList} then creates an input vector by using given trainSet and finds error.
     * Via the validationSet it finds the classification performance and reassigns the allocated weight Matrix with the matrix
     * that has the best accuracy and the Matrix V with the best Vector input.
     *
     * @param trainSet      InstanceList that is used to train.
     * @param validationSet InstanceList that is used to validate.
     * @param parameters    Multi layer perceptron parameters; seed, learningRate, etaDecrease, crossValidationRatio, epoch, hiddenNodes.
     */
    public MultiLayerPerceptronModel(InstanceList trainSet, InstanceList validationSet, MultiLayerPerceptronParameter parameters) {
        super(trainSet);
        Vector rMinusY, hidden, hiddenBiased, oneMinusHidden, tmph, tmpHidden, activationDerivative;
        int epoch;
        double learningRate;
        Matrix deltaW, deltaV, bestW, bestV;
        ClassificationPerformance currentClassificationPerformance, bestClassificationPerformance;
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
                    switch (activationFunction){
                        case SIGMOID:
                        default:
                            oneMinusHidden = calculateOneMinusHidden(hidden);
                            activationDerivative = oneMinusHidden.elementProduct(hidden);
                            break;
                        case TANH:
                            Vector one = new Vector(hidden.size(), 1.0);
                            hidden.tanh();
                            activationDerivative = one.difference(hidden.elementProduct(hidden));
                            break;
                        case RELU:
                            hidden.reluDerivative();
                            activationDerivative = hidden;
                            break;
                    }
                    tmpHidden = tmph.elementProduct(activationDerivative);
                    deltaW = tmpHidden.multiply(x);
                    deltaV.multiplyWithConstant(learningRate);
                    V.add(deltaV);
                    deltaW.multiplyWithConstant(learningRate);
                    W.add(deltaW);
                } catch (MatrixColumnMismatch | MatrixRowMismatch | MatrixDimensionMismatch | VectorSizeMismatch mismatch) {
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
     * The calculateOutput method calculates the forward single hidden layer by using Matrices W and V.
     */
    protected void calculateOutput() {
        try {
            calculateForwardSingleHiddenLayer(W, V, activationFunction);
        } catch (MatrixColumnMismatch matrixColumnMismatch) {
        }
    }

}
