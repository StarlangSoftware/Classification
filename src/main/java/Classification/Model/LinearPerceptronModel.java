package Classification.Model;

import Classification.InstanceList.InstanceList;
import Classification.Parameter.LinearPerceptronParameter;
import Classification.Performance.ClassificationPerformance;
import Math.*;

import java.io.Serializable;

public class LinearPerceptronModel extends NeuralNetworkModel implements Serializable {

    protected Matrix W;

    /**
     * Constructor that sets the {@link NeuralNetworkModel} nodes with given {@link InstanceList}.
     *
     * @param trainSet InstanceList that is used to train.
     */
    public LinearPerceptronModel(InstanceList trainSet) {
        super(trainSet);
    }

    /**
     * Constructor that takes {@link InstanceList}s as trainsSet and validationSet. Initially it allocates layer weights,
     * then creates an input vector by using given trainSet and finds error. Via the validationSet it finds the classification
     * performance and at the end it reassigns the allocated weight Matrix with the matrix that has the best accuracy.
     *
     * @param trainSet      InstanceList that is used to train.
     * @param validationSet InstanceList that is used to validate.
     * @param parameters    Linear perceptron parameters; learningRate, etaDecrease, crossValidationRatio, epoch.
     */
    public LinearPerceptronModel(InstanceList trainSet, InstanceList validationSet, LinearPerceptronParameter parameters) {
        this(trainSet);
        Vector rMinusY;
        int epoch;
        double learningRate;
        Matrix deltaW, bestW;
        ClassificationPerformance currentClassificationPerformance, bestClassificationPerformance;
        W = allocateLayerWeights(K, d + 1);
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
                } catch (MatrixColumnMismatch | MatrixDimensionMismatch | VectorSizeMismatch mismatch) {
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
     * The calculateOutput method calculates the {@link Matrix} y by multiplying Matrix W with {@link Vector} x.
     */
    protected void calculateOutput() {
        try {
            y = W.multiplyWithVectorFromRight(x);
        } catch (MatrixColumnMismatch matrixColumnMismatch) {
        }
    }

}
