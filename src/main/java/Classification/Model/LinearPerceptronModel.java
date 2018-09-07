package Classification.Model;

import Classification.InstanceList.InstanceList;
import Classification.Parameter.LinearPerceptronParameter;
import Classification.Performance.ClassificationPerformance;
import Math.*;

import java.io.Serializable;

public class LinearPerceptronModel extends NeuralNetworkModel implements Serializable{

    protected Matrix W;

    public LinearPerceptronModel(InstanceList trainSet){
        super(trainSet);
    }

    public LinearPerceptronModel(InstanceList trainSet, InstanceList validationSet, LinearPerceptronParameter parameters){
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
        for (int i = 0; i < epoch; i++){
            trainSet.shuffle(parameters.getSeed());
            for (int j = 0; j < trainSet.size(); j++){
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
            if (currentClassificationPerformance.getAccuracy() > bestClassificationPerformance.getAccuracy()){
                bestClassificationPerformance = currentClassificationPerformance;
                bestW = W.clone();
            }
            learningRate *= parameters.getEtaDecrease();
        }
        W = bestW;
    }

    protected void calculateOutput() {
        try {
            y = W.multiplyWithVectorFromRight(x);
        } catch (MatrixColumnMismatch matrixColumnMismatch) {
        }
    }

}
