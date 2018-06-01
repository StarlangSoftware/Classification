package Classification.Model;

import Classification.InstanceList.InstanceList;
import Classification.Parameter.MultiLayerPerceptronParameter;
import Classification.Performance.ClassificationPerformance;
import Math.*;

import java.io.Serializable;

public class MultiLayerPerceptronModel extends LinearPerceptronModel implements Serializable{
    private Matrix V;

    private void allocateWeights(int H){
        W = allocateLayerWeights(H, d + 1);
        V = allocateLayerWeights(K, H + 1);
    }

    public MultiLayerPerceptronModel(InstanceList trainSet, InstanceList validationSet, MultiLayerPerceptronParameter parameters){
        super(trainSet);
        Vector rMinusY, hidden, hiddenBiased, oneMinusHidden, tmph, tmpHidden;
        int epoch;
        double learningRate;
        Matrix deltaW, deltaV, bestW, bestV;
        ClassificationPerformance currentClassificationPerformance, bestClassificationPerformance;
        allocateWeights(parameters.getHiddenNodes());
        bestW = W.clone();
        bestV = V.clone();
        bestClassificationPerformance = new ClassificationPerformance(0.0);
        epoch = parameters.getEpoch();
        learningRate = parameters.getLearningRate();
        for (int i = 0; i < epoch; i++){
            trainSet.shuffle(parameters.getSeed());
            for (int j = 0; j < trainSet.size(); j++){
                createInputVector(trainSet.get(j));
                try {
                    hidden = calculateHidden(x, W);
                    hiddenBiased = hidden.biased();
                    rMinusY = calculateRMinusY(trainSet.get(j), hiddenBiased, V);
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
            currentClassificationPerformance = testClassifier(validationSet);
            if (currentClassificationPerformance.getAccuracy() > bestClassificationPerformance.getAccuracy()){
                bestClassificationPerformance = currentClassificationPerformance;
                bestW = W.clone();
                bestV = V.clone();
            }
            learningRate *= parameters.getEtaDecrease();
        }
        W = bestW;
        V = bestV;
    }

    protected void calculateOutput() {
        try {
            calculateForwardSingleHiddenLayer(W, V);
        } catch (MatrixColumnMismatch matrixColumnMismatch) {
        }
    }

}
