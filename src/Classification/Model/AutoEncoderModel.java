package Classification.Model;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Parameter.MultiLayerPerceptronParameter;
import Classification.Performance.Performance;
import Math.*;

import java.io.Serializable;

public class AutoEncoderModel extends NeuralNetworkModel implements Serializable {
    private Matrix V, W;

    private void allocateWeights(int H){
        W = allocateLayerWeights(H, d + 1);
        V = allocateLayerWeights(K, H + 1);
    }

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
            if (currentPerformance.getErrorRate() < bestPerformance.getErrorRate()){
                bestPerformance = currentPerformance;
                bestW = W.clone();
                bestV = V.clone();
            }
            learningRate *= 0.95;
        }
        W = bestW;
        V = bestV;
    }

    public Performance testAutoEncoder(InstanceList data){
        double total = data.size();
        double error = 0.0;
        for (int i = 0; i < total; i++){
            y = predictInput(data.get(i));
            r = data.get(i).toVector();
            try {
                error += r.difference(y).dotProduct();
            } catch (VectorSizeMismatch vectorSizeMismatch) {
            }
        }
        return new Performance(error / total);
    }

    private Vector predictInput(Instance instance) {
        createInputVector(instance);
        try {
            calculateForwardSingleHiddenLayer(W, V);
            return y;
        } catch (MatrixColumnMismatch matrixColumnMismatch) {
            return null;
        }
    }

    @Override
    protected void calculateOutput() {
        try {
            calculateForwardSingleHiddenLayer(W, V);
        } catch (MatrixColumnMismatch matrixColumnMismatch) {
        }
    }
}