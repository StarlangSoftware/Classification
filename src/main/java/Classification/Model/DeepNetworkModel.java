package Classification.Model;

import Classification.Performance.ClassificationPerformance;
import Classification.InstanceList.InstanceList;
import Classification.Parameter.DeepNetworkParameter;
import Math.*;

import java.io.Serializable;
import java.util.ArrayList;

public class DeepNetworkModel extends NeuralNetworkModel implements Serializable {
    private ArrayList<Matrix> weights;
    private int hiddenLayerSize;

    private void allocateWeights(DeepNetworkParameter parameters){
        weights = new ArrayList<>();
        weights.add(allocateLayerWeights(parameters.getHiddenNodes(0), d + 1));
        for (int i = 0; i < parameters.layerSize() - 1; i++){
            weights.add(allocateLayerWeights(parameters.getHiddenNodes(i + 1), parameters.getHiddenNodes(i) + 1));
        }
        weights.add(allocateLayerWeights(K, parameters.getHiddenNodes(parameters.layerSize() - 1) + 1));
        hiddenLayerSize = parameters.layerSize();
    }

    private ArrayList<Matrix> setBestWeights(){
        ArrayList<Matrix> bestWeights = new ArrayList<>();
        for (Matrix m : weights){
            bestWeights.add(m.clone());
        }
        return bestWeights;
    }

    public DeepNetworkModel(InstanceList trainSet, InstanceList validationSet, DeepNetworkParameter parameters) {
        super(trainSet);
        int epoch;
        double learningRate;
        Vector rMinusY, oneMinusHidden, tmpHidden, tmph;
        ClassificationPerformance currentClassificationPerformance, bestClassificationPerformance;
        ArrayList<Matrix> bestWeights;
        ArrayList<Matrix> deltaWeights = new ArrayList<>();
        ArrayList<Vector> hidden = new ArrayList<>();
        ArrayList<Vector> hiddenBiased = new ArrayList<>();
        allocateWeights(parameters);
        bestWeights = setBestWeights();
        bestClassificationPerformance = new ClassificationPerformance(0.0);
        epoch = parameters.getEpoch();
        learningRate = parameters.getLearningRate();
        for (int i = 0; i < epoch; i++) {
            trainSet.shuffle(parameters.getSeed());
            for (int j = 0; j < trainSet.size(); j++) {
                createInputVector(trainSet.get(j));
                try{
                    hidden.clear();
                    hiddenBiased.clear();
                    deltaWeights.clear();
                    for (int k = 0; k < hiddenLayerSize; k++){
                        if (k == 0){
                            hidden.add(calculateHidden(x, weights.get(k)));
                        } else {
                            hidden.add(calculateHidden(hiddenBiased.get(k - 1), weights.get(k)));
                        }
                        hiddenBiased.add(hidden.get(k).biased());
                    }
                    rMinusY = calculateRMinusY(trainSet.get(j), hiddenBiased.get(hiddenLayerSize - 1), weights.get(weights.size() - 1));
                    deltaWeights.add(0, rMinusY.multiply(hiddenBiased.get(hiddenLayerSize - 1)));
                    for (int k = weights.size() - 2; k >= 0; k--){
                        oneMinusHidden = calculateOneMinusHidden(hidden.get(k));
                        tmph = deltaWeights.get(0).elementProduct(weights.get(k + 1)).sumOfRows();
                        tmph.remove(0);
                        tmpHidden = oneMinusHidden.elementProduct(tmph);
                        if (k == 0){
                            deltaWeights.add(0, tmpHidden.multiply(x));
                        } else {
                            deltaWeights.add(0, tmpHidden.multiply(hiddenBiased.get(k - 1)));
                        }
                    }
                    for (int k = 0; k < weights.size(); k++){
                        deltaWeights.get(k).multiplyWithConstant(learningRate);
                        weights.get(k).add(deltaWeights.get(k));
                    }
                }catch(MatrixColumnMismatch | VectorSizeMismatch | MatrixDimensionMismatch mismatch){
                    System.out.println("Error");
                }
            }
            currentClassificationPerformance = testClassifier(validationSet);
            if (currentClassificationPerformance.getAccuracy() > bestClassificationPerformance.getAccuracy()){
                bestClassificationPerformance = currentClassificationPerformance;
                bestWeights = setBestWeights();
            }
            learningRate *= parameters.getEtaDecrease();
        }
        weights.clear();
        for (Matrix m : bestWeights){
            weights.add(m);
        }
    }

    protected void calculateOutput() {
        Vector hidden, hiddenBiased = null;
        try {
            for (int i = 0; i < weights.size() - 1; i++){
                if (i == 0){
                    hidden = calculateHidden(x, weights.get(i));
                } else {
                    hidden = calculateHidden(hiddenBiased, weights.get(i));
                }
                hiddenBiased = hidden.biased();
            }
            y = weights.get(weights.size() - 1).multiplyWithVectorFromRight(hiddenBiased);
        } catch (MatrixColumnMismatch matrixColumnMismatch) {
        }
    }

}
