package Classification.Model;

import java.io.Serializable;
import java.util.ArrayList;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Math.*;

public abstract class NeuralNetworkModel extends ValidatedModel implements Serializable{
    protected ArrayList<String> classLabels;
    protected int K, d;
    protected Vector x, y, r;
    protected abstract void calculateOutput();

    public NeuralNetworkModel(InstanceList trainSet){
        classLabels = trainSet.getDistinctClassLabels();
        K = classLabels.size();
        d = trainSet.get(0).continuousAttributeSize();
    }

    protected Matrix allocateLayerWeights(int row, int column){
        return new Matrix(row, column, -0.01, +0.01);
    }

    protected Vector normalizeOutput(Vector o){
        double sum = 0.0;
        double[] values = new double[o.size()];
        for (int i = 0; i < values.length; i++)
            sum += Math.exp(o.getValue(i));
        for (int i = 0; i < values.length; i++)
            values[i] = Math.exp(o.getValue(i)) / sum;
        return new Vector(values);
    }

    protected void createInputVector(Instance instance){
        x = instance.toVector();
        x.insert(0, 1.0);
    }

    protected Vector calculateHidden(Vector input, Matrix weights) throws MatrixColumnMismatch {
        Vector z;
        z = weights.multiplyWithVectorFromRight(input);
        z.sigmoid();
        return z;
    }

    protected Vector calculateOneMinusHidden(Vector hidden) throws VectorSizeMismatch {
        Vector one;
        one = new Vector(hidden.size(), 1.0);
        return one.difference(hidden);
    }

    protected void calculateForwardSingleHiddenLayer(Matrix W, Matrix V) throws MatrixColumnMismatch {
        Vector hidden, hiddenBiased;
        hidden = calculateHidden(x, W);
        hiddenBiased = hidden.biased();
        y = V.multiplyWithVectorFromRight(hiddenBiased);
    }

    protected Vector calculateRMinusY(Instance instance, Vector input, Matrix weights) throws MatrixColumnMismatch, VectorSizeMismatch {
        Vector o;
        r = new Vector(K, classLabels.indexOf(instance.getClassLabel()), 1.0);
        o = weights.multiplyWithVectorFromRight(input);
        y = normalizeOutput(o);
        return r.difference(y);
    }

    protected String predictWithCompositeInstance(ArrayList<String> possibleClassLabels){
        String predictedClass = possibleClassLabels.get(0);
        double maxY = -Double.MAX_VALUE;
        for (int i = 0; i < classLabels.size(); i++){
            if (possibleClassLabels.contains(classLabels.get(i)) && y.getValue(i) > maxY){
                maxY = y.getValue(i);
                predictedClass = classLabels.get(i);
            }
        }
        return predictedClass;
    }

    public String predict(Instance instance) {
        createInputVector(instance);
        calculateOutput();
        if (instance instanceof CompositeInstance){
            return predictWithCompositeInstance(((CompositeInstance)instance).getPossibleClassLabels());
        } else {
            return classLabels.get(y.maxIndex());
        }
    }

}
