package Classification.Model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Parameter.ActivationFunction;
import Math.*;

public abstract class NeuralNetworkModel extends ValidatedModel implements Serializable {
    protected ArrayList<String> classLabels;
    protected int K, d;
    protected Vector x, y, r;

    protected abstract void calculateOutput();

    /**
     * Constructor that sets the class labels, their sizes as K and the size of the continuous attributes as d.
     *
     * @param trainSet {@link InstanceList} to use as train set.
     */
    public NeuralNetworkModel(InstanceList trainSet) {
        classLabels = trainSet.getDistinctClassLabels();
        K = classLabels.size();
        d = trainSet.get(0).continuousAttributeSize();
    }

    /**
     * The allocateLayerWeights method returns a new {@link Matrix} with random weights.
     *
     * @param row    Number of rows.
     * @param column Number of columns.
     * @param random Random function to set weights.
     * @return Matrix with random weights.
     */
    protected Matrix allocateLayerWeights(int row, int column, Random random) {
        return new Matrix(row, column, -0.01, +0.01, random);
    }

    /**
     * The normalizeOutput method takes an input {@link Vector} o, gets the result for e^o of each element of o,
     * then sums them up. At the end, divides each e^o by the summation.
     *
     * @param o Vector to normalize.
     * @return Normalized vector.
     */
    protected Vector normalizeOutput(Vector o) {
        double sum = 0.0;
        double[] values = new double[o.size()];
        for (int i = 0; i < values.length; i++){
            sum += Math.exp(o.getValue(i));
        }
        for (int i = 0; i < values.length; i++){
            values[i] = Math.exp(o.getValue(i)) / sum;
        }
        return new Vector(values);
    }

    /**
     * The createInputVector method takes an {@link Instance} as an input. It converts given Instance to the {@link java.util.Vector}
     * and insert 1.0 to the first element.
     *
     * @param instance Instance to insert 1.0.
     */
    protected void createInputVector(Instance instance) {
        x = instance.toVector();
        x.insert(0, 1.0);
    }

    /**
     * The calculateHidden method takes a {@link Vector} input and {@link Matrix} weights, It multiplies the weights
     * Matrix with given input Vector than applies the sigmoid function and returns the result.
     *
     * @param input   Vector to multiply weights.
     * @param weights Matrix is multiplied with input Vector.
     * @param activationFunction Activation function.
     * @return Result of sigmoid function.
     * @throws MatrixColumnMismatch Returns: Number of columns of the matrix should be equal to the size of the vector.
     */
    protected Vector calculateHidden(Vector input, Matrix weights, ActivationFunction activationFunction) throws MatrixColumnMismatch {
        Vector z;
        z = weights.multiplyWithVectorFromRight(input);
        switch (activationFunction){
            case SIGMOID:
            default:
                z.sigmoid();
                break;
            case TANH:
                z.tanh();
                break;
            case RELU:
                z.relu();
                break;
        }
        return z;
    }

    /**
     * The calculateOneMinusHidden method takes a {@link Vector} as input. It creates a Vector of ones and
     * returns the difference between given Vector.
     *
     * @param hidden Vector to find difference.
     * @return Returns the difference between one's Vector and input Vector.
     * @throws VectorSizeMismatch Return: Number of items in both vectors must be the same.
     */
    protected Vector calculateOneMinusHidden(Vector hidden) throws VectorSizeMismatch {
        Vector one;
        one = new Vector(hidden.size(), 1.0);
        return one.difference(hidden);
    }

    /**
     * The calculateForwardSingleHiddenLayer method takes two matrices W and V. First it multiplies W with x, then
     * multiplies V with the result of the previous multiplication.
     *
     * @param W Matrix to multiply with x.
     * @param V Matrix to multiply.
     * @param activationFunction Activation function.
     * @throws MatrixColumnMismatch Returns: Number of columns of the matrix should be equal to the size of the vector.
     */
    protected void calculateForwardSingleHiddenLayer(Matrix W, Matrix V, ActivationFunction activationFunction) throws MatrixColumnMismatch {
        Vector hidden, hiddenBiased;
        hidden = calculateHidden(x, W, activationFunction);
        hiddenBiased = hidden.biased();
        y = V.multiplyWithVectorFromRight(hiddenBiased);
    }

    /**
     * The calculateRMinusY method creates a new {@link Vector} with given Instance, then it multiplies given
     * input Vector with given weights Matrix. After normalizing the output, it returns the difference between the newly created
     * Vector and normalized output.
     *
     * @param instance Instance is used to get class labels.
     * @param input    Vector to multiply weights.
     * @param weights  Matrix of weights/
     * @return Difference between newly created Vector and normalized output.
     * @throws MatrixColumnMismatch Returns: Number of columns of the matrix should be equal to the size of the vector.
     * @throws VectorSizeMismatch   Return: Number of items in both vectors must be the same.
     */
    protected Vector calculateRMinusY(Instance instance, Vector input, Matrix weights) throws MatrixColumnMismatch, VectorSizeMismatch {
        Vector o;
        r = new Vector(K, classLabels.indexOf(instance.getClassLabel()), 1.0);
        o = weights.multiplyWithVectorFromRight(input);
        y = normalizeOutput(o);
        return r.difference(y);
    }

    /**
     * The predictWithCompositeInstance method takes an ArrayList possibleClassLabels. It returns the class label
     * which has the maximum value of y.
     *
     * @param possibleClassLabels ArrayList that has the class labels.
     * @return The class label which has the maximum value of y.
     */
    protected String predictWithCompositeInstance(ArrayList<String> possibleClassLabels) {
        String predictedClass = possibleClassLabels.get(0);
        double maxY = -Double.MAX_VALUE;
        for (int i = 0; i < classLabels.size(); i++) {
            if (possibleClassLabels.contains(classLabels.get(i)) && y.getValue(i) > maxY) {
                maxY = y.getValue(i);
                predictedClass = classLabels.get(i);
            }
        }
        return predictedClass;
    }

    /**
     * The predict method takes an {@link Instance} as an input, converts it to a Vector and calculates the {@link Matrix} y by
     * multiplying Matrix W with {@link Vector} x. Then it returns the class label which has the maximum y value.
     *
     * @param instance Instance to predict.
     * @return The class label which has the maximum y.
     */
    public String predict(Instance instance) {
        createInputVector(instance);
        calculateOutput();
        if (instance instanceof CompositeInstance) {
            return predictWithCompositeInstance(((CompositeInstance) instance).getPossibleClassLabels());
        } else {
            return classLabels.get(y.maxIndex());
        }
    }

    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        createInputVector(instance);
        calculateOutput();
        HashMap<String, Double> result = new HashMap<>();
        for (int i = 0; i < classLabels.size(); i++){
            result.put(classLabels.get(i), y.getValue(i));
        }
        return result;
    }

}
