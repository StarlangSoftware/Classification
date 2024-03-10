package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.InstanceList.InstanceListOfSameClass;
import Classification.Model.LdaModel;
import Classification.InstanceList.Partition;
import Math.*;
import Classification.Parameter.Parameter;

import java.util.HashMap;

public class Lda extends Classifier {

    /**
     * Training algorithm for the linear discriminant analysis classifier (Introduction to Machine Learning, Alpaydin, 2015).
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))) {
            throw new DiscreteFeaturesNotAllowed();
        }
        String Ci;
        double w0i;
        Matrix covariance, classCovariance;
        Vector averageVector, wi;
        HashMap<String, Double> w0 = new HashMap<>();
        HashMap<String, Vector> w = new HashMap<>();
        DiscreteDistribution priorDistribution = trainSet.classDistribution();
        Partition classLists = new Partition(trainSet);
        covariance = new Matrix(trainSet.get(0).continuousAttributeSize(), trainSet.get(0).continuousAttributeSize());
        for (int i = 0; i < classLists.size(); i++) {
            averageVector = new Vector(classLists.get(i).continuousAttributeAverage());
            classCovariance = classLists.get(i).covariance(averageVector);
            classCovariance.multiplyWithConstant(classLists.get(i).size() - 1);
            try {
                covariance.add(classCovariance);
            } catch (MatrixDimensionMismatch ignored) {
            }
        }
        covariance.divideByConstant(trainSet.size() - classLists.size());
        try {
            covariance.inverse();
        } catch (DeterminantZero | MatrixNotSquare ignored) {
        }
        for (int i = 0; i < classLists.size(); i++) {
            Ci = ((InstanceListOfSameClass) classLists.get(i)).getClassLabel();
            averageVector = new Vector(classLists.get(i).continuousAttributeAverage());
            try {
                wi = covariance.multiplyWithVectorFromRight(averageVector);
                w.put(Ci, wi);
                w0i = -0.5 * wi.dotProduct(averageVector) + Math.log(priorDistribution.getProbability(Ci));
                w0.put(Ci, w0i);
            } catch (MatrixColumnMismatch | VectorSizeMismatch ignored) {
            }
        }
        model = new LdaModel(priorDistribution, w, w0);
    }

    @Override
    public void loadModel(String fileName) {
        model = new LdaModel(fileName);
    }
}
