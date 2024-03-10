package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.InstanceList.InstanceListOfSameClass;
import Classification.Model.QdaModel;
import Classification.Parameter.Parameter;
import Classification.InstanceList.Partition;
import Math.DiscreteDistribution;
import Math.*;

import java.util.HashMap;

public class Qda extends Classifier {

    /**
     * Training algorithm for the quadratic discriminant analysis classifier (Introduction to Machine Learning, Alpaydin, 2015).
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        if (!discreteCheck(trainSet.get(0))) {
            throw new DiscreteFeaturesNotAllowed();
        }
        String Ci;
        double determinant = 0, w0i;
        Matrix classCovariance, Wi;
        Vector averageVector, wi;
        HashMap<String, Double> w0 = new HashMap<>();
        HashMap<String, Vector> w = new HashMap<>();
        HashMap<String, Matrix> W = new HashMap<>();
        Partition classLists = new Partition(trainSet);
        DiscreteDistribution priorDistribution = trainSet.classDistribution();
        for (int i = 0; i < classLists.size(); i++) {
            Ci = ((InstanceListOfSameClass) classLists.get(i)).getClassLabel();
            averageVector = new Vector(classLists.get(i).continuousAttributeAverage());
            classCovariance = classLists.get(i).covariance(averageVector);
            try {
                determinant = classCovariance.determinant();
                classCovariance.inverse();
            } catch (DeterminantZero | MatrixNotSquare ignored) {
            }
            Wi = classCovariance.clone();
            Wi.multiplyWithConstant(-0.5);
            W.put(Ci, Wi);
            try {
                wi = classCovariance.multiplyWithVectorFromLeft(averageVector);
                w.put(Ci, wi);
                w0i = -0.5 * (wi.dotProduct(averageVector) + Math.log(determinant)) + Math.log(priorDistribution.getProbability(Ci));
                w0.put(Ci, w0i);
            } catch (MatrixRowMismatch | VectorSizeMismatch ignored) {
            }
        }
        model = new QdaModel(priorDistribution, W, w, w0);
    }

    @Override
    public void loadModel(String fileName) {
        model = new QdaModel(fileName);
    }
}
