package Classification.Filter;

import Classification.Attribute.AttributeType;
import Classification.Attribute.ContinuousAttribute;
import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.Instance.Instance;
import Math.*;

import java.util.ArrayList;

public class Pca extends TrainedFeatureFilter {
    private double covarianceExplained = 0.99;
    private ArrayList<Eigenvector> eigenvectors;
    private int numberOfDimensions = -1;

    /**
     * Constructor that sets the dataSet and covariance explained. Then calls train method.
     *
     * @param dataSet             DataSet that will bu used.
     * @param covarianceExplained Number that shows the explained covariance.
     */
    public Pca(DataSet dataSet, double covarianceExplained) {
        super(dataSet);
        this.covarianceExplained = covarianceExplained;
        train();
    }

    /**
     * Constructor that sets the dataSet and dimension. Then calls train method.
     *
     * @param dataSet            DataSet that will bu used.
     * @param numberOfDimensions Dimension number.
     */
    public Pca(DataSet dataSet, int numberOfDimensions) {
        super(dataSet);
        this.numberOfDimensions = numberOfDimensions;
        train();
    }

    /**
     * Constructor that sets the dataSet and dimension. Then calls train method.
     *
     * @param dataSet DataSet that will bu used.
     */
    public Pca(DataSet dataSet) {
        super(dataSet);
        train();
    }

    /**
     * The removeUnnecessaryEigenvectors methods takes an ArrayList of Eigenvectors. It first calculates the summation
     * of eigenValues. Then it finds the eigenvectors which have lesser summation than covarianceExplained and removes these
     * eigenvectors.
     */
    private void removeUnnecessaryEigenvectors() {
        double sum = 0.0, currentSum = 0.0;
        for (Eigenvector eigenvector : eigenvectors) {
            sum += eigenvector.eigenValue();
        }
        ArrayList<Eigenvector> toBeRemoved = new ArrayList<>();
        for (Eigenvector eigenvector : eigenvectors) {
            if (currentSum / sum < covarianceExplained) {
                currentSum += eigenvector.eigenValue();
            } else {
                toBeRemoved.add(eigenvector);
            }
        }
        eigenvectors.removeAll(toBeRemoved);
    }

    /**
     * The removeAllEigenvectorsExceptTheMostImportantK method takes an {@link ArrayList} of {@link Eigenvector}s and removes the
     * surplus eigenvectors when the number of eigenvectors is greater than the dimension.
     */
    private void removeAllEigenvectorsExceptTheMostImportantK() {
        ArrayList<Eigenvector> toBeRemoved = new ArrayList<>();
        int i = 0;
        for (Eigenvector eigenvector : eigenvectors) {
            if (i >= numberOfDimensions) {
                toBeRemoved.add(eigenvector);
            }
            i++;
        }
        eigenvectors.removeAll(toBeRemoved);
    }

    /**
     * The train method creates an averageVector from continuousAttributeAverage and a covariance {@link Matrix} from that averageVector.
     * Then finds the eigenvectors of that covariance matrix and removes its unnecessary eigenvectors.
     */
    @Override
    public void train() {
        Vector averageVector = new Vector(dataSet.getInstanceList().continuousAttributeAverage());
        Matrix covariance = dataSet.getInstanceList().covariance(averageVector);
        try {
            eigenvectors = covariance.characteristics();
            if (numberOfDimensions != -1) {
                removeAllEigenvectorsExceptTheMostImportantK();
            } else {
                removeUnnecessaryEigenvectors();
            }
        } catch (MatrixNotSymmetric matrixNotSymmetric) {
        }
    }

    /**
     * The convertInstance method takes an {@link Instance} as an input and creates a {@link java.util.Vector} attributes from continuousAttributes.
     * After removing all attributes of given instance, it then adds new {@link ContinuousAttribute} by using the dot
     * product of attributes Vector and the eigenvectors.
     *
     * @param instance Instance that will be converted to {@link ContinuousAttribute} by using eigenvectors.
     */
    @Override
    protected void convertInstance(Instance instance) {
        Vector attributes = new Vector(instance.continuousAttributes());
        instance.removeAllAttributes();
        for (Eigenvector eigenvector : eigenvectors) {
            try {
                instance.addAttribute(new ContinuousAttribute(attributes.dotProduct(eigenvector)));
            } catch (VectorSizeMismatch vectorSizeMismatch) {
            }
        }
    }

    /**
     * The convertDataDefinition method gets the data definitions of the dataSet and removes all the attributes. Then adds
     * new attributes as CONTINUOUS.
     */
    @Override
    protected void convertDataDefinition() {
        DataDefinition dataDefinition = dataSet.getDataDefinition();
        dataDefinition.removeAllAttributes();
        for (int i = 0; i < eigenvectors.size(); i++) {
            dataDefinition.addAttribute(AttributeType.CONTINUOUS);
        }
    }
}
