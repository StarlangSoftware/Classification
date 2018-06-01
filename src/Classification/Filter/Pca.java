package Classification.Filter;

import Classification.Attribute.AttributeType;
import Classification.Attribute.ContinuousAttribute;
import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.Instance.Instance;
import Math.*;

import java.util.ArrayList;

public class Pca extends TrainedFeatureFilter{
    private double covarianceExplained = 0.99;
    private ArrayList<Eigenvector> eigenvectors;
    private int numberOfDimensions = -1;

    public Pca(DataSet dataSet, double covarianceExplained) {
        super(dataSet);
        this.covarianceExplained = covarianceExplained;
        train();
    }

    public Pca(DataSet dataSet, int numberOfDimensions) {
        super(dataSet);
        this.numberOfDimensions = numberOfDimensions;
        train();
    }

    public Pca(DataSet dataSet){
        super(dataSet);
        train();
    }

    private void removeUnnecessaryEigenvectors(ArrayList<Eigenvector> eigenvectors){
        double sum = 0.0, currentSum = 0.0;
        for (Eigenvector eigenvector : eigenvectors){
            sum += eigenvector.eigenValue();
        }
        ArrayList<Eigenvector> toBeRemoved = new ArrayList<>();
        for (Eigenvector eigenvector : eigenvectors){
            if (currentSum / sum < covarianceExplained){
                currentSum += eigenvector.eigenValue();
            } else {
                toBeRemoved.add(eigenvector);
            }
        }
        eigenvectors.removeAll(toBeRemoved);
    }

    private void removeAllEigenvectorsExceptTheMostImportantK(ArrayList<Eigenvector> eigenvectors){
        ArrayList<Eigenvector> toBeRemoved = new ArrayList<>();
        int i = 0;
        for (Eigenvector eigenvector : eigenvectors){
            if (i >= numberOfDimensions){
                toBeRemoved.add(eigenvector);
            }
            i++;
        }
        eigenvectors.removeAll(toBeRemoved);
    }

    @Override
    public void train() {
        Vector averageVector = new Vector(dataSet.getInstanceList().continuousAttributeAverage());
        Matrix covariance = dataSet.getInstanceList().covariance(averageVector);
        try {
            eigenvectors = covariance.characteristics();
            if (numberOfDimensions != -1){
                removeAllEigenvectorsExceptTheMostImportantK(eigenvectors);
            } else {
                removeUnnecessaryEigenvectors(eigenvectors);
            }
        } catch (MatrixNotSymmetric matrixNotSymmetric) {
        }
    }

    @Override
    protected void convertInstance(Instance instance) {
        Vector attributes = new Vector(instance.continuousAttributes());
        instance.removeAllAttributes();
        for (Eigenvector eigenvector : eigenvectors){
            try {
                instance.addAttribute(new ContinuousAttribute(attributes.dotProduct(eigenvector)));
            } catch (VectorSizeMismatch vectorSizeMismatch) {
            }
        }
    }

    @Override
    protected void convertDataDefinition() {
        DataDefinition dataDefinition = dataSet.getDataDefinition();
        dataDefinition.removeAllAttributes();
        for (int i = 0; i < eigenvectors.size(); i++){
            dataDefinition.addAttribute(AttributeType.CONTINUOUS);
        }
    }
}
