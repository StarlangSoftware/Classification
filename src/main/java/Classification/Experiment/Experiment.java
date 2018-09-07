package Classification.Experiment;

import Classification.Classifier.Classifier;
import Classification.DataSet.DataSet;
import Classification.FeatureSelection.FeatureSubSet;
import Classification.Parameter.Parameter;

public class Experiment {
    private Classifier classifier;
    private Parameter parameter;
    private DataSet dataSet;

    public Experiment(Classifier classifier, Parameter parameter, DataSet dataSet) {
        this.classifier = classifier;
        this.parameter = parameter;
        this.dataSet = dataSet;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public Parameter getParameter() {
        return parameter;
    }

    public DataSet getDataSet() {
        return dataSet;
    }

    public Experiment featureSelectedExperiment(FeatureSubSet featureSubSet){
        return new Experiment(classifier, parameter, dataSet.getSubSetOfFeatures(featureSubSet));
    }

}
