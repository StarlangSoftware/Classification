package Classification.Experiment;

import Classification.Classifier.Classifier;
import Classification.DataSet.DataSet;
import Classification.FeatureSelection.FeatureSubSet;
import Classification.Parameter.Parameter;

public class Experiment {
    private final Classifier classifier;
    private final Parameter parameter;
    private final DataSet dataSet;

    /**
     * Constructor for a specific machine learning experiment
     * @param classifier Classifier used in the machine learning experiment
     * @param parameter Parameter(s) of the classifier.
     * @param dataSet DataSet on which the classifier is run.
     */
    public Experiment(Classifier classifier, Parameter parameter, DataSet dataSet) {
        this.classifier = classifier;
        this.parameter = parameter;
        this.dataSet = dataSet;
    }

    /**
     * Accessor for the classifier attribute.
     * @return Classifier attribute.
     */
    public Classifier getClassifier() {
        return classifier;
    }

    /**
     * Accessor for the parameter attribute.
     * @return Parameter attribute.
     */
    public Parameter getParameter() {
        return parameter;
    }

    /**
     * Accessor for the dataSet attribute.
     * @return DataSet attribute.
     */
    public DataSet getDataSet() {
        return dataSet;
    }

    /**
     * Construct and returns a feature selection experiment.
     * @param featureSubSet Feature subset used in the feature selection experiment
     * @return Experiment constructed
     */
    public Experiment featureSelectedExperiment(FeatureSubSet featureSubSet) {
        return new Experiment(classifier, parameter, dataSet.getSubSetOfFeatures(featureSubSet));
    }

}
