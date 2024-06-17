package Classification.Experiment;

import Classification.DataSet.DataSet;
import Classification.FeatureSelection.FeatureSubSet;
import Classification.Model.Model;
import Classification.Parameter.Parameter;

public class Experiment {
    private final Model model;
    private final Parameter parameter;
    private final DataSet dataSet;

    /**
     * Constructor for a specific machine learning experiment
     * @param model Model used in the machine learning experiment
     * @param parameter Parameter(s) of the classifier.
     * @param dataSet DataSet on which the classifier is run.
     */
    public Experiment(Model model, Parameter parameter, DataSet dataSet) {
        this.model = model;
        this.parameter = parameter;
        this.dataSet = dataSet;
    }

    /**
     * Accessor for the model attribute.
     * @return Model attribute.
     */
    public Model getModel() {
        return model;
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
        return new Experiment(model, parameter, dataSet.getSubSetOfFeatures(featureSubSet));
    }

}
