package Classification.Experiment;

import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.Model;
import Classification.Parameter.Parameter;
import Classification.Performance.Performance;
import Sampling.CrossValidation;
import Sampling.KFoldCrossValidation;

public class SingleRunWithK implements SingleRun {
    private final int K;

    /**
     * Constructor for SingleRunWithK class. Basically sets K parameter of the K-fold cross-validation.
     *
     * @param K K of the K-fold cross-validation.
     */
    public SingleRunWithK(int K) {
        this.K = K;
    }

    /**
     * Runs first fold of a K fold cross-validated experiment for the given classifier with the given parameters.
     * The experiment result will be returned.
     * @param model Model for the experiment
     * @param parameter Hyperparameters of the classifier of the experiment
     * @param crossValidation K-fold crossvalidated dataset.
     * @return The experiment result of the first fold of the K-fold cross-validated experiment.
     * @throws DiscreteFeaturesNotAllowed If the classifier does not allow discrete features and the dataset contains
     * discrete features, DiscreteFeaturesNotAllowed will be thrown.
     */
    protected Performance runExperiment(Model model, Parameter parameter, CrossValidation<Instance> crossValidation) throws DiscreteFeaturesNotAllowed {
        InstanceList trainSet = new InstanceList(crossValidation.getTrainFold(0));
        InstanceList testSet = new InstanceList(crossValidation.getTestFold(0));
        return model.singleRun(parameter, trainSet, testSet);
    }


    /**
     * Execute Single K-fold cross-validation with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return A Performance instance
     */
    public Performance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        KFoldCrossValidation<Instance> crossValidation = new KFoldCrossValidation<>(experiment.getDataSet().getInstances(), K, experiment.getParameter().getSeed());
        return runExperiment(experiment.getModel(), experiment.getParameter(), crossValidation);
    }
}