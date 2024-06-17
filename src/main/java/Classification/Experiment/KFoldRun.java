package Classification.Experiment;

import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Model.Model;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Parameter.Parameter;
import Sampling.CrossValidation;
import Sampling.KFoldCrossValidation;

public class KFoldRun implements MultipleRun {
    protected int K;

    /**
     * Constructor for KFoldRun class. Basically sets K parameter of the K-fold cross-validation.
     *
     * @param K K of the K-fold cross-validation.
     */
    public KFoldRun(int K) {
        this.K = K;
    }

    /**
     * Runs a K fold cross-validated experiment for the given classifier with the given parameters. The experiment
     * results will be added to the experimentPerformance.
     * @param model Model for the experiment
     * @param parameter Hyperparameters of the classifier of the experiment
     * @param experimentPerformance Storage to add experiment results
     * @param crossValidation K-fold crossvalidated dataset.
     * @throws DiscreteFeaturesNotAllowed If the classifier does not allow discrete features and the dataset contains
     * discrete features, DiscreteFeaturesNotAllowed will be thrown.
     */
    protected void runExperiment(Model model, Parameter parameter, ExperimentPerformance experimentPerformance, CrossValidation<Instance> crossValidation) throws DiscreteFeaturesNotAllowed {
        for (int i = 0; i < K; i++) {
            InstanceList trainSet = new InstanceList(crossValidation.getTrainFold(i));
            InstanceList testSet = new InstanceList(crossValidation.getTestFold(i));
            model.train(trainSet, parameter);
            experimentPerformance.add(model.test(testSet));
        }
    }

    /**
     * Execute K-fold cross-validation with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An ExperimentPerformance instance.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        KFoldCrossValidation<Instance> crossValidation = new KFoldCrossValidation<>(experiment.getDataSet().getInstances(), K, experiment.getParameter().getSeed());
        runExperiment(experiment.getModel(), experiment.getParameter(), result, crossValidation);
        return result;
    }
}
