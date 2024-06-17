package Classification.Experiment;

import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Model.Model;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Parameter.Parameter;
import Sampling.CrossValidation;
import Sampling.KFoldCrossValidation;

import java.util.Random;

public class KFoldRunSeparateTest extends KFoldRun {
    /**
     * Constructor for KFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation.
     *
     * @param K K of the K-fold cross-validation.
     */
    public KFoldRunSeparateTest(int K) {
        super(K);
    }

    /**
     * Runs a K fold cross-validated experiment for the given classifier with the given parameters. Testing will be
     * done on the separate test set. The experiment results will be added to the experimentPerformance.
     * @param model Model for the experiment
     * @param parameter Hyperparameters of the classifier of the experiment
     * @param experimentPerformance Storage to add experiment results
     * @param crossValidation K-fold crossvalidated dataset.
     * @param testSet Test set on which experiment performance is calculated.
     * @throws DiscreteFeaturesNotAllowed If the classifier does not allow discrete features and the dataset contains
     * discrete features, DiscreteFeaturesNotAllowed will be thrown.
     */
    protected void runExperiment(Model model, Parameter parameter, ExperimentPerformance experimentPerformance, CrossValidation<Instance> crossValidation, InstanceList testSet) throws DiscreteFeaturesNotAllowed {
        for (int i = 0; i < K; i++) {
            InstanceList trainSet = new InstanceList(crossValidation.getTrainFold(i));
            model.train(trainSet, parameter);
            experimentPerformance.add(model.test(testSet));
        }
    }

    /**
     * Execute K-fold cross-validation with separate test set with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An ExperimentPerformance instance.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        InstanceList instanceList = experiment.getDataSet().getInstanceList();
        Partition partition = new Partition(instanceList, 0.25, new Random(experiment.getParameter().getSeed()), true);
        KFoldCrossValidation<Instance> crossValidation = new KFoldCrossValidation<>(partition.get(1).getInstances(), K, experiment.getParameter().getSeed());
        runExperiment(experiment.getModel(), experiment.getParameter(), result, crossValidation, partition.get(0));
        return result;
    }

}
