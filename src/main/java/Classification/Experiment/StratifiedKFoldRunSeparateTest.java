package Classification.Experiment;

import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Sampling.StratifiedKFoldCrossValidation;

import java.util.Random;

public class StratifiedKFoldRunSeparateTest extends KFoldRunSeparateTest{
    /**
     * Constructor for StratifiedKFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation.
     *
     * @param K K of the K-fold cross-validation.
     */
    public StratifiedKFoldRunSeparateTest(int K) {
        super(K);
    }

    /**
     * Execute Stratified K-fold cross-validation with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An ExperimentPerformance instance.
     * @throws DiscreteFeaturesNotAllowed Exception for discrete features.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        InstanceList instanceList = experiment.getDataSet().getInstanceList();
        Partition partition = new Partition(instanceList, 0.25, new Random(experiment.getParameter().getSeed()), true);
        StratifiedKFoldCrossValidation<Instance> crossValidation = new StratifiedKFoldCrossValidation<>(new Partition(partition.get(1)).getLists(), K, experiment.getParameter().getSeed());
        runExperiment(experiment.getModel(), experiment.getParameter(), result, crossValidation, partition.get(0));
        return result;
    }

}
