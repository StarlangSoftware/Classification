package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
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

    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        InstanceList instanceList = experiment.getDataSet().getInstanceList();
        Partition partition = instanceList.partition(0.25, new Random(experiment.getParameter().getSeed()));
        StratifiedKFoldCrossValidation<Instance> crossValidation = new StratifiedKFoldCrossValidation<>(partition.get(1).divideIntoClasses().getLists(), K, experiment.getParameter().getSeed());
        runExperiment(experiment.getClassifier(), experiment.getParameter(), result, crossValidation, partition.get(0));
        return result;
    }

}
