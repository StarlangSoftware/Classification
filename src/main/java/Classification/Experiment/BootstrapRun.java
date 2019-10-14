package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Instance.Instance;
import Classification.Performance.ExperimentPerformance;
import Classification.InstanceList.InstanceList;
import Sampling.Bootstrap;

public class BootstrapRun implements MultipleRun {
    private int numberOfBootstraps;

    /**
     * Constructor for BootstrapRun class. Basically sets the number of bootstrap runs.
     *
     * @param numberOfBootstraps Number of bootstrap runs.
     */
    public BootstrapRun(int numberOfBootstraps) {
        this.numberOfBootstraps = numberOfBootstraps;
    }

    /**
     * Execute the bootstrap run with the given classifier on the given data set using the given parameters.
     *
     * @param experiment Experiment to be run.
     * @return An ExperimentPerformance instance.
     */
    public ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance result = new ExperimentPerformance();
        for (int i = 0; i < numberOfBootstraps; i++) {
            Bootstrap<Instance> bootstrap = new Bootstrap<>(experiment.getDataSet().getInstances(), i + experiment.getParameter().getSeed());
            InstanceList bootstrapSample = new InstanceList(bootstrap.getSample());
            experiment.getClassifier().train(bootstrapSample, experiment.getParameter());
            result.add(experiment.getClassifier().test(experiment.getDataSet().getInstanceList()));
        }
        return result;
    }
}
