package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Performance.ExperimentPerformance;

public interface MultipleRun {
    ExperimentPerformance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed;
}
