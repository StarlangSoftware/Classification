package Classification.Experiment;

import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Performance.Performance;

public interface SingleRun {
    Performance execute(Experiment experiment) throws DiscreteFeaturesNotAllowed;
}
