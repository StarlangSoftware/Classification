package Classification.Experiment;

import Classification.Classifier.*;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Parameter.*;
import Classification.Performance.ExperimentPerformance;
import org.junit.Test;

import static org.junit.Assert.*;

public class BootstrapRunTest extends ClassifierTest {

    @Test
    public void testExecute() throws DiscreteFeaturesNotAllowed {
        BootstrapRun bootstrapRun = new BootstrapRun(50);
        ExperimentPerformance experimentPerformance = bootstrapRun.execute(new Experiment(new C45(), new C45Parameter(1, true, 0.2), iris));
        assertEquals(4.16, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new C45(), new C45Parameter(1, true, 0.2), tictactoe));
        assertEquals(12.82, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new Knn(), new KnnParameter(1, 3, new EuclidianDistance()), bupa));
        assertEquals(24.23, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new Knn(), new KnnParameter(1, 3, new EuclidianDistance()), dermatology));
        assertEquals(8.43, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new Lda(), new Parameter(1), bupa));
        assertEquals(32.03, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new Lda(), new Parameter(1), dermatology));
        assertEquals(2.68, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new LinearPerceptron(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
        assertEquals(3.05, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new LinearPerceptron(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
        assertEquals(2.42, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new NaiveBayes(), new Parameter(1), car));
        assertEquals(14.14, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new NaiveBayes(), new Parameter(1), nursery));
        assertEquals(9.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new Bagging(), new BaggingParameter(1, 50), tictactoe));
        assertEquals(2.83, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new Bagging(), new BaggingParameter(1, 50), car));
        assertEquals(3.20, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new Dummy(), new Parameter(1), nursery));
        assertEquals(66.75, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
        experimentPerformance = bootstrapRun.execute(new Experiment(new Dummy(), new Parameter(1), iris));
        assertEquals(66.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 0.01);
    }
}