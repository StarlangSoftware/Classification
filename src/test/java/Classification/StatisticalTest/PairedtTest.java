package Classification.StatisticalTest;

import Classification.Classifier.*;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Experiment.Experiment;
import Classification.Experiment.KFoldRun;
import Classification.Parameter.*;
import Classification.Performance.ExperimentPerformance;
import org.junit.Test;

import static org.junit.Assert.*;

public class PairedtTest extends ClassifierTest {

    @Test
    public void testCompare() throws DiscreteFeaturesNotAllowed, StatisticalTestNotApplicable {
        KFoldRun kFoldRun = new KFoldRun(10);
        ExperimentPerformance experimentPerformance1 = kFoldRun.execute(new Experiment(new C45(), new C45Parameter(1, true, 0.2), iris));
        ExperimentPerformance experimentPerformance2 = kFoldRun.execute(new Experiment(new LinearPerceptron(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
        Pairedt pairedt = new Pairedt();
        assertEquals(0.070, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 0.001);
        experimentPerformance1 = kFoldRun.execute(new Experiment(new C45(), new C45Parameter(1, true, 0.2), tictactoe));
        experimentPerformance2 = kFoldRun.execute(new Experiment(new Bagging(), new BaggingParameter(1, 50), tictactoe));
        assertEquals(0.0000037, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 0.0000001);
        experimentPerformance1 = kFoldRun.execute(new Experiment(new Lda(), new Parameter(1), dermatology));
        experimentPerformance2 = kFoldRun.execute(new Experiment(new LinearPerceptron(), new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
        assertEquals(0.7797, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 0.0001);
        experimentPerformance1 = kFoldRun.execute(new Experiment(new Dummy(), new Parameter(1), nursery));
        experimentPerformance2 = kFoldRun.execute(new Experiment(new NaiveBayes(), new Parameter(1), nursery));
        assertEquals(0.0, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 0.0001);
        experimentPerformance1 = kFoldRun.execute(new Experiment(new NaiveBayes(), new Parameter(1), car));
        experimentPerformance2 = kFoldRun.execute(new Experiment(new Bagging(), new BaggingParameter(1, 50), car));
        assertEquals(0.0000049, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 0.0000001);
        experimentPerformance1 = kFoldRun.execute(new Experiment(new Knn(), new KnnParameter(1, 3, new EuclidianDistance()), bupa));
        experimentPerformance2 = kFoldRun.execute(new Experiment(new Lda(), new Parameter(1), bupa));
        assertEquals(0.0960, pairedt.compare(experimentPerformance1, experimentPerformance2).getPValue(), 0.0001);
    }
}