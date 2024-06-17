package Classification.Filter;

import Classification.Classifier.*;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Model.DecisionTree.DecisionTree;
import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Model.KnnModel;
import Classification.Model.LinearPerceptronModel;
import Classification.Parameter.C45Parameter;
import Classification.Parameter.KnnParameter;
import Classification.Parameter.LinearPerceptronParameter;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DiscreteToIndexedTest extends ClassifierTest {

    @Test
    public void testLinearPerceptron() throws DiscreteFeaturesNotAllowed {
        LinearPerceptronModel linearPerceptron = new LinearPerceptronModel();
        LinearPerceptronParameter linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
        DiscreteToIndexed discreteToIndexed = new DiscreteToIndexed(car);
        discreteToIndexed.convert();
        linearPerceptron.train(car.getInstanceList(), linearPerceptronParameter);
        assertEquals(6.01, 100 * linearPerceptron.test(car.getInstanceList()).getErrorRate(), 0.01);
        discreteToIndexed = new DiscreteToIndexed(tictactoe);
        discreteToIndexed.convert();
        linearPerceptron.train(tictactoe.getInstanceList(), linearPerceptronParameter);
        assertEquals(1.67, 100 * linearPerceptron.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testKnn() {
        KnnModel knn = new KnnModel();
        KnnParameter knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
        DiscreteToIndexed discreteToIndexed = new DiscreteToIndexed(car);
        discreteToIndexed.convert();
        knn.train(car.getInstanceList(), knnParameter);
        assertEquals(21.06, 100 * knn.test(car.getInstanceList()).getErrorRate(), 0.01);
        discreteToIndexed = new DiscreteToIndexed(tictactoe);
        discreteToIndexed.convert();
        knn.train(tictactoe.getInstanceList(), knnParameter);
        assertEquals(32.57, 100 * knn.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testC45() {
        DecisionTree c45 = new DecisionTree();
        C45Parameter c45Parameter = new C45Parameter(1, true, 0.2);
        DiscreteToIndexed discreteToIndexed = new DiscreteToIndexed(car);
        discreteToIndexed.convert();
        c45.train(car.getInstanceList(), c45Parameter);
        assertEquals(2.78, 100 * c45.test(car.getInstanceList()).getErrorRate(), 0.01);
        discreteToIndexed = new DiscreteToIndexed(tictactoe);
        discreteToIndexed.convert();
        c45.train(tictactoe.getInstanceList(), c45Parameter);
        assertEquals(4.49, 100 * c45.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        discreteToIndexed = new DiscreteToIndexed(nursery);
        discreteToIndexed.convert();
        c45.train(nursery.getInstanceList(), c45Parameter);
        assertEquals(0.52, 100 * c45.test(nursery.getInstanceList()).getErrorRate(), 0.01);
    }

}