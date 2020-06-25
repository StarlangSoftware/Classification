package Classification.Filter;

import Classification.Classifier.*;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Parameter.C45Parameter;
import Classification.Parameter.KnnParameter;
import Classification.Parameter.LinearPerceptronParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class DiscreteToContinuousTest extends ClassifierTest {

    @Test
    public void testLinearPerceptron() throws DiscreteFeaturesNotAllowed {
        LinearPerceptron linearPerceptron = new LinearPerceptron();
        LinearPerceptronParameter linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
        DiscreteToContinuous discreteToContinuous = new DiscreteToContinuous(car);
        discreteToContinuous.convert();
        linearPerceptron.train(car.getInstanceList(), linearPerceptronParameter);
        assertEquals(6.01, 100 * linearPerceptron.test(car.getInstanceList()).getErrorRate(), 0.01);
        discreteToContinuous = new DiscreteToContinuous(tictactoe);
        discreteToContinuous.convert();
        linearPerceptron.train(tictactoe.getInstanceList(), linearPerceptronParameter);
        assertEquals(1.67, 100 * linearPerceptron.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        discreteToContinuous = new DiscreteToContinuous(nursery);
        discreteToContinuous.convert();
        linearPerceptron.train(nursery.getInstanceList(), linearPerceptronParameter);
        assertEquals(7.01, 100 * linearPerceptron.test(nursery.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testKnn() {
        Knn knn = new Knn();
        KnnParameter knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
        DiscreteToContinuous discreteToContinuous = new DiscreteToContinuous(car);
        discreteToContinuous.convert();
        knn.train(car.getInstanceList(), knnParameter);
        assertEquals(21.06, 100 * knn.test(car.getInstanceList()).getErrorRate(), 0.01);
        discreteToContinuous = new DiscreteToContinuous(tictactoe);
        discreteToContinuous.convert();
        knn.train(tictactoe.getInstanceList(), knnParameter);
        assertEquals(32.57, 100 * knn.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testC45() {
        C45 c45 = new C45();
        C45Parameter c45Parameter = new C45Parameter(1, true, 0.2);
        DiscreteToContinuous discreteToContinuous = new DiscreteToContinuous(car);
        discreteToContinuous.convert();
        c45.train(car.getInstanceList(), c45Parameter);
        assertEquals(29.98, 100 * c45.test(car.getInstanceList()).getErrorRate(), 0.01);
        discreteToContinuous = new DiscreteToContinuous(tictactoe);
        discreteToContinuous.convert();
        c45.train(tictactoe.getInstanceList(), c45Parameter);
        assertEquals(34.66, 100 * c45.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }

}