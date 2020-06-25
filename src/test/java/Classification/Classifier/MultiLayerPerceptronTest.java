package Classification.Classifier;

import Classification.Parameter.MultiLayerPerceptronParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class MultiLayerPerceptronTest extends ClassifierTest{

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron();
        MultiLayerPerceptronParameter multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.1, 0.99, 0.2, 100, 3);
        multiLayerPerceptron.train(iris.getInstanceList(), multiLayerPerceptronParameter);
        assertEquals(2.67, 100 * multiLayerPerceptron.test(iris.getInstanceList()).getErrorRate(), 0.01);
        multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 30);
        multiLayerPerceptron.train(bupa.getInstanceList(), multiLayerPerceptronParameter);
        assertEquals(28.12, 100 * multiLayerPerceptron.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 20);
        multiLayerPerceptron.train(dermatology.getInstanceList(), multiLayerPerceptronParameter);
        assertEquals(1.09, 100 * multiLayerPerceptron.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

}