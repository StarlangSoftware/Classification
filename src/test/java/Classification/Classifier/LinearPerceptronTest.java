package Classification.Classifier;

import Classification.Parameter.LinearPerceptronParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class LinearPerceptronTest extends ClassifierTest{

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        LinearPerceptron linearPerceptron = new LinearPerceptron();
        LinearPerceptronParameter linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
        linearPerceptron.train(iris.getInstanceList(), linearPerceptronParameter);
        assertEquals(3.33, 100 * linearPerceptron.test(iris.getInstanceList()).getErrorRate(), 0.01);
        linearPerceptronParameter = new LinearPerceptronParameter(1, 0.001, 0.99, 0.2, 100);
        linearPerceptron.train(bupa.getInstanceList(), linearPerceptronParameter);
        assertEquals(31.88, 100 * linearPerceptron.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
        linearPerceptron.train(dermatology.getInstanceList(), linearPerceptronParameter);
        assertEquals(0.82, 100 * linearPerceptron.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

}