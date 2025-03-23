package Classification.Classifier;

import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Model.NeuralNetwork.MultiLayerPerceptronModel;
import Classification.Parameter.ActivationFunction;
import Classification.Parameter.MultiLayerPerceptronParameter;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class MultiLayerPerceptronTest extends ClassifierTest{

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        MultiLayerPerceptronModel multiLayerPerceptron = new MultiLayerPerceptronModel();
        MultiLayerPerceptronParameter multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.1, 0.99, 0.2, 100, 3, ActivationFunction.SIGMOID);
        multiLayerPerceptron.train(iris.getInstanceList(), multiLayerPerceptronParameter);
        multiLayerPerceptron.saveTxt("models/multiLayerPerceptron-iris.txt");
        multiLayerPerceptron.loadModel("models/multiLayerPerceptron-iris.txt");
        assertEquals(2.67, 100 * multiLayerPerceptron.test(iris.getInstanceList()).getErrorRate(), 0.01);
        multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 30, ActivationFunction.SIGMOID);
        multiLayerPerceptron.train(bupa.getInstanceList(), multiLayerPerceptronParameter);
        multiLayerPerceptron.saveTxt("models/multiLayerPerceptron-bupa.txt");
        multiLayerPerceptron.loadModel("models/multiLayerPerceptron-bupa.txt");
        assertEquals(29.57, 100 * multiLayerPerceptron.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 20, ActivationFunction.SIGMOID);
        multiLayerPerceptron.train(dermatology.getInstanceList(), multiLayerPerceptronParameter);
        multiLayerPerceptron.saveTxt("models/multiLayerPerceptron-dermatology.txt");
        multiLayerPerceptron.loadModel("models/multiLayerPerceptron-dermatology.txt");
        assertEquals(1.37, 100 * multiLayerPerceptron.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

}