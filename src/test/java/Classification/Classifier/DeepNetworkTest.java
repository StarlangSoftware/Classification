package Classification.Classifier;

import Classification.Parameter.ActivationFunction;
import Classification.Parameter.DeepNetworkParameter;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.*;

public class DeepNetworkTest extends ClassifierTest{

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        DeepNetwork deepNetwork = new DeepNetwork();
        DeepNetworkParameter deepNetworkParameter = new DeepNetworkParameter(1, 0.1, 0.99, 0.2, 100, new ArrayList<Integer>(Arrays.asList(5, 5)), ActivationFunction.SIGMOID);
        deepNetwork.train(iris.getInstanceList(), deepNetworkParameter);
        deepNetwork.getModel().saveTxt("models/deepNetwork-iris.txt");
        deepNetwork.loadModel("models/deepNetwork-iris.txt");
        assertEquals(1.33, 100 * deepNetwork.test(iris.getInstanceList()).getErrorRate(), 0.01);
        deepNetworkParameter = new DeepNetworkParameter(1, 0.01, 0.99, 0.2, 100, new ArrayList<Integer>(Arrays.asList(15, 15)), ActivationFunction.SIGMOID);
        deepNetwork.train(bupa.getInstanceList(), deepNetworkParameter);
        deepNetwork.getModel().saveTxt("models/deepNetwork-bupa.txt");
        deepNetwork.loadModel("models/deepNetwork-bupa.txt");
        assertEquals(28.99, 100 * deepNetwork.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        deepNetworkParameter = new DeepNetworkParameter(1, 0.01, 0.99, 0.2, 100, new ArrayList<Integer>(Arrays.asList(20)), ActivationFunction.SIGMOID);
        deepNetwork.train(dermatology.getInstanceList(), deepNetworkParameter);
        deepNetwork.getModel().saveTxt("models/deepNetwork-dermatology.txt");
        deepNetwork.loadModel("models/deepNetwork-dermatology.txt");
        assertEquals(1.09, 100 * deepNetwork.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

}