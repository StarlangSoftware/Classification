package Classification.Classifier;

import Classification.Model.Svm.KernelType;
import Classification.Parameter.SvmParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class SvmTest extends ClassifierTest{

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        Svm svm = new Svm();
        SvmParameter svmParameter = new SvmParameter(1, KernelType.LINEAR, 1, 0.0, 1.0, 1.0);
        svm.train(iris.getInstanceList(), svmParameter);
        assertEquals(0.66, 100 * svm.test(iris.getInstanceList()).getErrorRate(), 0.01);
        svmParameter = new SvmParameter(1, KernelType.LINEAR, 1, 0.0, 1.0, 0.1);
        svm.train(bupa.getInstanceList(), svmParameter);
        assertEquals(42.03, 100 * svm.test(bupa.getInstanceList()).getErrorRate(), 0.01);
    }

}