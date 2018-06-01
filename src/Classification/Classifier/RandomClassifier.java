package Classification.Classifier;

import Classification.InstanceList.InstanceList;
import Classification.Model.RandomModel;
import Classification.Parameter.Parameter;

import java.util.ArrayList;

public class RandomClassifier extends Classifier{

	@Override
	public void train(InstanceList trainSet, Parameter parameters) {
		model = new RandomModel(new ArrayList<String>(trainSet.classDistribution().keySet()));
	}

}
